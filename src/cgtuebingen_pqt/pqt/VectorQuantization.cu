#include "VectorQuantization.hh"

#define OUTPUT
#include "helper.hh"

namespace pqt {

/** default constructor */

VectorQuantization::VectorQuantization(uint _dim) :
		d_dim(_dim), d_codeBook(NULL) {
}

VectorQuantization::~VectorQuantization() {
	if (d_codeBook)
		cudaFree(d_codeBook);
}

//__device__ float sqr(const float &x) {
//	return x * x;
//}

/** for all vectors in A compute distance to all vectors in B of dimension _dim */

__global__ void calcDistKernel(float* _res, const float * _A, const float * _B,
		uint _Arows, uint _Brows, uint _dim) {

	extern __shared__ float shm[];

	float *Ablock = shm;
	float *Bblock = shm + blockDim.x * blockDim.y;
	float *AB = Bblock + blockDim.x * blockDim.y;

	uint id = threadIdx.x + threadIdx.y * blockDim.x;
	uint arow = threadIdx.y + blockIdx.y * blockDim.y;
	uint brow = threadIdx.y + blockIdx.x * blockDim.x;
	uint ocol = threadIdx.x + blockIdx.x * blockDim.x;

	uint AOffs = threadIdx.y * blockDim.x;
	uint BOffs = threadIdx.x * blockDim.x;

	AB[id] = 0.;

	int j = blockIdx.z;
	{
//	for (int j = 0; j < _Acols; j += blockDim.x) {
		// load block of A and B
		uint col = threadIdx.x + j * blockDim.x;

		Bblock[id] = 0.;
		Ablock[id] = 0.;
		if (col < _dim) {
			if (brow < _Brows)
				Bblock[id] = _B[brow * _dim + col];
			if (arow < _Arows)
				Ablock[id] = _A[arow * _dim + col];
		}
		__syncthreads();

//		if ((col < _Acols) && (arow < _Arows))
//			printf(" A B %i %f %f \n", id, Ablock[id], Bblock[id]);

		// compute partial differences
		for (int i = 0; i < blockDim.x; i++) {
			AB[id] += sqr(Ablock[AOffs + i] - Bblock[BOffs + i]);
		}
		__syncthreads();

	}

// write out the result
	if ((arow < _Arows) && (ocol < _Brows)) {
		//	_res[arow][ocol] += AB[id];
		atomicAdd(_res + (arow * _Brows + ocol), AB[id]);
//			printf(" AB %i %i %i %f \n", id, arow, ocol, AB[id]);
	}

}

void VectorQuantization::calcDist(float* _res, const float* _A, const float* _B,
		uint _Arows, uint _Brows, uint _dim) const {

	const uint blockSize = 16;

	dim3 block(blockSize, blockSize, 1);

	uint shmSize = (blockSize * blockSize * 3) * sizeof(float);
//std::cout << "requested shm: " << shmSize << std::endl;

	cudaMemset(_res, 0, _Arows * _Brows * sizeof(float));

	dim3 grid(idiv(_Brows, blockSize), idiv(_Arows, blockSize),
			idiv(_dim, blockSize));

	calcDistKernel<<<grid, block, shmSize>>>(_res, _A, _B, _Arows, _Brows,
			_dim);
	checkCudaErrors(cudaDeviceSynchronize());

//	outputMat("dist: ", _res, 10, 10);

}

/** blockd Id.x corresponds to the cluster center, blockId.y is used to span multiple kernels
 * will update the codebook vector of this center. As each y block is only adding some of the input vectors the last block is normalizing the vector
 */__global__ void avgClusterKernel(float* _codebook, float * _count,
		uint *_retirementCount, uint _yElem, uint _dim, const float * _A,
		uint _N, uint* _assignd) {

	__shared__ bool amLast;
	extern __shared__ float shm[];

	float count = 0;

	for (int i = threadIdx.x; i < _dim; i += blockDim.x) {
		shm[i] = 0.;
	}

	int stop = (blockIdx.y + 1) * _yElem;
	stop = (stop < _N) ? stop : _N;
	// accumulate the vectors that belong to this cluster center
	for (int n = blockIdx.y * _yElem; n < stop; n++) {
		if (_assignd[n] == blockIdx.x) {
			const float *v = _A + n * _dim;
			for (int i = threadIdx.x; i < _dim; i += blockDim.x) {
				shm[i] += v[i];
			}
			if (threadIdx.x == 0)
				count++;
		}
	}

	// store the result
	__syncthreads();
	for (int i = threadIdx.x; i < _dim; i += blockDim.x) {
		atomicAdd(_codebook + blockIdx.x * _dim + i, shm[i]);
	}
	__threadfence();

	if (threadIdx.x == 0) {
		atomicAdd(_count + blockIdx.x, count);
		uint ticket = atomicInc(_retirementCount + blockIdx.x, gridDim.y);
		// If the ticket ID is equal to the number of blocks, we are the last block!
		amLast = (ticket == gridDim.y - 1);
	}
	__syncthreads();

	// the last block is responsible for dividing by the number of vectors added to this center
	if (amLast) {
		for (int i = threadIdx.x; i < _dim; i += blockDim.x) {
			_codebook[blockIdx.x * _dim + i] /= _count[blockIdx.x];
		}
		// reset retirement count for next iteration
		if (threadIdx.x == 0) {
			_retirementCount[blockIdx.x] = 0;
		}
	}
}

/** compute the maximum Radius for each cluster, one block per cluster
 */__global__ void maxRadKernel(float* _maxRad, const uint* _assign, uint _N,
		const float* _distMat) {
	extern __shared__ float shm[];

	float* sharedMax = shm;

	uint c = blockIdx.x;

	float vMax = 0.;

	for (uint i = threadIdx.x; i < _N; i += blockDim.x) {
		if (_assign[i] == c) {
			float vMax2 = _distMat[i* gridDim.x + c];
			if (vMax2 > vMax) {
				vMax = vMax2;
			}
		}
	}

	sharedMax[threadIdx.x] = vMax;

	for (uint stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
		__syncthreads();
		if (threadIdx.x < stride) {
			float vMax2 = sharedMax[threadIdx.x + stride];
			if (vMax2 > vMax) {
				vMax = vMax2;
				sharedMax[threadIdx.x] = vMax;
			}
		}
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		_maxRad[blockIdx.x] = sharedMax[0];
	}
}


void VectorQuantization::getMaxRad(float *_maxRad, uint _nCluster, const uint* _assign, uint _N, const float* _distMat) const {

	dim3 block(256, 1, 1);
	dim3 grid(_nCluster, 1, 1);

	uint shmsize = block.x * sizeof(float);

	maxRadKernel<<<grid, block, shmsize>>>(_maxRad, _assign, _N, _distMat);

	outputVec( "maxDist", _maxRad, _nCluster);
}


__global__ void assignKernel(uint* _assign, uint _N, const float* _distMat,
		uint _nClusters) {
	extern __shared__ float shm[];

	float* sharedMin = shm;
	uint* sharedIdx = (uint*) (shm + blockDim.x);

	for (int row = blockIdx.x; row < _N; row += gridDim.x) {
		// initialize with first element
		const float* matRow = _distMat + row * _nClusters;
		float vMin = matRow[0];
		uint minIdx = 0;

		for (uint i = threadIdx.x; i < _nClusters; i += blockDim.x) {
			float vMin2 = matRow[i];
			if (vMin2 < vMin) {
				vMin = vMin2;
				minIdx = i;
			}
		}

		sharedMin[threadIdx.x] = vMin;
		sharedIdx[threadIdx.x] = minIdx;

		for (uint stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
			__syncthreads();
			if (threadIdx.x < stride) {
				float vMin2 = sharedMin[threadIdx.x + stride];
				if (vMin2 < vMin) {
					vMin = vMin2;
					sharedMin[threadIdx.x] = vMin;
					sharedIdx[threadIdx.x] = sharedIdx[threadIdx.x + stride];
				}
			}
		}
		__syncthreads();

		if (threadIdx.x == 0) {
			_assign[row] = sharedIdx[0];
		}
	}
}

void VectorQuantization::getAssignment(uint* _assignd, const float* _distMat,
		uint _N, uint _nClusters) const {
	// perform a minimum reduction per vector _N
	dim3 block;
	setReductionBlocks(block, _nClusters);

	uint nBlocks = (_N < MAX_BLOCKS) ? _N : MAX_BLOCKS;
	dim3 grid(nBlocks, 1, 1);

	uint shmSize = block.x * 2 * sizeof(float);
	assignKernel<<<grid, block, shmSize>>>(_assignd, _N, _distMat, _nClusters);
	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void splitVectorKernel(float* _codeBook, uint _dim, uint _nClusters,
		float _epsilon) {

	uint idx = blockIdx.x * _dim + threadIdx.x;
	float orig = _codeBook[idx];

	_codeBook[idx] = orig * (1. + _epsilon);

	idx += _nClusters * _dim;

	_codeBook[idx] = orig * (1. - _epsilon);

}

void VectorQuantization::splitCodeBook(uint &_nClusters, float _epsilon) {

	dim3 block(d_dim, 1, 1);

	dim3 grid(_nClusters, 1, 1);

	splitVectorKernel<<<grid, block>>>(d_codeBook, d_dim, _nClusters, _epsilon);

	_nClusters *= 2;

}

void VectorQuantization::createCodeBook(uint _k, const float* _A, uint _N) {

	uint *assign = new uint[_N];
	uint *old_assign = new uint[_N];

	uint *assignd;
	float* countd;
	uint* retirementCountd;
	float* distd;
	float* maxRadd;

	cudaMalloc(&assignd, _N * sizeof(uint));

	cudaMalloc(&d_codeBook, _k * d_dim * sizeof(float));
	cudaMalloc(&countd, _k * sizeof(float));
	cudaMalloc(&retirementCountd, _k * sizeof(uint));
	cudaMalloc(&distd, _k * _N * sizeof(float));
	cudaMalloc(&maxRadd, _k * sizeof(float));

	uint nClusters = 1;
	// initialize to get the first cluster average
	cudaMemset(assignd, 0, _N * sizeof(uint));

	cudaMemset(retirementCountd, 0, _k * sizeof(uint));
	cudaMemset(countd, 0, _k * sizeof(int));
	cudaMemset(d_codeBook, 0, d_dim * sizeof(float));

	uint yElem = 16;

	dim3 block(d_dim, 1, 1);
	dim3 grid(nClusters, idiv(_N, yElem), 1);

	size_t shmSize = d_dim * sizeof(float);
	avgClusterKernel<<<grid, block, shmSize>>>(d_codeBook, countd,
			retirementCountd, yElem, d_dim, _A, _N, assignd);

	float epsilon = 0.0001;

	while (nClusters < _k) {

		splitCodeBook(nClusters, epsilon);
		std::cout << "nClusters" << nClusters << std::endl;

		uint converged = 0;

		do {

			cudaMemset(countd, 0, _k * sizeof(int));

			calcDist(distd, _A, d_codeBook, _N, nClusters, d_dim);

			getAssignment(assignd, distd, _N, nClusters);

//			getMaxRad(maxRadd, nClusters, assignd, _N, distd);
			//outputVecUint("Assign", assignd, _N);

			uint yElem = 256;

			dim3 block(d_dim, 1, 1);
			dim3 grid(nClusters, idiv(_N, yElem), 1);

			size_t shmSize = d_dim * sizeof(float);
			avgClusterKernel<<<grid, block, shmSize>>>(d_codeBook, countd,
					retirementCountd, yElem, d_dim, _A, _N, assignd);

			std::cout << nClusters << std::endl;
//			outputVec("count:", countd, nClusters);
			//outputVec("avg: ", d_codeBook, d_dim);

			cudaMemcpy(assign, assignd, _N * sizeof(uint),
					cudaMemcpyDeviceToHost);
			converged = 0;
			for (int i = 0; i < _N; i++) {
				if (assign[i] != old_assign[i]) {
					converged++;
				}
			}
			memcpy(old_assign, assign, _N * sizeof(uint));

			std::cout << "non- converged" << converged << std::endl;
		} while (converged > 0.001 * _N);

		// outputMat("dist:", distd, _N, nClusters);
		// outputMat("codebook", d_codeBook, nClusters, d_dim );

		getMaxRad(maxRadd, nClusters, assignd, _N, distd);


	}

	cudaFree(distd);
	cudaFree(countd);
	cudaFree(retirementCountd);
	cudaFree(assignd);

	delete[] old_assign;
	delete[] assign;
}
} /* namespace */

