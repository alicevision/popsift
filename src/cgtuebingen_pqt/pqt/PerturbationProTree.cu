#include "PerturbationProTree.hh"
#include "sortingNetworks_common.h"
#include "bitonicSort.cuh"
#include "triangle.cuh"
#include "helper.hh"

#include <algorithm>
#include <iostream>
#include <fstream>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <sys/stat.h>


namespace pqt {


PerturbationProTree::PerturbationProTree(uint _dim, uint _p, uint _p2) :
		ProTree(_dim, _p, _p2), d_multiCodeBook(NULL), d_multiCodeBook2(NULL), d_codeBookDistL2(
		NULL), d_codeBookDistL1L2(NULL), d_lineIdx(NULL), d_lineLambda(
		NULL), d_l2Idx(NULL), d_lineParts(0) {

	int y;
	for (y = 0; y < 64; y++)
		if (!((1 - 1) >> y))
			break;
	d_dimBits = y - 1;

	d_dimBits = 7 - 1;

	std::cout << "dimBits " << d_dimBits << std::endl;

	d_nDBs = 1;

}

PerturbationProTree::~PerturbationProTree() {

	if (d_lineIdx)
		cudaFree(d_lineIdx);
	if (d_lineLambda)
		cudaFree(d_lineLambda);
	if (d_l2Idx)
		cudaFree(d_l2Idx);

	if (d_codeBookDistL1L2)
		cudaFree(d_codeBookDistL1L2);

	if (d_codeBookDistL2)
		cudaFree(d_codeBookDistL2);

	if (d_multiCodeBook)
		cudaFree(d_multiCodeBook);

	if (d_multiCodeBook2)
		cudaFree(d_multiCodeBook2);

}

void PerturbationProTree::writeTreeToFile(const std::string& _name) {

	std::ofstream f(_name.c_str(), std::ofstream::out | std::ofstream::binary);

	f << d_dim << std::endl;
	f << d_p << std::endl;
	f << d_p2 << std::endl;
	f << d_nClusters << std::endl;
	f << d_nClusters2 << std::endl;
	f << d_nDBs << std::endl;

	float * cb1Host = new float[d_nDBs * d_nClusters * d_dim];
	float * cb2Host = new float[d_nDBs * d_nClusters * d_nClusters2 * d_dim];

	cudaMemcpy(cb1Host, d_multiCodeBook,
			d_nDBs * d_nClusters * d_dim * sizeof(float),
			cudaMemcpyDeviceToHost);
	cudaMemcpy(cb2Host, d_multiCodeBook2,
			d_nDBs * d_nClusters * d_nClusters2 * d_dim * sizeof(float),
			cudaMemcpyDeviceToHost);

	char* cc = (char*) cb1Host;
	for (int i = 0; i < 10; i++) {
		std::cout << int(cc[i]) << " ";
	}
	std::cout << std::endl;

	std::cout << "cb1[12]: " << cb1Host[12] << std::endl;
	std::cout << "cb2[12]: " << cb2Host[12] << std::endl;

	f.write((char*) cb1Host, d_nDBs * d_nClusters * d_dim * sizeof(float));
	f.write((char*) cb2Host,
			d_nDBs * d_nClusters * d_nClusters2 * d_dim * sizeof(float));

	f.close();

	delete[] cb2Host;
	delete[] cb1Host;

	if (d_sparseBin) {
		uint nBins = pow(d_nClusters, d_p);
		ofstream fs((_name + string("_sparse")).c_str(),
				std::ofstream::out | std::ofstream::binary);

		uint* sparseHost = new uint[nBins];

		cudaMemcpy(sparseHost, d_sparseBin, nBins * sizeof(uint),
				cudaMemcpyDeviceToHost);

		fs.write((char*) sparseHost, nBins * sizeof(uint));

		fs.close();

		delete[] sparseHost;
	}

}

void PerturbationProTree::readTreeFromFile(const std::string& _name) {

	ifstream f(_name.c_str(), std::ofstream::in | std::ofstream::binary);

	f >> d_dim;
	f >> d_p;
	f >> d_p2;
	f >> d_nClusters;
	f >> d_nClusters2;
	f >> d_nDBs;

	f.ignore(1);

	std::cout << d_dim << std::endl;
	std::cout << d_p << std::endl;
	std::cout << d_p2 << std::endl;
	std::cout << d_nClusters << std::endl;
	std::cout << d_nClusters2 << std::endl;
	std::cout << d_nDBs << std::endl;

	d_vl = d_dim / d_p;
	d_vl2 = d_dim / d_p2;

	if (d_multiCodeBook)
		cudaFree(d_multiCodeBook);

	if (d_multiCodeBook2)
		cudaFree(d_multiCodeBook2);

	if (d_distSeq)
		cudaFree(d_distSeq);

	float * cb1Host = new float[d_nDBs * d_nClusters * d_dim];
	float * cb2Host = new float[d_nDBs * d_nClusters * d_nClusters2 * d_dim];

	cudaMalloc(&d_multiCodeBook, d_nDBs * d_nClusters * d_dim * sizeof(float));
	cudaMalloc(&d_multiCodeBook2,
			d_nDBs * d_nClusters * d_nClusters2 * d_dim * sizeof(float));

	f.read((char*) cb1Host, d_nDBs * d_nClusters * d_dim * sizeof(float));
	f.read((char*) cb2Host,
			d_nDBs * d_nClusters * d_nClusters2 * d_dim * sizeof(float));

	char* cc = (char*) cb1Host;
	for (int i = 0; i < 10; i++) {
		std::cout << int(cc[i]) << " ";
	}
	std::cout << std::endl;

	std::cout << "cb perturbation 0:" << std::endl;
	for (int i = 0; i < 100; i++)
		std::cout << "\t" << cb1Host[i];
	std::cout << std::endl;
	std::cout << std::endl;

	std::cout << "cb perturbation 1: " << std::endl;
	for (int i = 0; i < 100; i++)
		std::cout << "\t" << cb1Host[i + d_nClusters * d_dim];
	std::cout << std::endl;
	std::cout << std::endl;

	std::cout << "cb1[12]: " << cb1Host[12] << std::endl;
	std::cout << "cb2[12]: " << cb2Host[12] << std::endl;

	cudaMemcpy(d_multiCodeBook, cb1Host,
			d_nDBs * d_nClusters * d_dim * sizeof(float),
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_multiCodeBook2, cb2Host,
			d_nDBs * d_nClusters * d_nClusters2 * d_dim * sizeof(float),
			cudaMemcpyHostToDevice);

	checkCudaErrors(cudaDeviceSynchronize());

	f.close();

	prepareDistSequence(d_nClusters2 * NUM_NEIGHBORS, d_groupParts);

	delete[] cb2Host;
	delete[] cb1Host;

	string sparseName(_name + string("_sparse"));

	struct stat buffer;

	if (stat(sparseName.c_str(), &buffer) == 0) {
		uint nBins = pow(d_nClusters, d_p);
		ifstream fs((_name + string("_sparse")).c_str(),
				std::ifstream::in | std::ifstream::binary);

		uint* sparseHost = new uint[nBins];

		fs.read((char*) sparseHost, nBins * sizeof(uint));

		cudaMalloc(&d_sparseBin, nBins * sizeof(uint));
		cudaMemcpy(d_sparseBin, sparseHost, nBins * sizeof(uint),
				cudaMemcpyHostToDevice);

		fs.close();

		delete[] sparseHost;
	}

}

__device__ uint pertIdx(uint _i, uint _dimBits, uint _cb) {

	//!! TODO
//	_cb = 1;

	if (_cb == 0)
		return _i;

//	if (_i ==0) printf("bits: %d \n", _dimBits);

	_cb -= 1;

	uint maxBit = _i >> _dimBits;
	uint mask = (1 << _dimBits) - 1;
	uint remain = _i & mask;

	mask = (1 << _cb) - 1;

	return (maxBit << _cb) + ((remain >> _cb) << (_cb + 1)) + (remain & mask);

}

__global__ void perturbationKernel(float* _pertA, const float* _A, uint _N,
		uint _dimBits, uint _pert) {

	extern __shared__ float shm[];

	for (int iter = blockIdx.x; iter < _N; iter += gridDim.x) {
		__syncthreads();

		shm[threadIdx.x] = _A[iter * blockDim.x + threadIdx.x];

		uint pIdx = pertIdx(threadIdx.x, _dimBits, _pert);

		__syncthreads();

		_pertA[iter * blockDim.x + threadIdx.x] = shm[pIdx];

	}
}

void PerturbationProTree::perturbVectors(float* _pertA, const float* _A,
		uint _N, uint _pert) {
	dim3 block(d_dim, 1, 1);
	dim3 grid((_N < 1024) ? _N : 1024, 1, 1);

	uint shm = (d_dim) * sizeof(float);
	perturbationKernel<<<grid, block, shm>>>(_pertA, _A, _N, d_dimBits, _pert);

	checkCudaErrors(cudaDeviceSynchronize());
}

void PerturbationProTree::createTree(uint _nClusters1, uint _nClusters2,
		const float* _A, uint _N) {

	if (!d_multiCodeBook)
		cudaMalloc(&d_multiCodeBook,
				d_nDBs * d_p * _nClusters1 * d_vl * sizeof(float));
	if (!d_multiCodeBook2)
		cudaMalloc(&d_multiCodeBook2,
				d_nDBs * d_p * _nClusters1 * _nClusters2 * d_vl
						* sizeof(float));

	float* pertA;
	cudaMalloc(&pertA, _N * d_dim * sizeof(float));

	

	ProTree::createTree(_nClusters1, _nClusters2, pertA, _N);

	cudaMemcpy(d_multiCodeBook , d_codeBook, _nClusters1 * d_dim * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_multiCodeBook2 , d_codeBook2, _nClusters1 * _nClusters2 * d_dim * sizeof(float), cudaMemcpyDeviceToDevice);

	

	checkCudaErrors(cudaDeviceSynchronize());

	cudaFree(pertA);
	cudaFree(d_codeBook);
	cudaFree(d_codeBook2);

}

void PerturbationProTree::createTreeSplitSparse(uint _nClusters1,
		uint _nClusters2, const float* _A, uint _N, bool _sparse) {

	if (!d_multiCodeBook)
		cudaMalloc(&d_multiCodeBook,
				d_nDBs * d_p * _nClusters1 * d_vl * sizeof(float));
	if (!d_multiCodeBook2)
		cudaMalloc(&d_multiCodeBook2,
				d_nDBs * d_p * _nClusters1 * _nClusters2 * d_vl
						* sizeof(float));

	float* pertA;
	cudaMalloc(&pertA, _N * d_dim * sizeof(float));

	for (int pert = 0; pert < d_nDBs; pert++) {

		perturbVectors(pertA, _A, _N, pert);

//		ProTree::createTree(_nClusters1, _nClusters2, pertA, _N);
		ProTree::createTreeSplitSparse(_nClusters1, _nClusters2, pertA, _N, 0.3,
				_sparse);

		cudaMemcpy(d_multiCodeBook + d_dim * _nClusters1 * pert, d_codeBook,
				_nClusters1 * d_dim * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_multiCodeBook2 + d_dim * _nClusters1 * _nClusters2 * pert,
				d_codeBook2, _nClusters1 * _nClusters2 * d_dim * sizeof(float),
				cudaMemcpyDeviceToDevice);

	}

	checkCudaErrors(cudaDeviceSynchronize());

	cudaFree(pertA);
	cudaFree(d_codeBook);
	cudaFree(d_codeBook2);

}

// each block is responsible for one vector, blockDim.x should be _dim
// requires _dim * float shared memory
// looping multiple times to process all B vectors
// _vl is the length of the _p vector segments (should be 2^n)
__global__ void assignPerturbationClusterKernel(uint *_assign, const float* _A,
		const float* _B, uint _Arows, uint _Brows, uint _dim, uint _p, uint _vl,
		uint _dimBits) {

	extern __shared__ float shmb[];

	float *shm = shmb + _dim;

	float minVal;
	uint minIdx;

//	if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
//		for (int i = 0; i < _dim; i++)
//			printf("%d: %d %d %d \n", i, pertIdx(i, _dimBits, 0),
//					pertIdx(i, _dimBits, 1), pertIdx(i, _dimBits, 2)) ;
//	}
//

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		__syncthreads();

		// load the vector to shared mem
		shmb[threadIdx.x] = _B[iter * _dim + threadIdx.x];

		__syncthreads();

		for (uint pert = 0; pert < 1; pert++) {

			if (threadIdx.x < _p) {
				minVal = 10000000.;
				minIdx = 0;
			}

			uint pIdx = pertIdx(threadIdx.x, _dimBits, pert);
			// load perturbed vector
			float b = shmb[pIdx];

			const float* A = _A + pert * _Arows * _dim;

			// loop over all vectors of A
			for (int a = 0; a < _Arows; a++) {
				shm[threadIdx.x] = sqr(b - A[a * _dim + threadIdx.x]);

				// compute the sum of differences for the vector segments
				// i.e. all _p segments in parallel
				for (uint stride = _vl >> 1; stride > 0; stride >>= 1) {
					__syncthreads();
					if ((threadIdx.x) < _p * stride) {
						uint p = threadIdx.x / stride * _vl;
						uint bias = threadIdx.x % stride;
						shm[p + bias] += shm[p + bias + stride];
					}
				}
				__syncthreads();

				if (threadIdx.x < _p) {

					// select the minimum of each segment
					float val = shm[threadIdx.x * _vl];
					if (val < minVal) {
						minVal = val;
						minIdx = a;
					}
				}

				__syncthreads();
			}

			// write out decision
			if (threadIdx.x < _p) {
				_assign[(iter * 1 + pert) * _p + threadIdx.x] =
						minIdx;
			}

		} // perturbation

	} // iter
}

void PerturbationProTree::getAssignment(uint *_assign, const float* _A,
		const float* _B, uint _Arows, uint _Brows) const {

	dim3 block(d_dim, 1, 1);
	dim3 grid((_Brows < 1024) ? _Brows : 1024, 1, 1);

	uint shm = 2 * d_dim * sizeof(float);
	assignPerturbationClusterKernel<<<grid, block, shm>>>(_assign, _A, _B,
			_Arows, _Brows, d_dim, d_p, d_vl, d_dimBits);

	checkCudaErrors(cudaDeviceSynchronize());
}

// each block is responsible for one vector, blockDim.x should be _dim
// requires _dim * float shared memory
// looping multiple times to process all B vectors
// _vl is the length of the _p vector segments (should be 2^n)
__global__ void assignPerturbationClusterKernel2(uint *_assign,
		const float* _cb2, const float* _B, uint _nClusters2, uint _Brows,
		uint _dim, uint _p, uint _vl, const uint* _assign1, uint _nClusters1,
		uint _dimBits) {

	extern __shared__ float shmb[];

	float* shm = shmb + _dim;
	uint* code1 = (uint*) shm + _dim;

	float minVal;
	uint minIdx;

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		__syncthreads();

		// load the vector to shared mem
		shmb[threadIdx.x] = _B[iter * _dim + threadIdx.x];

		for (uint pert = 0; pert < 1; pert++) {

			__syncthreads();
			if (threadIdx.x < _p) {
				minVal = 10000000.;
				minIdx = 0;

				code1[threadIdx.x] = _assign1[(iter * 1 + pert)
						* _p + threadIdx.x];
			}
			__syncthreads();

			uint pIdx = pertIdx(threadIdx.x, _dimBits, pert);
			// load perturbed vector
			float b = shmb[pIdx];

			const float* A = _cb2 + pert * _nClusters1 * _nClusters2 * _dim;

			// each segment needs a different codebook
			uint p = threadIdx.x / _vl;
			const float* cb = A
					+ getCBIdx(p, code1[p], _nClusters1, _vl, _nClusters2)
					+ (threadIdx.x % _vl);

			// loop over all vectors of A
			for (int a = 0; a < _nClusters2; a++) {

				float s = sqr(b - cb[a * _vl]);
				shm[threadIdx.x] = s; // sqr(b - cb[a * _vl]);

				// compute the sum of differences for the vector segments
				// i.e. all _p segments in parallel
				for (uint stride = _vl >> 1; stride > 0; stride >>= 1) {
					__syncthreads();

					if ((threadIdx.x) < _p * stride) {
						uint p = threadIdx.x / stride * _vl;
						uint bias = threadIdx.x % stride;
						shm[p + bias] += shm[p + bias + stride];
					}
				}
				__syncthreads();

				if (threadIdx.x < _p) {
					// select the minimum of each segment
					float val = shm[threadIdx.x * _vl];
					if (val < minVal) {
						minVal = val;
						minIdx = a;
					}
				}

				__syncthreads();
			}

			// write out decision
			if (threadIdx.x < _p) {
				_assign[(iter * 1 + pert) * _p + threadIdx.x] =
						minIdx;
			}
		} // perturbation

	} // iter
}

void PerturbationProTree::getAssignment2(uint *_assign2, const float* _A,
		const float* _B, uint _Arows, uint _Brows, const uint *_assign1,
		uint _nClusters1) const {

	dim3 block(d_dim, 1, 1);
	dim3 grid((_Brows < 1024) ? _Brows : 1024, 1, 1);

	uint shm = (2 * d_dim + d_p) * sizeof(float);
	assignPerturbationClusterKernel2<<<grid, block, shm>>>(_assign2, _A, _B,
			_Arows, _Brows, d_dim, d_p, d_vl, _assign1, _nClusters1, d_dimBits);

	checkCudaErrors(cudaDeviceSynchronize());
}

__device__ void calcIdx(volatile uint* _shm, const uint* _assign,
		const uint* _assign2, uint _p, uint _nClusters, uint _nClusters2,
		uint _iter, uint _pert,  uint _nBins) {

// load assignment vector into shm;
	if (threadIdx.x < _p) {
		uint offs = (_iter * 1 + _pert) * _p + threadIdx.x;
		_shm[threadIdx.x] = _assign[offs] * _nClusters2 + _assign2[offs];
	}

// assume implicit synchronization as num threads is smaller than
	if (threadIdx.x == 0) {
		for (int p = 1; p < _p; p++)
			_shm[0] = _shm[0] * _nClusters * _nClusters2 + _shm[p];
		_shm[0] += _nBins * _pert;

#if USE_HASH
		_shm[0] = _shm[0] % HASH_SIZE;
#endif

	}

	__syncthreads();
}

__global__ void countBinsKernel(uint* _bins, const uint* _assign,
		uint* _assign2, uint _N, uint _p, uint _nClusters, uint _nClusters2,
		 uint _nBins) {

	extern __shared__ float shmf[];

	uint* shm = (uint*) shmf;

	for (int iter = blockIdx.x; iter < _N; iter += gridDim.x) {

		__syncthreads();

		for (int pert = 0; pert < 1; pert++) {
			calcIdx(shm, _assign, _assign2, _p, _nClusters, _nClusters2, iter,
					pert, _nBins);

			if (threadIdx.x == 0) {

				atomicInc(_bins + shm[0], _N);

				if (iter == 0) {
					printf("bin: %d %d \n", pert, shm[0]);
				}

			}
		}
	}
}

void PerturbationProTree::countBins(uint* _bins, const uint* _assign,
		uint* _assign2, uint _N) {

	dim3 block(d_p + d_p, 1, 1);

	dim3 grid((_N < 1024) ? _N : 1024, 1, 1);

	uint shmsize = (d_p + d_p) * sizeof(uint);

	if (!_bins) {
		std::cout << "did not get binCount array " << std::endl;
		exit(1);
	}

#if USE_HASH
	cudaMemset(_bins, 0, HASH_SIZE * sizeof(uint));
#else
	cudaMemset(_bins, 0, d_nDBs * d_nBins * sizeof(uint));
#endif

	countBinsKernel<<<grid, block, shmsize>>>(_bins, _assign, _assign2, _N, d_p,
			d_nClusters, d_nClusters2,  d_nBins);

	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void countBinsKernel(uint* _bins, const int* _assignedBins, uint _N) {

	for (int iter = blockIdx.x * blockDim.x + threadIdx.x;
			iter < _N * 1; iter += gridDim.x * blockDim.x) {

		if (_assignedBins[iter] >= 0)
			atomicInc(_bins + _assignedBins[iter], _N);

	}
}

void PerturbationProTree::countBins(uint* _bins, const int* _assignedBins,
		uint _N) {

	uint n = _N * d_nDBs;
	uint nThreads = (n < 256) ? n : 256;
	uint nBlocks = idiv(n, nThreads);
	nBlocks = (nBlocks < 65000) ? nBlocks : 65000;

	dim3 block(nThreads, 1, 1);

	dim3 grid(nBlocks, 1, 1);

	if (!_bins) {
		std::cout << "did not get binCount array " << std::endl;
		exit(1);
	}

#if USE_HASH
	cudaMemset(_bins, 0, HASH_SIZE * sizeof(uint));
#else
	cudaMemset(_bins, 0, d_nDBs * d_nBins * sizeof(uint));
#endif
	countBinsKernel<<<grid, block>>>(_bins, _assignedBins, _N);

	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void sortIdxKernel(uint* _dbIdx, uint* _binCount,
		const uint* _prefix, const uint* _assign, const uint* _assign2, uint _N,
		uint _p, uint _nClusters, uint _nClusters2,
		uint _nBins) {

	extern __shared__ float shmf[];

	uint* shm = (uint*) shmf;

	for (int iter = blockIdx.x; iter < _N; iter += gridDim.x) {

		for (int pert = 0; pert < 1; pert++) {

			calcIdx(shm, _assign, _assign2, _p, _nClusters, _nClusters2, iter,
					pert, _nBins);

			if (threadIdx.x == 0) {
				uint pos = atomicInc(_binCount + shm[0], _N);
				if ((_prefix[shm[0]] + pos) > 1 * _N) {
					printf("out of range!: %d, %d, %d \n", _prefix[shm[0]], pos,
							shm[0]);
				}

				_dbIdx[_prefix[shm[0]] + pos] = iter;

			}
		}
	}
}

void PerturbationProTree::sortIdx(uint* _dbIdx, const uint* _assign,
		const uint* _assign2, uint _N) {

	dim3 block(d_p + d_p, 1, 1);

	dim3 grid((_N < 1024) ? _N : 1024, 1, 1);

	uint shmsize = (d_p + d_p) * sizeof(uint);

#if USE_HASH
	cudaMemset(d_binCounts, 0, HASH_SIZE * sizeof(uint));
#else
	cudaMemset(d_binCounts, 0, d_nDBs * d_nBins * sizeof(uint));
#endif

	sortIdxKernel<<<grid, block, shmsize>>>(_dbIdx, d_binCounts, d_binPrefix,
			_assign, _assign2, _N, d_p, d_nClusters, d_nClusters2,
			d_nBins);

	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void sortIdxKernel(uint* _dbIdx, uint* _binCount,
		const uint* _prefix, const int* _assignedBins, uint _N) {

	for (int iter = blockIdx.x * blockDim.x + threadIdx.x;
			iter < _N * 1; iter += gridDim.x * blockDim.x) {

		if (_assignedBins[iter] >= 0) {
			uint pos = atomicInc(_binCount + _assignedBins[iter], _N);

			_dbIdx[_prefix[_assignedBins[iter]] + pos] = iter / 1;
		}
	}
}

void PerturbationProTree::sortIdx(uint* _dbIdx, const int* _assignedBins,
		uint _N) {

	uint n = _N * d_nDBs;
	uint nThreads = (n < 1024) ? n : 1024;
	uint nBlocks = idiv(n, nThreads);
	nBlocks = (nBlocks < 65000) ? nBlocks : 65000;

	dim3 block(nThreads, 1, 1);
	dim3 grid(nBlocks, 1, 1);

#if USE_HASH
	cudaMemset(d_binCounts, 0, HASH_SIZE * sizeof(uint));
#else
	cudaMemset(d_binCounts, 0, d_nDBs * d_nBins * sizeof(uint));
#endif

	sortIdxKernel<<<grid, block>>>(_dbIdx, d_binCounts, d_binPrefix,
			_assignedBins, _N);

	checkCudaErrors(cudaDeviceSynchronize());
}

void PerturbationProTree::buildDB(const float* _A, uint _N) {

	uint* assignd;
	uint* assignd2;

	cudaMalloc(&assignd, _N * d_p * d_nDBs * sizeof(uint));
	cudaMalloc(&assignd2, _N * d_p * d_nDBs * sizeof(uint));

	if ((assignd == NULL) || (assignd2 == NULL)) {
		std::cout << "buildDB did not get memory!" << std::endl;
		exit(1);
	}

	getAssignment(assignd, d_multiCodeBook, _A, d_nClusters, _N);

//	outputVecUint("assign", assignd, 256);

	getAssignment2(assignd2, d_multiCodeBook2, _A, d_nClusters2, _N, assignd,
			d_nClusters);

//	outputVecUint("assignd2", assignd2, 200);

	std::cout << "clusters: " << d_nClusters << "  " << d_nClusters2 << std::endl;

	std::cout << "number of data bases " << d_nDBs << std::endl;

	d_nBins = pow(d_nClusters, d_p) * pow(d_nClusters2, d_p);

	std::cout << "number of bins: " << d_nBins << std::endl;

#if USE_HASH
	cudaMalloc(&d_binPrefix, HASH_SIZE * sizeof(uint));

	cudaMalloc(&d_binCounts, HASH_SIZE * sizeof(uint));

	cudaMalloc(&d_dbIdx, d_nDBs * _N * sizeof(uint));
	countBins(d_binCounts, assignd, assignd2, _N);

	histogram(HASH_SIZE);

	cudaMemset(d_binPrefix, 0, HASH_SIZE * sizeof(uint));

	scan(d_binPrefix, d_binCounts, HASH_SIZE, false);

#else
	cudaMalloc(&d_binPrefix, d_nDBs * d_nBins * sizeof(uint));

	cudaMalloc(&d_binCounts, d_nDBs * d_nBins * sizeof(uint));

	cudaMalloc(&d_dbIdx, d_nDBs * _N * sizeof(uint));

	countBins(d_binCounts, assignd, assignd2, _N);

	histogram(d_nBins);

//	outputVecUint("binCounts1: ", d_binCounts + 11000, 1000);
//	outputVecUint("binCounts2: ", d_binCounts + 11000 + 16777216, 1000);

	cudaMemset(d_binPrefix, 0, d_nDBs * d_nBins * sizeof(uint));

	scan(d_binPrefix, d_binCounts, d_nDBs * d_nBins, false);
#endif

	cudaMemset(d_dbIdx, 0, d_nDBs * _N * sizeof(uint));

	sortIdx(d_dbIdx, assignd, assignd2, _N);

// store references to original vectors
	d_dbVec = _A;
	d_NdbVec = _N; // store number o orginal vectors

	cudaFree(assignd2);
	cudaFree(assignd);

	std::cout << "done with buildDB " << std::endl;

}

__global__ void assignPerturbationBestBinKernel2(int *_assignBin,
		const float* _cb2, const float* _B, uint _nClusters2, uint _Brows,
		uint _dim, uint _p, uint _vl, const uint* _assign1, uint _nClusters1,
		uint _k1, uint _NP2, uint _c1scale, uint _dimBits, uint _nBins,
		 uint* _binIdx = NULL, float* _binDist = NULL) {

	extern __shared__ float shmb[];

	float* shmIter = shmb + _dim;

	float* shm = shmIter;
	shmIter += blockDim.x;

	uint* binL1 = (uint*) shmIter;
	shmIter += _p;

	float* val = shmIter;
	shmIter += _p;

	uint* idx = (uint*) shmIter;
	shmIter += _p;

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		__syncthreads();
		// load the vector to shared mem
		if (threadIdx.x < _dim)
			shmb[threadIdx.x] = _B[iter * _dim + threadIdx.x];

		for (uint pert = 0; pert < 1; pert++) {
			__syncthreads();
			uint pIdx = pertIdx(threadIdx.x, _dimBits, pert);
			// load perturbed vector
			float b = shmb[pIdx];

			// loop over the best k1 first-level bins
			for (int k = 0; k < _k1; k++) {

				if (threadIdx.x < _p) {
					binL1[threadIdx.x] =
							_assign1[(iter * 1 + pert) * _k1 * _p
									+ k * _p + threadIdx.x];
				}
				__syncthreads();

				const float* A = _cb2 + pert * _nClusters1 * _nClusters2 * _dim;
				// each segment needs a different codebook
				const float* cb;
				if (threadIdx.x < _dim) {
					uint p = threadIdx.x / _vl;
					cb = A
							+ getCBIdx(p, binL1[p], _nClusters1, _vl,
									_nClusters2) + (threadIdx.x % _vl);
				}
				// loop over all vectors of A
				for (int binL2 = 0; binL2 < _nClusters2; binL2++) {

					if (threadIdx.x < _dim)
						shm[threadIdx.x] = sqr(b - cb[binL2 * _vl]);

					// compute the sum of differences for the vector segments
					// i.e. all _p segments in parallel
					for (uint stride = _vl >> 1; stride > 0; stride >>= 1) {
						__syncthreads();
						if ((threadIdx.x) < _p * stride) {
							uint p = threadIdx.x / stride * _vl;
							uint bias = threadIdx.x % stride;
							shm[p + bias] += shm[p + bias + stride];
						}
					}
					__syncthreads();

					// store the best result
					if (threadIdx.x < _p) {

						if ((val[threadIdx.x] > shm[threadIdx.x * _vl])
								|| ((k + binL2) == 0)) {

							val[threadIdx.x] = shm[threadIdx.x * _vl];
							idx[threadIdx.x] = binL2
									+ binL1[threadIdx.x] * _c1scale;
						}
					}

					__syncthreads();
				}
			}

			if (_binDist) {
				if ((pert == 0) && (threadIdx.x < _p)) {
					_binIdx[iter * _p + threadIdx.x] = idx[threadIdx.x];
					_binDist[iter * _p + threadIdx.x] = val[threadIdx.x];
				}
			}

			__syncthreads();

			// write out the compact best bin index
			if (threadIdx.x == 0) {
				for (int p = 1; p < _p; p++)
					idx[0] = idx[0] * _nClusters1 * _nClusters2 + idx[p];
				idx[0] += _nBins * pert;
#if USE_HASH
				idx[0] = idx[0] % HASH_SIZE;
#endif
				_assignBin[(iter * 1 + pert)] = idx[0];

			}

		}
	}
}

__global__ void assignPerturbationBestBinKernel2Sparse(int *_assignBin,
		const float* _cb2, const float* _B, uint _nClusters2, uint _Brows,
		uint _dim, uint _p, uint _vl, const uint* _assign1, uint _nClusters1,
		uint _k1, uint _NP2, uint _c1scale, uint _dimBits, uint _nBins,
		const uint *_sparseBin, bool _sparse,
		uint* _binIdx = NULL, float* _binDist = NULL) {

	extern __shared__ float shmb[];

	float* shmIter = shmb + _dim;

	float* shm = shmIter;
	shmIter += blockDim.x;

	uint* binL1 = (uint*) shmIter;
	shmIter += _p;

	float* val = shmIter;
	shmIter += _p;

	uint* idx = (uint*) shmIter;
	shmIter += _p;

	bool &process = *(bool*) shmIter;
	shmIter += 1;

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		__syncthreads();

		// check if this vector is actually in a sparse / dense region
		if (threadIdx.x < _p) {
			binL1[threadIdx.x] = _assign1[(iter * 1) * _k1 * _p
					+ threadIdx.x];
		}
		if (threadIdx.x == 0) {

			uint binIdx;
			binIdx = binL1[0];
			binIdx = binL1[1] + binIdx * _nClusters1;
			binIdx = binL1[2] + binIdx * _nClusters1;
			binIdx = binL1[3] + binIdx * _nClusters1;

			process = true;
			if (_sparseBin[binIdx] != _sparse) {
				for (uint pert = 0; pert < 1; pert++) {
					_assignBin[(iter * 1 + pert)] = -1;
				}
				process = false;
			}

		}

		__syncthreads();

		if (!process)
			continue;

		// load the vector to shared mem
		if (threadIdx.x < _dim)
			shmb[threadIdx.x] = _B[iter * _dim + threadIdx.x];

		for (uint pert = 0; pert < 1; pert++) {
			__syncthreads();
			uint pIdx = pertIdx(threadIdx.x, _dimBits, pert);

			// load perturbed vector
			float b = shmb[pIdx];

			// loop over the best k1 first-level bins
			for (int k = 0; k < _k1; k++) {

				if (threadIdx.x < _p) {
					binL1[threadIdx.x] =
							_assign1[(iter * 1 + pert) * _k1 * _p
									+ k * _p + threadIdx.x];
				}
				__syncthreads();

				const float* A = _cb2 + pert * _nClusters1 * _nClusters2 * _dim;
				// each segment needs a different codebook
				const float* cb;
				if (threadIdx.x < _dim) {
					uint p = threadIdx.x / _vl;
					cb = A
							+ getCBIdx(p, binL1[p], _nClusters1, _vl,
									_nClusters2) + (threadIdx.x % _vl);
				}
				// loop over all vectors of A
				for (int binL2 = 0; binL2 < _nClusters2; binL2++) {

					if (threadIdx.x < _dim)
						shm[threadIdx.x] = sqr(b - cb[binL2 * _vl]);

					// compute the sum of differences for the vector segments
					// i.e. all _p segments in parallel
					for (uint stride = _vl >> 1; stride > 0; stride >>= 1) {
						__syncthreads();
						if ((threadIdx.x) < _p * stride) {
							uint p = threadIdx.x / stride * _vl;
							uint bias = threadIdx.x % stride;
							shm[p + bias] += shm[p + bias + stride];
						}
					}
					__syncthreads();

					// store the best result
					if (threadIdx.x < _p) {

						if ((val[threadIdx.x] > shm[threadIdx.x * _vl])
								|| ((k + binL2) == 0)) {

							val[threadIdx.x] = shm[threadIdx.x * _vl];
							idx[threadIdx.x] = binL2
									+ binL1[threadIdx.x] * _c1scale;
						}
					}

					__syncthreads();
				}
			}

			if (_binDist) {
				if ((pert == 0) && (threadIdx.x < _p)) {
					_binIdx[iter * _p + threadIdx.x] = idx[threadIdx.x];
					_binDist[iter * _p + threadIdx.x] = val[threadIdx.x];
				}
			}

			__syncthreads();

			// write out the compact best bin index
			if (threadIdx.x == 0) {
				for (int p = 1; p < _p; p++)
					idx[0] = idx[0] * _nClusters1 * _nClusters2 + idx[p];
				idx[0] += _nBins * pert;
#if USE_HASH
				idx[0] = idx[0] % HASH_SIZE;
#endif
				_assignBin[(iter * 1 + pert)] = idx[0];

			}

		}
	}
}

void PerturbationProTree::getBestBinAssignment2(int *_assignBin,
		const float* _cb2, const float* _B, uint _nClusters2, uint _Brows,
		const uint *_assign1, uint _k1, uint _nClusters1) const {

	uint NP2 = log2(_k1 * _nClusters2);

	std::cout << "NP2 " << NP2 << std::endl;

//	assert(d_dim >= (2 * NP2));

	int nThreads = (d_dim > (2 * NP2)) ? d_dim : (2 * NP2);

	dim3 block(nThreads, 1, 1);
	dim3 grid((_Brows < 1024) ? _Brows : 1024, 1, 1);

	uint shm = (d_dim + nThreads + 3 * d_p) * sizeof(float);

	std::cout << "shm" << shm << std::endl;

	std::cout << "d_nBins" << d_nBins << std::endl;

	uint c1scale = _nClusters2;

	assignPerturbationBestBinKernel2<<<grid, block, shm>>>(_assignBin, _cb2, _B,
			_nClusters2, _Brows, d_dim, d_p, d_vl, _assign1, _nClusters1, _k1,
			NP2, c1scale, d_dimBits, d_nBins);

	checkCudaErrors(cudaDeviceSynchronize());
}

void PerturbationProTree::getBestBinAssignment2Sparse(int *_assignBin,
		const float* _cb2, const float* _B, uint _nClusters2, uint _Brows,
		const uint *_assign1, uint _k1, uint _nClusters1, bool _sparse) const {

	uint NP2 = log2(_k1 * _nClusters2);

	std::cout << "NP2 " << NP2 << std::endl;

//	assert(d_dim >= (2 * NP2));

	int nThreads = (d_dim > (2 * NP2)) ? d_dim : (2 * NP2);

	dim3 block(nThreads, 1, 1);
	dim3 grid((_Brows < 1024) ? _Brows : 1024, 1, 1);

	uint shm = (d_dim + nThreads + 3 * d_p + 1) * sizeof(float);

	std::cout << "shm" << shm << std::endl;

	std::cout << "d_nBins" << d_nBins << std::endl;

	uint c1scale = _nClusters2;

	assignPerturbationBestBinKernel2Sparse<<<grid, block, shm>>>(_assignBin,
			_cb2, _B, _nClusters2, _Brows, d_dim, d_p, d_vl, _assign1,
			_nClusters1, _k1, NP2, c1scale, d_dimBits, d_nBins,
			d_sparseBin, _sparse);

	checkCudaErrors(cudaDeviceSynchronize());
}

void PerturbationProTree::getBestBinLineAssignment2(int *_assignBin,
		uint* _l2Idx, float* _l2dist, const float* _cb2, const float* _B,
		uint _nClusters2, uint _Brows, const uint *_assign1, uint _k1,
		uint _nClusters1) const {

	uint NP2 = log2(_k1 * _nClusters2);

	std::cout << "NP2 " << NP2 << std::endl;

//	assert(d_dim >= (2 * NP2));

	int nThreads = (d_dim > (2 * NP2)) ? d_dim : (2 * NP2);

	dim3 block(nThreads, 1, 1);
	dim3 grid((_Brows < 1024) ? _Brows : 1024, 1, 1);

	uint shm = (d_dim + nThreads + 2 * d_p) * sizeof(float);

	std::cout << "shm" << shm << std::endl;

	std::cout << "d_nBins" << d_nBins << std::endl;

	uint c1scale = _nClusters2;

	assignPerturbationBestBinKernel2<<<grid, block, shm>>>(_assignBin, _cb2, _B,
			_nClusters2, _Brows, d_dim, d_p, d_vl, _assign1, _nClusters1, _k1,
			NP2, c1scale, d_dimBits, d_nBins, _l2Idx, _l2dist);

	checkCudaErrors(cudaDeviceSynchronize());
}

void PerturbationProTree::setDB(uint _N, const uint* _prefix,
		const uint* _counts, const uint* _dbIdx) {

	d_nBins = pow(d_nClusters, d_p) * pow(d_nClusters2, d_p);

	d_dbVec = NULL;
	d_NdbVec = _N;

	if (!d_binPrefix) {
		cudaMalloc(&d_binPrefix, HASH_SIZE * sizeof(uint));

		std::cout << "got binPrefix: " << d_binPrefix << std::endl;

		cudaMalloc(&d_binCounts, HASH_SIZE * sizeof(uint));

		cudaMalloc(&d_dbIdx, d_nDBs * _N * sizeof(uint));
	}

	if ((d_binPrefix == NULL) || (d_binCounts == NULL) || (d_dbIdx == NULL)) {
		std::cout << "setDB did not get memory!" << std::endl;
		std::cout << "sizes: " << (HASH_SIZE * sizeof(uint)) << std::endl;
		std::cout << "sizes: " << (_N * d_nDBs * sizeof(uint)) << std::endl;
		exit(1);
	}
	if (HASH_SIZE < 100000000) {
		cudaMemcpy(d_binPrefix, _prefix, HASH_SIZE * sizeof(uint),
				cudaMemcpyHostToDevice);
		cudaMemcpy(d_binCounts, _counts, HASH_SIZE * sizeof(uint),
				cudaMemcpyHostToDevice);
	} else {
		uint chunk = 100000000;
		int chunks = HASH_SIZE / chunk;

		for (int c = 0; c < chunks; c++) {
			cudaMemcpy(d_binPrefix + c * chunk, _prefix + c * chunk,
					chunk * sizeof(uint), cudaMemcpyHostToDevice);
			cudaMemcpy(d_binCounts + c * chunk, _counts + c * chunk,
					chunk * sizeof(uint), cudaMemcpyHostToDevice);
		}
	}

	cudaMemcpy(d_dbIdx, _dbIdx, _N * sizeof(uint), cudaMemcpyHostToDevice);

	histogram(HASH_SIZE);

}

void PerturbationProTree::buildKBestDB(const float* _A, uint _N) {

	uint* assignd;
	int* assignedBins;


	uint k1 = 16;

	d_nBins = pow(d_nClusters, d_p) * pow(d_nClusters2, d_p);

	cudaMalloc(&assignd, k1 * _N * d_p * d_nDBs * sizeof(uint));
	cudaMalloc(&assignedBins, _N * d_nDBs * sizeof(uint));

	if ((assignd == NULL) || (assignedBins == NULL)) {
		std::cout << "buildDB did not get memory!" << std::endl;
		std::cout << "sizes: " << (k1 * _N * d_p * d_nDBs * sizeof(uint)) << std::endl;
		std::cout << "sizes: " << (_N * d_nDBs * sizeof(uint)) << std::endl;
		exit(1);
	}

	getKBestAssignment(assignd, d_multiCodeBook, _A, d_nClusters, _N, k1);



	getBestBinAssignment2(assignedBins, d_multiCodeBook2, _A, d_nClusters2, _N,
			assignd, k1, d_nClusters);


	std::cout << "clusters: " << d_nClusters << "  " << d_nClusters2 << std::endl;

	std::cout << "number of data bases " << d_nDBs << std::endl;

	std::cout << "number of bins: " << d_nBins << std::endl;

#if USE_HASH

	if (!d_binPrefix) {
		cudaMalloc(&d_binPrefix, HASH_SIZE * sizeof(uint));

		cudaMalloc(&d_binCounts, HASH_SIZE * sizeof(uint));

		cudaMalloc(&d_dbIdx, d_nDBs * _N * sizeof(uint));
	}

	countBins(d_binCounts, assignedBins, _N);

	histogram(HASH_SIZE);

	cudaMemset(d_binPrefix, 0, HASH_SIZE * sizeof(uint));

	scan(d_binPrefix, d_binCounts, HASH_SIZE, false);
#else

	cudaMalloc(&d_binPrefix, d_nDBs * d_nBins * sizeof(uint));

	cudaMalloc(&d_binCounts, d_nDBs * d_nBins * sizeof(uint));

	cudaMalloc(&d_dbIdx, d_nDBs * _N * sizeof(uint));

	countBins(d_binCounts, assignedBins, _N);

	histogram(d_nBins);

//	outputVecUint("binCounts1: ", d_binCounts, 1000);
//	outputVecUint("binCounts2: ", d_binCounts + 11000 + 16777216, 1000);

	cudaMemset(d_binPrefix, 0, d_nDBs * d_nBins * sizeof(uint));

	scan(d_binPrefix, d_binCounts, d_nDBs * d_nBins, false);
#endif

	cudaMemset(d_dbIdx, 0, d_nDBs * _N * sizeof(uint));

	sortIdx(d_dbIdx, assignedBins, _N);

// store references to original vectors
	d_dbVec = _A;
	d_NdbVec = _N; // store number o orginal vectors

	cudaFree(assignedBins);
	cudaFree(assignd);

	std::cout << "done with buildDB " << std::endl;

}

void PerturbationProTree::buildKBestDBSparse(const float* _A, uint _N,
		bool _sparse) {

	uint* assignd;
	int* assignedBins;

	// TODO
	uint k1 = 1;
//	uint k1 = 8;
//	k1 = 32;

	d_nBins = pow(d_nClusters, d_p) * pow(d_nClusters2, d_p);

	cudaMalloc(&assignd, k1 * _N * d_p * d_nDBs * sizeof(uint));
	cudaMalloc(&assignedBins, _N * d_nDBs * sizeof(uint));

	if ((assignd == NULL) || (assignedBins == NULL)) {
		std::cout << "buildDB did not get memory!" << std::endl;
		std::cout << "sizes: " << (k1 * _N * d_p * d_nDBs * sizeof(uint)) << std::endl;
		std::cout << "sizes: " << (_N * d_nDBs * sizeof(uint)) << std::endl;
		exit(1);
	}

	getKBestAssignment(assignd, d_multiCodeBook, _A, d_nClusters, _N, k1);

//	outputVecUint("assignd", assignd, 1000);

#if 0
	/*** new block to test lines */
	float* assignVal;
	uint* assignIdx;
	cudaMalloc(&assignVal,
			k1 * d_nClusters2 * _N * d_p * d_nDBs * sizeof(float));
	cudaMalloc(&assignIdx,
			k1 * d_nClusters2 * _N * d_p * d_nDBs * sizeof(uint));

	if ((assignVal == NULL) || (assignIdx == NULL)) {
		std::cout << "buildDB assignVal/assignIdx did not get memory!" << std::endl;
		exit(1);
	}

	getKBestAssignment2(assignVal, assignIdx, d_multiCodeBook2, _A,
			d_nClusters2, _N, assignd, d_nClusters, k1);

	testLineDist(assignVal, assignIdx, k1, _N);

	/*** end of block */
#endif

	getBestBinAssignment2Sparse(assignedBins, d_multiCodeBook2, _A,
			d_nClusters2, _N, assignd, k1, d_nClusters, _sparse);

//	outputVecUint("assignedBins", assignedBins + 900000, 200);

	outputVecInt("assignedBins", assignedBins, 1000);

	std::cout << "clusters: " << d_nClusters << "  " << d_nClusters2 << std::endl;

	std::cout << "number of data bases " << d_nDBs << std::endl;

	std::cout << "number of bins: " << d_nBins << std::endl;

#if USE_HASH

	if (!d_binPrefix) {
		cudaMalloc(&d_binPrefix, HASH_SIZE * sizeof(uint));

		cudaMalloc(&d_binCounts, HASH_SIZE * sizeof(uint));

		cudaMalloc(&d_dbIdx, d_nDBs * _N * sizeof(uint));
	}

	countBins(d_binCounts, assignedBins, _N);

	histogram(HASH_SIZE);

	cudaMemset(d_binPrefix, 0, HASH_SIZE * sizeof(uint));

	scan(d_binPrefix, d_binCounts, HASH_SIZE, false);
#else

	cudaMalloc(&d_binPrefix, d_nDBs * d_nBins * sizeof(uint));

	cudaMalloc(&d_binCounts, d_nDBs * d_nBins * sizeof(uint));

	cudaMalloc(&d_dbIdx, d_nDBs * _N * sizeof(uint));

	countBins(d_binCounts, assignedBins, _N);

	histogram(d_nBins);

//	outputVecUint("binCounts1: ", d_binCounts, 1000);
//	outputVecUint("binCounts2: ", d_binCounts + 11000 + 16777216, 1000);

	cudaMemset(d_binPrefix, 0, d_nDBs * d_nBins * sizeof(uint));

	scan(d_binPrefix, d_binCounts, d_nDBs * d_nBins, false);
#endif

	cudaMemset(d_dbIdx, 0, d_nDBs * _N * sizeof(uint));

	sortIdx(d_dbIdx, assignedBins, _N);

// store references to original vectors
	d_dbVec = _A;
	d_NdbVec = _N; // store number o orginal vectors

	cudaFree(assignedBins);
	cudaFree(assignd);

	std::cout << "done with buildDBSparse " << std::endl;

}

void PerturbationProTree::buildKBestLineDB(const float* _A, uint _N) {

	uint* assignd;

	float* l1Dist;
	float* l2Dist;
	uint* l2Idx;

	int* assignedBins;

	uint k1 = 8;

	d_nBins = pow(d_nClusters, d_p) * pow(d_nClusters2, d_p);

	cudaMalloc(&assignd, k1 * _N * d_p * d_nDBs * sizeof(uint));
	cudaMalloc(&l1Dist, d_nClusters * _N * d_p * sizeof(float));
	cudaMalloc(&l2Dist, _N * d_p * sizeof(float));
	cudaMalloc(&l2Idx, _N * d_p * sizeof(uint));

	cudaMalloc(&assignedBins, _N * d_nDBs * sizeof(uint));

	if ((assignd == NULL) || (assignedBins == NULL)) {
		std::cout << "buildDB did not get memory!" << std::endl;
		exit(1);
	}

	getKBestLineAssignment(assignd, l1Dist, d_multiCodeBook, _A, d_nClusters,
			_N, k1);

//	outputVecUint("assignd", assignd, 1000);

	getBestBinLineAssignment2(assignedBins, l2Idx, l2Dist, d_multiCodeBook2, _A,
			d_nClusters2, _N, assignd, k1, d_nClusters);

	outputVecUint("l2Idx", l2Idx, 1000);

	assembleLines(l1Dist, l2Idx, l2Dist, _N);

//	outputVecUint("assignedBins", assignedBins + 900000, 200);

	std::cout << "clusters: " << d_nClusters << "  " << d_nClusters2 << std::endl;

	std::cout << "number of data bases " << d_nDBs << std::endl;

	std::cout << "number of bins: " << d_nBins << std::endl;

#if USE_HASH
	cudaMalloc(&d_binPrefix, HASH_SIZE * sizeof(uint));

	cudaMalloc(&d_binCounts, HASH_SIZE * sizeof(uint));

	cudaMalloc(&d_dbIdx, d_nDBs * _N * sizeof(uint));

	countBins(d_binCounts, assignedBins, _N);

	histogram(HASH_SIZE);

	cudaMemset(d_binPrefix, 0, HASH_SIZE * sizeof(uint));

	scan(d_binPrefix, d_binCounts, HASH_SIZE, false);
#else

	cudaMalloc(&d_binPrefix, d_nDBs * d_nBins * sizeof(uint));

	cudaMalloc(&d_binCounts, d_nDBs * d_nBins * sizeof(uint));

	cudaMalloc(&d_dbIdx, d_nDBs * _N * sizeof(uint));

	countBins(d_binCounts, assignedBins, _N);

	histogram(d_nBins);

//	outputVecUint("binCounts1: ", d_binCounts, 1000);
//	outputVecUint("binCounts2: ", d_binCounts + 11000 + 16777216, 1000);

	cudaMemset(d_binPrefix, 0, d_nDBs * d_nBins * sizeof(uint));

	scan(d_binPrefix, d_binCounts, d_nDBs * d_nBins, false);
#endif

	cudaMemset(d_dbIdx, 0, d_nDBs * _N * sizeof(uint));

	sortIdx(d_dbIdx, assignedBins, _N);

// store references to original vectors
	d_dbVec = _A;
	d_NdbVec = _N; // store number o orginal vectors

//	cudaFree(l2Idx); // do not free as this array is stored in d_l2Idx;
	cudaFree(l2Dist);

	cudaFree(l1Dist);
	cudaFree(assignedBins);
	cudaFree(assignd);

	std::cout << "done with buildDB " << std::endl;

}

// each block is responsible for one vector, blockDim.x should be _dim
// requires _dim * float shared memory
// looping multiple times to process all B vectors
// _vl is the length of the _p vector segments (should be 2^n)
__global__ void assignPerturbationKBestClusterKernel2(float *_assignVal,
		uint* _assignIdx, const float* _cb2, const float* _B, uint _nClusters2,
		uint _Brows, uint _dim, uint _p, uint _vl, const uint* _assign1,
		uint _nClusters1, uint _k1, uint _NP2, uint _c1scale, uint _dimBits) {

	extern __shared__ float shmb[];

	float* shm = shmb + _dim;
	float* shmIter = shm + blockDim.x;

	uint* shmIdx = (uint*) shmIter;
	shmIter += blockDim.x;

	uint* binL1 = (uint*) shmIter;
	shmIter += _p;

	float* val = shmIter;
	shmIter += _p * _k1 * _nClusters2;

	uint* idx = (uint*) shmIter;
	shmIter += _p * _k1 * _nClusters2;

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		__syncthreads();
		// load the vector to shared mem
		if (threadIdx.x < _dim)
			shmb[threadIdx.x] = _B[iter * _dim + threadIdx.x];

		for (uint pert = 0; pert < 1; pert++) {
			__syncthreads();
			uint pIdx = pertIdx(threadIdx.x, _dimBits, pert);
			// load perturbed vector
			float b = shmb[pIdx];

			// loop over the best k1 first-level bins
			for (int k = 0; k < _k1; k++) {

				if (threadIdx.x < _p) {
					binL1[threadIdx.x] =
							_assign1[(iter * 1 + pert) * _k1 * _p
									+ k * _p + threadIdx.x];

//					printf(" npert: %d tid: %d pert: %d binL1 %d \n", 1, threadIdx.x, pert, binL1[threadIdx.x]);
				}
				__syncthreads();

				const float* A = _cb2 + pert * _nClusters1 * _nClusters2 * _dim;
				// each segment needs a different codebook
				const float* cb;
				if (threadIdx.x < _dim) {
					uint p = threadIdx.x / _vl;
					cb = A
							+ getCBIdx(p, binL1[p], _nClusters1, _vl,
									_nClusters2) + (threadIdx.x % _vl);
				}
				// loop over all vectors of A
				for (int binL2 = 0; binL2 < _nClusters2; binL2++) {

					if (threadIdx.x < _dim)
						shm[threadIdx.x] = sqr(b - cb[binL2 * _vl]);

					// compute the sum of differences for the vector segments
					// i.e. all _p segments in parallel
					for (uint stride = _vl >> 1; stride > 0; stride >>= 1) {
						__syncthreads();
						if ((threadIdx.x) < _p * stride) {
							uint p = threadIdx.x / stride * _vl;
							uint bias = threadIdx.x % stride;
							shm[p + bias] += shm[p + bias + stride];
						}
					}
					__syncthreads();

					// store the result
					if (threadIdx.x < _p) {

						val[binL2 + k * _nClusters2
								+ threadIdx.x * _k1 * _nClusters2] =
								shm[threadIdx.x * _vl];
						idx[binL2 + k * _nClusters2
								+ threadIdx.x * _k1 * _nClusters2] = binL2
								+ binL1[threadIdx.x] * _c1scale;
					}

					__syncthreads();
				}
			}

			// sort the results
			for (int i = 0; i < _p; i++) {

				if (threadIdx.x < _NP2)
					shm[threadIdx.x] = 1000000000.;

				// copy to original shm
				if (threadIdx.x < _k1 * _nClusters2) {
					shm[threadIdx.x] = val[threadIdx.x + i * _k1 * _nClusters2];
					shmIdx[threadIdx.x] = idx[threadIdx.x
							+ i * _k1 * _nClusters2];
				}
				__syncthreads();

				bitonic3(shm, shmIdx, _NP2);

				if (threadIdx.x < _k1 * _nClusters2) {
					val[threadIdx.x + i * _k1 * _nClusters2] = shm[threadIdx.x];
					idx[threadIdx.x + i * _k1 * _nClusters2] =
							shmIdx[threadIdx.x];
				}

				__syncthreads();

			}

			// write out the sorted bins
			for (int p = 0; p < _p; p++) {
				if (threadIdx.x < _k1 * _nClusters2) {
					_assignVal[(iter * 1 + pert) * _k1 * _p
							* _nClusters2 + p * _k1 * _nClusters2 + threadIdx.x] =
							val[threadIdx.x + p * _k1 * _nClusters2];
					_assignIdx[(iter * 1 + pert) * _k1 * _p
							* _nClusters2 + p * _k1 * _nClusters2 + threadIdx.x] =
							idx[threadIdx.x + p * _k1 * _nClusters2];
				}
			}

		}
	}
}

void PerturbationProTree::getKBestAssignment2(float *_assignVal,
		uint *_assignIdx, const float* _cb2, const float* _B, uint _nClusters2,
		uint _Brows, const uint *_assign1, uint _nClusters1, uint _k1) const {

	uint NP2 = log2(_k1 * _nClusters2);

	std::cout << "NP2 " << NP2 << std::endl;

//	assert(d_dim >= (2 * NP2));

	int nThreads = (d_dim > (2 * NP2)) ? d_dim : (2 * NP2);

	dim3 block(nThreads, 1, 1);
	dim3 grid((_Brows < 1024) ? _Brows : 1024, 1, 1);

	uint shm = (d_dim + 2 * nThreads + d_p + 2 * _k1 * d_p * d_nClusters2)
			* sizeof(float);

	std::cout << "shm" << shm << std::endl;

	uint c1scale = d_nClusters2;

	assignPerturbationKBestClusterKernel2<<<grid, block, shm>>>(_assignVal,
			_assignIdx, _cb2, _B, _nClusters2, _Brows, d_dim, d_p, d_vl,
			_assign1, _nClusters1, _k1, NP2, c1scale, d_dimBits);

	checkCudaErrors(cudaDeviceSynchronize());
}

__device__ bool isTriangle(float _a2, float _b2, float _c2) {

	float a, b, c;

	a = sqrt(_a2);
	b = sqrt(_b2);
	c = sqrt(_c2);

	if ((a + b) < c)
		return false;
	if ((b + c) < a)
		return false;
	if ((a + c) < b)
		return false;

	return true;

}

/** for each db point the function projects the point onto a line between the best cluster2 and any of the other selected cluster2.
 * it returns the distance to the closest line and the index of the corresponding cluster id.
 * This is only done for the first perturbation and for all parts individually.
 *
 * The kernel should only be called for blockDim <= 32 !
 */

__global__ void lineProjectionKernel(float *_lineDist, float* _clusterDist,
		const float* _assignVal, const uint* _assignIdx, const float* _cbDist,
		uint _p, uint _k1, uint _nClusters1, uint _nClusters2, uint _N) {

	extern __shared__ float shm[];

	float* shmIter = shm;

	float* val = shmIter;
	shmIter += blockDim.x;
	uint* idx = (uint*) shmIter;
	shmIter += blockDim.x;
	float* lambda = shmIter;
	shmIter += blockDim.x;
	float* dist = shmIter;
	shmIter += blockDim.x;

	for (int iter = blockIdx.x; iter < _N; iter += gridDim.x) {

		uint pert = 0;

		float clusterDist = 0.;
		float lineDist = 0.;

		__syncthreads();

		for (int p = 0; p < _p; p++) {

			// read the sorted assignment
			val[threadIdx.x] = _assignVal[(iter * 1 + pert) * _k1
					* _p * _nClusters2 + p * _k1 * _nClusters2 + threadIdx.x];
			idx[threadIdx.x] = _assignIdx[(iter * 1 + pert) * _k1
					* _p * _nClusters2 + p * _k1 * _nClusters2 + threadIdx.x];

			__syncthreads();
#if 1
			if (threadIdx.x > 0) {
//				const float* cbdist = _cbDist
//						+ idx[0] * _nClusters1 * _nClusters2 * _p
//						+ idx[threadIdx.x] * _p + p;

				const float* cbdist = _cbDist
						+ p * _nClusters1 * _nClusters2 * _nClusters1
								* _nClusters2;
				float c2 = *(cbdist + idx[0] * _nClusters1 * _nClusters2
						+ idx[threadIdx.x]);

				lambda[threadIdx.x] = project(val[0], val[threadIdx.x], c2,
						dist[threadIdx.x]);

				if (!isTriangle(val[0], val[threadIdx.x], c2))
					printf("tIdx: %d %f (abc) %f %f %f \n", threadIdx.x,
							dist[threadIdx.x], (val[0]), (val[threadIdx.x]),
							(c2));

//				if (iter == 0) printf( "tIdx: %d %f == %d \n", threadIdx.x, c2, idx[threadIdx.x]);

//				dist[threadIdx.x]  = c2;
			} else {
				clusterDist += val[0];
				dist[0] = 123451234.;
			}
#else
			dist[threadIdx.x] = 123451234.;
			if (threadIdx.x == 0)
			clusterDist += val[0];

			const float* cbdist = _cbDist
			+ p * _nClusters1 * _nClusters2 * _nClusters1 * _nClusters2;

			for (int j = 0; j < blockDim.x; j++) {
				if (threadIdx.x != j) {
					float c2 = *(cbdist + idx[j] * _nClusters1 * _nClusters2
							+ idx[threadIdx.x]);

					float d2;
					lambda[threadIdx.x] = project(val[j], val[threadIdx.x], c2,
							d2);

					if (d2 < dist[threadIdx.x])
					dist[threadIdx.x] = d2;
				}

			}
#endif
			__syncthreads();

			// reduction to find best axis
			for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
				__syncthreads();
				if (threadIdx.x < stride) {
					if (dist[threadIdx.x] > dist[threadIdx.x + stride]) {
						dist[threadIdx.x] = dist[threadIdx.x + stride];
						idx[threadIdx.x] = idx[threadIdx.x + stride];
						lambda[threadIdx.x] = lambda[threadIdx.x + stride];
					}
				}
			}

			__syncthreads();

			if (threadIdx.x == 0) {

				if (iter == 0)
					printf("%d %f %f \n ", p, dist[0], lambda[0]);

				lineDist += dist[0];
			}

			__syncthreads();
		}

		if (threadIdx.x == 0) {
			_lineDist[iter] = lineDist;
			_clusterDist[iter] = clusterDist;
		}
	}

}

void PerturbationProTree::computeCBDist() {

	if (d_codeBookDistL2)
		cudaFree(d_codeBookDistL2);

	cudaMalloc(&d_codeBookDistL2,
			d_p * d_nClusters * d_nClusters * d_nClusters2 * d_nClusters2
					* sizeof(float));
	// consider codebook layout: one coodebook per part  p0 cb, p1 cb, ...

	uint distSize = d_nClusters * d_nClusters * d_nClusters2 * d_nClusters2;
	for (int p = 0; p < d_p; p++) {
		calcDist(d_codeBookDistL2 + p * distSize,
				d_multiCodeBook2 + d_nClusters * d_nClusters2 * d_vl * p,
				d_multiCodeBook2 + d_nClusters * d_nClusters2 * d_vl * p,
				d_nClusters * d_nClusters2, d_nClusters * d_nClusters2, d_vl,
				1);
	}
}

void PerturbationProTree::computeCBL1L2Dist() {

	if (d_codeBookDistL1L2)
		cudaFree(d_codeBookDistL1L2);

	cudaMalloc(&d_codeBookDistL1L2,
			d_p * d_nClusters * d_nClusters * d_nClusters2 * sizeof(float));
	// consider codebook layout: one coodebook per part  p0 cb, p1 cb, ...

	uint distSize = d_nClusters * d_nClusters * d_nClusters2;
	for (int p = 0; p < d_p; p++) {
		calcDist(d_codeBookDistL1L2 + p * distSize,
				d_multiCodeBook2 + d_nClusters * d_nClusters2 * d_vl * p,
				d_multiCodeBook + d_nClusters * d_vl * p,
				d_nClusters * d_nClusters2, d_nClusters, d_vl, 1);
	}
}

#if 0
void PerturbationProTree::computeCBL1L1Dist(uint _nParts) {

	if (d_codeBookDistL1L2)
	cudaFree(d_codeBookDistL1L2);

	cudaMalloc(&d_codeBookDistL1L2,
			_nParts * d_nClusters * d_nClusters * sizeof(float));
	// consider codebook layout: one coodebook per part  p0 cb, p1 cb, ...

	uint distSize = d_nClusters * d_nClusters;
	uint vl = d_dim / _nParts;
	for (int p = 0; p < _nParts; p++) {
		calcDist(d_codeBookDistL1L2 + p * distSize,
				d_multiCodeBook + d_nClusters * vl * p,
				d_multiCodeBook + d_nClusters * vl * p, d_nClusters,
				d_nClusters, vl, 1);
	}

	outputVec("cbDist", d_codeBookDistL1L2, 16 * 4);
}
#endif

void PerturbationProTree::computeCBL1L1Dist(uint _nParts) {

	if (d_codeBookDistL1L2)
		cudaFree(d_codeBookDistL1L2);

	cudaMalloc(&d_codeBookDistL1L2,
			_nParts * d_nClusters * d_nClusters * sizeof(float));
	// consider codebook layout: one coodebook per part  p0 cb, p1 cb, ...

//	uint distSize = d_nClusters * d_nClusters;
//	uint vl = d_dim / _nParts;
	calcDist(d_codeBookDistL1L2, d_multiCodeBook, d_multiCodeBook, d_nClusters,
			d_nClusters, d_dim, _nParts);

	outputVec("cbDist", d_codeBookDistL1L2, 16 * 4);
}

void PerturbationProTree::testLineDist(const float* _assignVal,
		const uint* _assignIdx, uint _k1, uint _N) {

	computeCBDist();

	outputVec("cbdist", d_codeBookDistL2, 1000);

	float* clusterDist;
	float* lineDist;

	cudaMalloc(&clusterDist, _N * sizeof(float));
	cudaMalloc(&lineDist, _N * sizeof(float));

	uint nThreads = 256;

	nThreads = (nThreads > _k1 * d_nClusters2) ? _k1 * d_nClusters2 : nThreads;
	dim3 block(nThreads, 1, 1);
	dim3 grid((_N > 1024) ? 1024 : _N, 1, 1);
	uint shmSize = 4 * block.x * sizeof(float);

	lineProjectionKernel<<<grid, block, shmSize>>>(lineDist, clusterDist,
			_assignVal, _assignIdx, d_codeBookDistL2, d_p, _k1, d_nClusters,
			d_nClusters2, _N);

	checkCudaErrors(cudaDeviceSynchronize());

	float* cdist = new float[_N];
	float* ldist = new float[_N];

	cudaMemcpy(cdist, clusterDist, _N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(ldist, lineDist, _N * sizeof(float), cudaMemcpyDeviceToHost);

	float minD, maxD, avgD;

	minD = cdist[0];
	maxD = cdist[0];
	avgD = 0;
	for (int i = 0; i < _N; i++) {
		if (cdist[i] < minD)
			minD = cdist[i];
		if (cdist[i] > maxD)
			maxD = cdist[i];
		avgD += cdist[i];
	}
	std::cout << "cluster dist (min, max, avg) " << minD << " " << maxD << " "
			<< (avgD / _N) << std::endl;

//	for (int i = 0; i < 1000; i++)
//		std::cout << "\t " << ldist[i];
//
//	std::cout << std::endl;

	minD = ldist[0];
	maxD = ldist[0];
	avgD = 0;
	for (int i = 0; i < _N; i++) {
		if (ldist[i] < minD)
			minD = ldist[i];
		if (ldist[i] > maxD)
			maxD = ldist[i];
		avgD += ldist[i];
	}
	std::cout << "line dist (min, max, avg)    " << minD << " " << maxD << " "
			<< (avgD / _N) << std::endl;

	cudaFree(lineDist);
	cudaFree(clusterDist);
}

/** for each db point the function projects the point onto a line between the best cluster2 and any of the other selected cluster2.
 * it returns the distance to the closest line and the index of the corresponding cluster id.
 * This is only done for the first perturbation and for all parts individually.
 *
 * The kernel should only be called for blockDim <= 32 !
 */

__global__ void lineProjectionKernel(uint *_lineIdx, float* _lineLambda,
		const float* _cbDist, const float* _l1Dist, const uint* _l2Idx,
		const float* _l2Dist, uint _nClusters1, uint _nClusters2, uint _p,
		uint _N) {

	extern __shared__ float shm[];

	float* shmIter = shm;

	float* l2Dist = shmIter;
	shmIter += _p;
	uint* idx = (uint*) shmIter;
	shmIter += _p;
	float* lambda = shmIter;
	shmIter += blockDim.x;
	float* dist = shmIter;
	shmIter += blockDim.x;
	float* lIdx = shmIter;
	shmIter += blockDim.x;

	for (int iter = blockIdx.x; iter < _N; iter += gridDim.x) {

		__syncthreads();

		if (threadIdx.x < _p) {
			l2Dist[threadIdx.x] = _l2Dist[iter * _p + threadIdx.x];
			idx[threadIdx.x] = _l2Idx[iter * _p + threadIdx.x];
		}
		__syncthreads();

		float l1Dist = _l1Dist[iter * _nClusters1 * _p + threadIdx.x];

		uint p = threadIdx.x / _nClusters1;
		uint pIdx = threadIdx.x % _nClusters1;

		if (idx[p] > (8 * 8 * 8 * 4)) {
			printf("%d %d", p, idx[p]);
		}

		float c2 = _cbDist[p * _nClusters1 * _nClusters2 + idx[p] * _nClusters1
				+ pIdx];

		lambda[threadIdx.x] = project(l2Dist[threadIdx.x], l1Dist, c2,
				dist[threadIdx.x]);

		lIdx[threadIdx.x] = pIdx;

		__syncthreads();

		// reduction to find best axis
		for (int stride = _nClusters1 >> 1; stride > 0; stride >>= 1) {
			__syncthreads();
			if (pIdx < stride) {
				if (dist[threadIdx.x] > dist[threadIdx.x + stride]) {
					dist[threadIdx.x] = dist[threadIdx.x + stride];
					lIdx[threadIdx.x] = lIdx[threadIdx.x + stride];
					lambda[threadIdx.x] = lambda[threadIdx.x + stride];
				}
			}
		}

		__syncthreads();

		if (threadIdx.x < p) {
			_lineIdx[iter * _p + threadIdx.x] = lIdx[threadIdx.x * _nClusters1];
			_lineLambda[iter * _p + threadIdx.x] = lambda[threadIdx.x
					* _nClusters1];
		}

	}

}

void PerturbationProTree::assembleLines(const float* _l1Dist, uint* _l2Idx,
		const float* _l2Dist, uint _N) {

	computeCBL1L2Dist();

	if (!d_lineLambda)
		cudaMalloc(&d_lineLambda, d_p * _N * sizeof(float));

	if (!d_lineIdx)
		cudaMalloc(&d_lineIdx, d_p * _N * sizeof(uint));

//	if (!d_l2Idx)
//		cudaMalloc(&d_l2Idx, d_p * _N * sizeof(uint));
	d_l2Idx = _l2Idx;

	uint nThreads = d_p * d_nClusters;
	dim3 block(nThreads, 1, 1);
	dim3 grid((_N > 1024) ? 1024 : _N, 1, 1);
	uint shmSize = (2 * d_p + 3 * block.x) * sizeof(float);

	lineProjectionKernel<<<grid, block, shmSize>>>(d_lineIdx, d_lineLambda,
			d_codeBookDistL1L2, _l1Dist, _l2Idx, _l2Dist, d_nClusters,
			d_nClusters2, d_p, _N);

	checkCudaErrors(cudaDeviceSynchronize());

}

#if 0
__global__ void selectBinKernel(uint* _assign, uint* _nBins,
		const float *_assignVal, const uint* _assignIdx,
		const uint* _nElemPerBin, uint _Arows, uint _Brows, uint _p, uint _vl,
		uint _nClusters1, uint _nClusters2, uint _k1, uint _k, uint _maxTrials,
		uint _maxOutBin, uint _c1scale, const uint *_distSeq, uint _numDistSeq,
		uint _distCluster, uint 1, uint _nBinsPerDB) {

// instead of the Dijkstra do the brute-force thing
	extern __shared__ float shm[];

	float* shmIter = shm;

	float* val = shmIter;
	shmIter += _p * _k1 * _Arows;
	uint* idx = (uint*) shmIter;
	shmIter += _p * _k1 * _Arows;
	uint* numbers = (uint*) shmIter;
	shmIter += _p;
	uint* denom = (uint*) shmIter;
	shmIter += _p;

	uint* outIdx = (uint*) shmIter;
	shmIter += blockDim.x;

	float* dist = (float*) shmIter;
	shmIter += blockDim.x;

	uint* nElem = (uint*) shmIter;
	shmIter += blockDim.x;

	uint* outBin = (uint*) shmIter;
	shmIter += _maxOutBin;

	uint& nOutBins = *(uint*) shmIter;
	shmIter += 1;

	uint& nElements = *(uint*) shmIter;
	shmIter += 1;

	uint& nIter = *(uint*) shmIter;
	shmIter += 1;

	if (threadIdx.x < _p) {
		numbers[threadIdx.x] = _distCluster;
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		denom[0] = 1;
		for (int i = 1; i < _p; i++) {
			denom[i] = denom[i - 1] * numbers[i - 1];
		}
	}

	__syncthreads();

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		for (uint pert = 0; pert < 1; pert++) {
			__syncthreads();
			// read the sorted assignment
			for (int p = 0; p < _p; p++) {
				if (threadIdx.x < _k1 * _Arows) {
					val[threadIdx.x + p * _k1 * _Arows] = _assignVal[(iter
							* 1 + pert) * _k1 * _p * _Arows
					+ p * _k1 * _Arows + threadIdx.x];
					idx[threadIdx.x + p * _k1 * _Arows] = _assignIdx[(iter
							* 1 + pert) * _k1 * _p * _Arows
					+ p * _k1 * _Arows + threadIdx.x];

				}
			}

			// TODO loop multiple times to include sufficiently many bins at the end

			if (threadIdx.x == 0) {
				nOutBins = 0;
				nElements = 0;
				nIter = 0;

			}

			__syncthreads();

			while ((nElements < _k) && (nIter < _maxTrials)
					&& (nOutBins < _maxOutBin)) {

				// generate all possible bins within the bounds given by numbers[]
				// calc the corresponding binIdx in the DB and the distance to the cluster center
				dist[threadIdx.x] = 0.;

				// TODO fix 4
				uint bin[4];// maximum number for p
				for (int p = 0; p < _p; p++) {
					bin[p] = (_distSeq[nIter * blockDim.x + threadIdx.x]
							/ denom[p]) % numbers[p];
					dist[threadIdx.x] += val[bin[p] + p * _k1 * _Arows];
					bin[p] = idx[bin[p] + p * _k1 * _Arows];

				}

				if (threadIdx.x >= _numDistSeq) {
					dist[threadIdx.x] = 99999999.;
				}

				__syncthreads();

				// TODO _p1 + _p2
				outIdx[threadIdx.x] = calcIdxSequential(bin, _p, _nClusters1,
						_nClusters2, _c1scale) + pert * _nBinsPerDB;

#if USE_HASH
				outIdx[threadIdx.x] = outIdx[threadIdx.x] % HASH_SIZE;
#endif

				//	printf("%d --- %d \n", threadIdx.x, outIdx[threadIdx.x]);

//				if (threadIdx.x < 100)
//					printf("%d: %d %d === %f  -- %d \n", threadIdx.x, bin[0],
//							bin[1], dist[threadIdx.x], outIdx[threadIdx.x]);

				__syncthreads();

				// sort all cluster centers based on the distance
				bitonic3(dist, outIdx, blockDim.x);

				//	if (outIdx[threadIdx.x] < )
				nElem[threadIdx.x] = _nElemPerBin[outIdx[threadIdx.x]];

				__syncthreads();

				// collect the number of vectors in all the bins
				// prepare output of bins with one or more vectors until the maximum number of vectors is reached
				// (performs a sequential reduction)
				if (threadIdx.x == 0) {

					for (int i = 0; i < blockDim.x; i++) {
						if ((nElements > _k) || (nOutBins > _maxOutBin))
						break;

//						if (i < 10)
//							printf("outIdx: %d, %f, %d \n", outIdx[i], dist[i],
//									nElem[i]);

						int n = nElem[i];
						if (n > 0) {
							outBin[nOutBins] = outIdx[i];
							nOutBins++;
							nElements += n;
						}
					}

					nIter++;
				}

				__syncthreads();
			}

			// write out result
			for (int b = threadIdx.x; b < nOutBins; b += blockDim.x)
			_assign[(iter * 1 + pert) * _maxOutBin + b] =
			outBin[b];

			__syncthreads();

			if (threadIdx.x == 0) {
				_nBins[iter * 1 + pert] = nOutBins;
			}
		}
	}
}
#else
__global__ void selectBinKernel(uint* _assign, uint* _nBins,
		const float *_assignVal, const uint* _assignIdx,
		const uint* _nElemPerBin, uint _Arows, uint _Brows, uint _p, uint _vl,
		uint _nClusters1, uint _nClusters2, uint _k1, uint _k, uint _maxTrials,
		uint _maxOutBin, uint _c1scale, const uint *_distSeq, uint _numDistSeq,
		uint _distCluster, uint _nBinsPerDB) {

// instead of the Dijkstra do the brute-force thing
	extern __shared__ float shm[];

	float* shmIter = shm;

	float* val = shmIter;
	shmIter += _p * _k1 * _Arows;
	uint* idx = (uint*) shmIter;
	shmIter += _p * _k1 * _Arows;
	uint* numbers = (uint*) shmIter;
	shmIter += _p;
	uint* denom = (uint*) shmIter;
	shmIter += _p;

	uint* outIdx = (uint*) shmIter;
	shmIter += blockDim.x;

	float* dist = (float*) shmIter;
	shmIter += blockDim.x;

	uint* nElem = (uint*) shmIter;
	shmIter += blockDim.x;

	uint* sBins = (uint*) shmIter;
	shmIter += 4 * blockDim.x;

	uint& nOutBins = *(uint*) shmIter;
	shmIter += 1;

	uint& nElements = *(uint*) shmIter;
	shmIter += 1;

	uint& nIter = *(uint*) shmIter;
	shmIter += 1;

	if (threadIdx.x < _p) {
		numbers[threadIdx.x] = _distCluster;
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		denom[0] = 1;
		for (int i = 1; i < _p; i++) {
			denom[i] = denom[i - 1] * numbers[i - 1];
		}
	}

//	if (threadIdx.x == 0)
//				printf("string select Bin \n \n \n");
//
//			__syncthreads();

	__syncthreads();

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

//		if (threadIdx.x == 0)
//			printf("iter %d \n", iter);

		__syncthreads();

		for (uint pert = 0; pert < 1; pert++) {
			__syncthreads();
			// read the sorted assignment
			for (int p = 0; p < _p; p++) {
				if (threadIdx.x < _k1 * _Arows) {
					val[threadIdx.x + p * _k1 * _Arows] = _assignVal[(iter
							* 1 + pert) * _k1 * _p * _Arows
							+ p * _k1 * _Arows + threadIdx.x];
					idx[threadIdx.x + p * _k1 * _Arows] = _assignIdx[(iter
							* 1 + pert) * _k1 * _p * _Arows
							+ p * _k1 * _Arows + threadIdx.x];

				}
			}

			// TODO loop multiple times to include sufficiently many bins at the end

			if (threadIdx.x == 0) {
				nOutBins = 0;
				nElements = 0;
				nIter = 0;

			}

//			if (threadIdx.x == 0)
//				printf("before iter %d \n", nIter);
			__syncthreads();

			while ((nElements < _k) && (nIter < _maxTrials)
					&& (nOutBins < _maxOutBin)) {

				__syncthreads();
//				if (threadIdx.x == 0)
//					printf("iter %d \n", nIter);
//				__syncthreads();

				// generate all possible bins within the bounds given by numbers[]
				// calc the corresponding binIdx in the DB and the distance to the cluster center
//				dist[threadIdx.x] = 0.;

				// TODO fix 4
				// uint bin[4];				// maximum number for p

				uint* bin = sBins + _p * threadIdx.x;

				float ddd = 0.;
//				for (int p = 0; p < _p; p++) {
//					uint bp = (_distSeq[nIter * blockDim.x + threadIdx.x] /  denom[p])  % numbers[p];
//
//					uint ii = idx[bp + p * _k1 * _Arows];
//					ddd += val[bp + p * _k1 * _Arows];
//
//					bin[p] = ii;
//
//				}
				uint bp;

				// explicitly unrolled to circumvent compiler segmentation fault
				bp = (_distSeq[nIter * blockDim.x + threadIdx.x] / denom[0])
						% numbers[0];
				ddd += val[bp + 0 * _k1 * _Arows];
				bin[0] = idx[bp + 0 * _k1 * _Arows];

				bp = (_distSeq[nIter * blockDim.x + threadIdx.x] / denom[1])
						% numbers[1];
				ddd += val[bp + 1 * _k1 * _Arows];
				bin[1] = idx[bp + 1 * _k1 * _Arows];

				bp = (_distSeq[nIter * blockDim.x + threadIdx.x] / denom[2])
						% numbers[2];
				ddd += val[bp + 2 * _k1 * _Arows];
				bin[2] = idx[bp + 2 * _k1 * _Arows];
				bp = (_distSeq[nIter * blockDim.x + threadIdx.x] / denom[3])
						% numbers[3];
				ddd += val[bp + 3 * _k1 * _Arows];
				bin[3] = idx[bp + 3 * _k1 * _Arows];

				dist[threadIdx.x] = ddd;
//
//				if (threadIdx.x >= _numDistSeq) {
//					dist[threadIdx.x] = 99999999.;
//				}

				__syncthreads();

#if 1
				// TODO _p1 + _p2
				outIdx[threadIdx.x] = calcIdxSequential(bin, _p, _nClusters1,
						_nClusters2, _c1scale) + pert * _nBinsPerDB;

#if USE_HASH
				outIdx[threadIdx.x] = outIdx[threadIdx.x] % HASH_SIZE;
#endif

				//	printf("%d --- %d \n", threadIdx.x, outIdx[threadIdx.x]);

//				if (threadIdx.x < 100)
//					printf("%d: %d %d === %f  -- %d \n", threadIdx.x, bin[0],
//							bin[1], dist[threadIdx.x], outIdx[threadIdx.x]);

				__syncthreads();

				// sort all cluster centers based on the distance
				bitonic3(dist, outIdx, blockDim.x);

				//	if (outIdx[threadIdx.x] < )
				nElem[threadIdx.x] = 0;

				nElem[threadIdx.x] = _nElemPerBin[outIdx[threadIdx.x]];

				uint maxVecPB = 2048;
				nElem[threadIdx.x] =
						(nElem[threadIdx.x] < maxVecPB) ?
								nElem[threadIdx.x] : maxVecPB;
//				nElem[threadIdx.x] = 1;

				uint nElReg = nElem[threadIdx.x];

//				if ((blockIdx.x == 2) && (threadIdx.x == 0)) {
//								for (int k = 0; k < blockDim.x; k++)
//									printf("ne %d %d %d %5d \n ", _k, outIdx[k], nElem[k], nElements );
//
//								printf("\n");
//							}

				__syncthreads();

				scan_block2(nElem, true);

				if ((threadIdx.x > 0)
						&& (nElem[threadIdx.x - 1] + nElements) >= _k) {
					nElReg = 0;
				}

				__syncthreads();
				if (threadIdx.x == 0)
					nElements += nElem[blockDim.x - 1];

				__syncthreads();
				if (nElReg)
					nElem[threadIdx.x] = 1;
				else
					nElem[threadIdx.x] = 0;

				__syncthreads();
				scan_block2(nElem, false);

				if (nElReg) {
					uint pos = nElem[threadIdx.x] + nOutBins;
					if (pos < _maxOutBin) {
						_assign[(iter * 1 + pert) * _maxOutBin
								+ pos] = outIdx[threadIdx.x];
//						if (iter == 3) {
//							printf("outputBin %d at pos %d %d \n", outIdx[threadIdx.x], pos, ((iter * 1 + pert) * _maxOutBin
//								+ pos) );
//						}
					}
				}

				__syncthreads();

				if (threadIdx.x == 0) {
					nOutBins += nElem[blockDim.x - 1];

					nIter++;

//					printf("iter %d, nOutBins %d, nEelm: %d \n", nIter,
//							nOutBins, nElements);
				}
#endif
				__syncthreads();
			}

			__syncthreads();

			if (threadIdx.x == 0) {

				_nBins[iter * 1 + pert] =
						(nOutBins > _maxOutBin) ? _maxOutBin : nOutBins;
//				printf("iter %d, nBins %d, nElements: %d \n ", nIter, nOutBins,
//						nElements);
			}
		}
	}
}
#endif

__global__ void binSortKernel(uint* _assign, const float* _dists,
		const uint* _nBins, uint _maxBin, uint _N) {

	extern __shared__ float shm[];

	float* shmIter = shm;

	float* dist = shmIter;
	shmIter += 4096;
	uint* idx = (uint*) shmIter;
	shmIter += 4096;

	if ((blockIdx.x == 0) && (threadIdx.x == 0))
		printf("binSortKernel \n");
	// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _N; iter += gridDim.x) {

		uint nB = (_nBins[iter] < 4096) ? _nBins[iter] : 4096;

		// load bin sequence;

		uint p = threadIdx.x;
		for (; p < nB; p += blockDim.x) {
			dist[p] = _dists[iter * _maxBin + p];
			idx[p] = _assign[iter * _maxBin + p];
		}
		for (; p < 4096; p += blockDim.x) {
			dist[p] = 10000000000.;
			idx[p] = 0;
		}

		__syncthreads();

		bitonicLarge(dist, idx, 4096);

		// write out the result
		for (int p = threadIdx.x; p < _nBins[iter]; p += blockDim.x) {
			_assign[iter * _maxBin + p] = idx[threadIdx.x];
		}

		__syncthreads();

	}
}
__global__ void selectBinKernelUnsorted(uint* _assign, float* _dists,
		uint* _nBins, const float *_assignVal, const uint* _assignIdx,
		const uint* _nElemPerBin, uint _Arows, uint _Brows, uint _p, uint _vl,
		uint _nClusters1, uint _nClusters2, uint _k1, uint _kMax, uint _k,
		uint _maxTrials, uint _maxOutBin, uint _c1scale, const uint *_distSeq,
		uint _numDistSeq, uint _distCluster, 
		uint _nBinsPerDB) {

// instead of the Dijkstra do the brute-force thing
	extern __shared__ float shm[];

	float* shmIter = shm;

	float* val = shmIter;
	shmIter += _p * _kMax;
	uint* idx = (uint*) shmIter;
	shmIter += _p * _kMax;
	uint* numbers = (uint*) shmIter;
	shmIter += _p;
	uint* denom = (uint*) shmIter;
	shmIter += _p;

	uint* outIdx = (uint*) shmIter;
	shmIter += blockDim.x;

	float* dist = (float*) shmIter;
	shmIter += blockDim.x;

	uint* nElem = (uint*) shmIter;
	shmIter += blockDim.x;

	uint* sBins = (uint*) shmIter;
	shmIter += 4 * blockDim.x;

	uint& nOutBins = *(uint*) shmIter;
	shmIter += 1;

	uint& nElements = *(uint*) shmIter;
	shmIter += 1;

	uint& nIter = *(uint*) shmIter;
	shmIter += 1;

	if (threadIdx.x < _p) {
		numbers[threadIdx.x] = _distCluster;
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		denom[0] = 1;
		for (int i = 1; i < _p; i++) {
			denom[i] = denom[i - 1] * numbers[i - 1];
		}
	}

//	if (threadIdx.x == 0)
//				printf("string select Bin \n \n \n");
//
//			__syncthreads();

	__syncthreads();

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

//		if (threadIdx.x == 0)
//			printf("iter %d \n", iter);

		__syncthreads();

		for (uint pert = 0; pert < 1; pert++) {
			__syncthreads();
			// read the sorted assignment
			for (int p = 0; p < _p; p++) {
				if (threadIdx.x < _kMax) {
					val[threadIdx.x + p * _kMax] = _assignVal[(iter
							* 1 + pert) * _k1 * _p * _Arows
							+ p * _k1 * _Arows + threadIdx.x];
					idx[threadIdx.x + p * _kMax] = _assignIdx[(iter
							* 1 + pert) * _k1 * _p * _Arows
							+ p * _k1 * _Arows + threadIdx.x];

				}
			}

			// TODO loop multiple times to include sufficiently many bins at the end

			if (threadIdx.x == 0) {
				nOutBins = 0;
				nElements = 0;
				nIter = 0;

			}

//			if (threadIdx.x == 0)
//				printf("before iter %d \n", nIter);
			__syncthreads();

			while ((nElements < _k) && (nIter < _maxTrials)
					&& (nOutBins < _maxOutBin)) {

				__syncthreads();
//				if (threadIdx.x == 0)
//					printf("iter %d \n", nIter);
//				__syncthreads();

				// generate all possible bins within the bounds given by numbers[]
				// calc the corresponding binIdx in the DB and the distance to the cluster center
//				dist[threadIdx.x] = 0.;

				// TODO fix 4
				// uint bin[4];				// maximum number for p

				uint* bin = sBins + _p * threadIdx.x;

				float ddd = 0.;
//				for (int p = 0; p < _p; p++) {
//					uint bp = (_distSeq[nIter * blockDim.x + threadIdx.x] /  denom[p])  % numbers[p];
//
//					uint ii = idx[bp + p * _k1 * _Arows];
//					ddd += val[bp + p * _k1 * _Arows];
//
//					bin[p] = ii;
//
//				}
				uint bp;

				// explicitly unrolled to circumvent compiler segmentation fault
				bp = (_distSeq[nIter * blockDim.x + threadIdx.x] / denom[0])
						% numbers[0];
				ddd += val[bp + 0 * _kMax];
				bin[0] = idx[bp + 0 * _kMax];

				bp = (_distSeq[nIter * blockDim.x + threadIdx.x] / denom[1])
						% numbers[1];
				ddd += val[bp + 1 * _kMax];
				bin[1] = idx[bp + 1 * _kMax];

				bp = (_distSeq[nIter * blockDim.x + threadIdx.x] / denom[2])
						% numbers[2];
				ddd += val[bp + 2 * _kMax];
				bin[2] = idx[bp + 2 * _kMax];
				bp = (_distSeq[nIter * blockDim.x + threadIdx.x] / denom[3])
						% numbers[3];
				ddd += val[bp + 3 * _kMax];
				bin[3] = idx[bp + 3 * _kMax];

				dist[threadIdx.x] = ddd;
//
//				if (threadIdx.x >= _numDistSeq) {
//					dist[threadIdx.x] = 99999999.;
//				}

				__syncthreads();

#if 1
				// TODO _p1 + _p2
				outIdx[threadIdx.x] = calcIdxSequential(bin, _p, _nClusters1,
						_nClusters2, _c1scale) + pert * _nBinsPerDB;

				__syncthreads();

#if USE_HASH
				outIdx[threadIdx.x] = outIdx[threadIdx.x] % HASH_SIZE;
#endif

				if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
					printf("binIdx: %d %d %d %d -> %d \n", bin[0], bin[1],
							bin[2], bin[3], outIdx[threadIdx.x]);
				}

				//	printf("%d --- %d \n", threadIdx.x, outIdx[threadIdx.x]);

//				if (threadIdx.x < 100)
//					printf("%d: %d %d === %f  -- %d \n", threadIdx.x, bin[0],
//							bin[1], dist[threadIdx.x], outIdx[threadIdx.x]);

				__syncthreads();

				//	if (outIdx[threadIdx.x] < )
				nElem[threadIdx.x] = 0;

				nElem[threadIdx.x] = _nElemPerBin[outIdx[threadIdx.x]];

				uint maxVecPB = 2;
				nElem[threadIdx.x] =
						(nElem[threadIdx.x] < maxVecPB) ?
								nElem[threadIdx.x] : maxVecPB;
//				nElem[threadIdx.x] = 1;

				uint nElReg = nElem[threadIdx.x];

//				if ((blockIdx.x == 2) && (threadIdx.x == 0)) {
//								for (int k = 0; k < blockDim.x; k++)
//									printf("ne %d %d %d %5d \n ", _k, outIdx[k], nElem[k], nElements );
//
//								printf("\n");
//							}

				__syncthreads();

				scan_block2(nElem, true);

				if ((threadIdx.x > 0)
						&& (nElem[threadIdx.x - 1] + nElements) >= _k) {
					nElReg = 0;
				}

				__syncthreads();
				if (threadIdx.x == 0)
					nElements += nElem[blockDim.x - 1];

				__syncthreads();
				if (nElReg)
					nElem[threadIdx.x] = 1;
				else
					nElem[threadIdx.x] = 0;

				__syncthreads();
				scan_block2(nElem, false);

				if (nElReg) {
					uint pos = nElem[threadIdx.x] + nOutBins;
					if (pos < _maxOutBin) {
						_assign[(iter * 1 + pert) * _maxOutBin
								+ pos] = outIdx[threadIdx.x];
						_dists[(iter * 1 + pert) * _maxOutBin
								+ pos] = dist[threadIdx.x];
//						if (iter == 3) {
//							printf("outputBin %d at pos %d %d \n", outIdx[threadIdx.x], pos, ((iter * 1 + pert) * _maxOutBin
//								+ pos) );
//						}
					}
				}

				__syncthreads();

				if (threadIdx.x == 0) {
					nOutBins += nElem[blockDim.x - 1];

					nIter++;

//					printf("iter %d, nOutBins %d, nEelm: %d \n", nIter,
//							nOutBins, nElements);
				}
#endif
				__syncthreads();
			}

			__syncthreads();

			if (threadIdx.x == 0) {

				_nBins[iter * 1 + pert] =
						(nOutBins > _maxOutBin) ? _maxOutBin : nOutBins;
				printf("iter %d, nBins %d, nElements: %d \n ", nIter, nOutBins,
						nElements);
			}
		}
	}
}

/** approximates the slope corresponding to the triangle with the smallest elements in the tensor product matrix spanned by val0 and val1.
 * The slope is sampled at position p = sqrt(2.f * _N) and p-1 for robustness.
 * The resulting slope index is stored in *_slopeIdx.
 */
inline __device__ void computeSlopeIdx(uint* _slopeIdx, const float *_val0,
		const float* _val1, uint _N) {

	// estimates the slope from the two last rows / column
	if (threadIdx.x == 0) {

		uint sampleIdx = sqrtf(2.f * _N);
		float slope = (_val1[sampleIdx] + _val1[sampleIdx - 1] - 2 * _val1[0])
				/ (_val0[sampleIdx] + _val0[sampleIdx - 1] - 2 * _val0[0]);

//		slope = 1./slope;

		int si = roundf(logf(slope) / logf(ANISO_BASE)) + (NUM_ANISO_DIR / 2);
		si = (si >= NUM_ANISO_DIR) ? (NUM_ANISO_DIR - 1) : si;
		si = (si < 0) ? 0 : si;

//		si = 5;

		*_slopeIdx = si;

//		printf("slope %d  %f -> %d \n ", _N, slope, si) ;
	}
	__syncthreads();
}

/** given the two vectors val0 and val1 with corresponding index data idx0 and idx1 the function returns an almost sorted ascending list of N elements of the tensor product matrix and the corresponding index information.
 * It uses the predefined heuristic distSeq (to be selected for the actual slope) with padding distCluster.
 * Indices are computed as idx0[i] * factor + idx1[j] for multi-dimensional addressing
 *
 */
inline __device__ void generate2DBins(float* _dist, uint* _bin, uint _N,
		const float *_val0, const float* _val1, const uint* _idx0,
		const uint* _idx1, const uint* _distSeq, uint _distCluster,
		uint _factor) {

	if (threadIdx.x < _N) {

		uint seqIdx = _distSeq[threadIdx.x];

		_dist[threadIdx.x] = _val0[seqIdx % _distCluster]
				+ _val1[seqIdx / _distCluster];

		_bin[threadIdx.x] = (_idx0[seqIdx % _distCluster] * _factor
				+ _idx1[seqIdx / _distCluster]);
	}

	__syncthreads();
}

inline __device__ void generate2DBins(float* _dist, uint* _bin, uint _N,
		const float *_val0, const float* _val1, const uint* _idx0,
		const uint* _idx1, const uint* _distSeq, uint _distCluster,
		uint _factor, uint _maxCoord) {

	if (threadIdx.x < _N) {

		uint seqIdx = _distSeq[threadIdx.x];

		uint x = seqIdx % _distCluster;
		uint y = seqIdx / _distCluster;

		if ((x < _maxCoord) && (y < _maxCoord)) {
			_dist[threadIdx.x] = _val0[x] + _val1[y];

			_bin[threadIdx.x] = (_idx0[x] * _factor + _idx1[y]);
		} else {
			_dist[threadIdx.x] = 99999999999.;
			_bin[threadIdx.x] = 0.;

		}
	}

	__syncthreads();
}

__global__ void selectBinKernel2D2Parts(uint* _assign, float* _dists,
		uint _maxOutBin, uint _nOutBin, const float *_assignVal,
		const uint* _assignIdx, uint _Arows, uint _Brows, uint _p,
		uint _nClusters1, uint _nClusters2, uint _k1, uint _kMax,
		const uint *_distSeq, uint _distCluster) {

	assert((2 * _nOutBin) < _maxOutBin);
	// prepare two-times 2D sorted lists using anisotropic heuristics and sorting
	extern __shared__ float shm[];

	float* shmIter = shm;

	float* val = shmIter;
	shmIter += 2 * _kMax;
	uint* idx = (uint*) shmIter;
	shmIter += 2 * _kMax;

	uint* outIdx = (uint*) shmIter;
	shmIter += blockDim.x;

	float* dist = (float*) shmIter;
	shmIter += blockDim.x;

	uint* slopeIdx = (uint*) shmIter;
	shmIter += 1;

	// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

//		if (threadIdx.x == 0)
//			printf("iter %d \n", iter);

		__syncthreads();

		for (uint pert = 0; pert < 1; pert++) {
			__syncthreads();

			// create sorted bin sequence for every two parts
			for (int pIter = 0; pIter < (_p / 2); pIter++) {

				// read the sorted assignment
				if (threadIdx.x < _kMax) {
					val[threadIdx.x] =
							_assignVal[(iter * 1 + pert) * _k1
									* _p * _Arows + (2 * pIter) * _k1 * _Arows
									+ threadIdx.x];
					idx[threadIdx.x] =
							_assignIdx[(iter * 1 + pert) * _k1
									* _p * _Arows + (2 * pIter) * _k1 * _Arows
									+ threadIdx.x];

					val[threadIdx.x + _kMax] = _assignVal[(iter
							* 1 + pert) * _k1 * _p * _Arows
							+ (2 * pIter + 1) * _k1 * _Arows + threadIdx.x];
					idx[threadIdx.x + _kMax] = _assignIdx[(iter
							* 1 + pert) * _k1 * _p * _Arows
							+ (2 * pIter + 1) * _k1 * _Arows + threadIdx.x];

				}

				__syncthreads();

				// generate first sequence

				computeSlopeIdx(slopeIdx, val, val + _kMax, _nOutBin);

				generate2DBins(dist, outIdx, _nOutBin, val, val + _kMax, idx,
						idx + _kMax, _distSeq + *slopeIdx * NUM_DISTSEQ,
						_distCluster, _nClusters1 * _nClusters2, _kMax);

				if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
					printf("2Parts: binIdx: %d %d -> %d \n", idx[0], idx[_kMax],
							outIdx[0]);
					printf("scale: %d \n", _nClusters1 * _nClusters2);
				}

				bitonic3(dist, outIdx, _nOutBin);

				// store intermediate result

				if (threadIdx.x < _nOutBin) {

					uint pos = (iter * 1 + pert) * _maxOutBin
							+ pIter * _nOutBin + threadIdx.x;
					_dists[pos] = dist[threadIdx.x];
					_assign[pos] = outIdx[threadIdx.x];
				}

				__syncthreads();
			}
		}
	}
}

/** produces a sequence of non-empty (almost sorted) candidate bins
 * It assumes that the sorted merged lists of the parts 0,1 and 2,3 are stored in _assign and dists
 *
 */
__global__ void selectBinKernel2DFinal(uint* _assign, float* _dists,
		uint* _nBins, const uint* _nElemPerBin, uint _nInBin, uint _Brows,
		uint _nClusters1, uint _nClusters2, uint _maxTrials, uint _k,
		uint _maxOutBin, const uint *_distSeq, uint _distCluster,
		 uint _nBinsPerDB) {

	// prepare two-times 2D sorted lists using anisotropic heuristics and sorting
	extern __shared__ float shm[];

	float* shmIter = shm;

	float* val = shmIter;
	shmIter += 2 * _nInBin;
	uint* idx = (uint*) shmIter;
	shmIter += 2 * _nInBin;

	uint* outIdx = (uint*) shmIter;
	shmIter += blockDim.x;

	float* dist = (float*) shmIter;
	shmIter += blockDim.x;

	uint* nElem = (uint*) shmIter;
	shmIter += blockDim.x;

	uint* slopeIdx = (uint*) shmIter;
	shmIter += 1;

	uint& nOutBins = *(uint*) shmIter;
	shmIter += 1;

	uint& nElements = *(uint*) shmIter;
	shmIter += 1;

	uint& nIter = *(uint*) shmIter;
	shmIter += 1;

	// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

//		if (threadIdx.x == 0)
//			printf("iter %d \n", iter);

		__syncthreads();

		for (uint pert = 0; pert < 1; pert++) {
			__syncthreads();

			// read the sorted output from the previous 2D merges.
			if (threadIdx.x < _nInBin) {
				uint pos = (iter * 1 + pert) * _maxOutBin
						+ threadIdx.x;
				// read first vector
				val[threadIdx.x] = _dists[pos];
				idx[threadIdx.x] = _assign[pos];

				// read second vector
				val[threadIdx.x + _nInBin] = _dists[pos + _nInBin];
				idx[threadIdx.x + _nInBin] = _assign[pos + _nInBin];

			}

			__syncthreads();

#if 1
			// determine Slope for the entire sequence (same slope for all iterations)
			computeSlopeIdx(slopeIdx, val, val + _nInBin, 1024);

			// loop multiple times to include sufficiently many bins at the end

			if (threadIdx.x == 0) {
				nOutBins = 0;
				nElements = 0;
				nIter = 0;

				if (blockIdx.x == 0)
					printf("slope %d \n ", *slopeIdx);

			}

			__syncthreads();

			while ((nElements < _k) && (nIter < _maxTrials)
					&& (nOutBins < _maxOutBin)) {
				__syncthreads();

				generate2DBins(dist, outIdx, blockDim.x, val, val + _nInBin,
						idx, idx + _nInBin,
						_distSeq + *slopeIdx * NUM_DISTSEQ + nIter * blockDim.x,
						_distCluster,
						_nClusters1 * _nClusters2 * _nClusters1 * _nClusters2,
						_nInBin);

				outIdx[threadIdx.x] = outIdx[threadIdx.x] % HASH_SIZE;

				bitonic3(dist, outIdx, blockDim.x);

				nElem[threadIdx.x] = _nElemPerBin[outIdx[threadIdx.x]];

//				if (nElem[threadIdx.x] >= 32)
//					nElem[threadIdx.x] = 0;

				uint maxVecPB = 2;
				nElem[threadIdx.x] =
						(nElem[threadIdx.x] < maxVecPB) ?
								nElem[threadIdx.x] : maxVecPB;

//				// TODO
//				nElem[threadIdx.x] = 1;

				uint nElReg = nElem[threadIdx.x];

				__syncthreads();

				scan_block2(nElem, true);

				if ((threadIdx.x > 0)
						&& (nElem[threadIdx.x - 1] + nElements) >= _k) {
					nElReg = 0;
				}

				__syncthreads();

				if (threadIdx.x == 0)
					nElements += nElem[blockDim.x - 1];

				__syncthreads();

				if (nElReg)
					nElem[threadIdx.x] = 1;
				else
					nElem[threadIdx.x] = 0;

				__syncthreads();
				scan_block2(nElem, false);

				if (nElReg) {
					uint pos = nElem[threadIdx.x] + nOutBins;
					if (pos < _maxOutBin) {
						_assign[(iter * 1 + pert) * _maxOutBin
								+ pos] = outIdx[threadIdx.x];
						_dists[(iter * 1 + pert) * _maxOutBin
								+ pos] = dist[threadIdx.x];
						//						if (iter == 3) {
						//							printf("outputBin %d at pos %d %d \n", outIdx[threadIdx.x], pos, ((iter * 1 + pert) * _maxOutBin
						//								+ pos) );
						//						}
					}
				}

				__syncthreads();

				if (threadIdx.x == 0) {
					nOutBins += nElem[blockDim.x - 1];

					nIter++;

					//					printf("iter %d, nOutBins %d, nEelm: %d \n", nIter,
					//							nOutBins, nElements);
				}
				__syncthreads();
			}
#endif

			__syncthreads();

			if (threadIdx.x == 0) {

				_nBins[iter * 1 + pert] =
						(nOutBins > _maxOutBin) ? _maxOutBin : nOutBins;
//				printf("iter %d, nBins %d, nElements: %d \n ", nIter, nOutBins,
//						nElements);
			}
		}

	}
}

__global__ void selectBinKernelFast(uint* _assign, uint* _nBins,
		const float *_assignVal, const uint* _assignIdx,
		const uint* _nElemPerBin, uint _Arows, uint _Brows, uint _p, uint _vl,
		uint _nClusters1, uint _nClusters2, uint _k1, uint _k, uint _maxTrials,
		uint _maxOutBin, uint _c1scale, const uint *_distSeq, uint _numDistSeq,
		uint _distCluster,  uint _nBinsPerDB) {

// instead of the Dijkstra do the brute-force thing
	extern __shared__ float shm[];

	float* shmIter = shm;

	float* val = shmIter;
	shmIter += _p * _k1 * _Arows;
	uint* idx = (uint*) shmIter;
	shmIter += _p * _k1 * _Arows;
	uint* numbers = (uint*) shmIter;
	shmIter += _p;
	uint* denom = (uint*) shmIter;
	shmIter += _p;

	uint* outIdx = (uint*) shmIter;
	shmIter += blockDim.x;

	float* dist = (float*) shmIter;
	shmIter += blockDim.x;

	uint* nElem = (uint*) shmIter;
	shmIter += blockDim.x;

	uint& nOutBins = *(uint*) shmIter;
	shmIter += 1;

	uint& nElements = *(uint*) shmIter;
	shmIter += 1;

	uint& nIter = *(uint*) shmIter;
	shmIter += 1;

	if (threadIdx.x < _p) {
		numbers[threadIdx.x] = _distCluster;
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		denom[0] = 1;
		for (int i = 1; i < _p; i++) {
			denom[i] = denom[i - 1] * numbers[i - 1];
		}
	}

	__syncthreads();

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

//		if (threadIdx.x == 0)
//			printf("iter %d \n", iter);
//
//		__syncthreads();

		for (uint pert = 0; pert < 1; pert++) {
			__syncthreads();
			// read the sorted assignment
			for (int p = 0; p < _p; p++) {
				if (threadIdx.x < _k1 * _Arows) {
					val[threadIdx.x + p * _k1 * _Arows] = _assignVal[(iter
							* 1 + pert) * _k1 * _p * _Arows
							+ p * _k1 * _Arows + threadIdx.x];
					idx[threadIdx.x + p * _k1 * _Arows] = _assignIdx[(iter
							* 1 + pert) * _k1 * _p * _Arows
							+ p * _k1 * _Arows + threadIdx.x];

				}
			}

			// TODO loop multiple times to include sufficiently many bins at the end

			if (threadIdx.x == 0) {
				nOutBins = 0;
				nElements = 0;
				nIter = 0;

			}

			__syncthreads();

			while ((nElements < _k) && (nIter < _maxTrials)
					&& (nOutBins < _maxOutBin)) {

// generate all possible bins within the bounds given by numbers[]
// calc the corresponding binIdx in the DB and the distance to the cluster center
//				dist[threadIdx.x] = 0.;

// TODO fix 4
// uint bin[4];				// maximum number for p

				float ddd = 0.;
				uint oIdx = 0;
				for (int p = 0; p < _p; p++) {
					uint bp = (_distSeq[nIter * blockDim.x + threadIdx.x]
							/ denom[p]) % numbers[p];

					ddd += val[bp + p * _k1 * _Arows];
					oIdx = (oIdx * _nClusters1 * _nClusters2)
							+ idx[bp + p * _k1 * _Arows];
				}

				dist[threadIdx.x] = ddd;
				outIdx[threadIdx.x] = oIdx % HASH_SIZE;

				__syncthreads();

// sort all cluster centers based on the distance
//				bitonic3(dist, outIdx, blockDim.x);

//	if (outIdx[threadIdx.x] < )
//				nElem[threadIdx.x] = _nElemPerBin[outIdx[threadIdx.x]];
				nElem[threadIdx.x] = 1;

#if 0
				uint nElReg = nElem[threadIdx.x];

				__syncthreads();

				scan_block2(nElem, true);

				if ((nElem[threadIdx.x] + nElements) >= _k) {
					nElReg = 0;
				}
				__syncthreads();
				if (threadIdx.x == 0)
				nElements += nElem[blockDim.x - 1];
				__syncthreads();
				if (nElReg)
				nElem[threadIdx.x] = 1;
				else
				nElem[threadIdx.x] = 0;

				__syncthreads();
				scan_block2(nElem, true);

				if (nElReg) {
					uint pos = nElem[threadIdx.x] + nOutBins;
					if (pos < _maxOutBin)
					_assign[(iter * 1 + pert) * _maxOutBin
					+ pos] = outIdx[threadIdx.x];
				}

				__syncthreads();

				if (threadIdx.x == 0) {
					nOutBins += nElem[blockDim.x - 1];

					nIter++;
				}
#else

				uint pos = threadIdx.x + nOutBins;
				if (pos < _maxOutBin)
					_assign[(iter * 1 + pert) * _maxOutBin + pos] =
							outIdx[threadIdx.x];
				if (threadIdx.x == 0) {
					nOutBins += blockDim.x;

					nIter++;
				}
#endif
				__syncthreads();
			}

			__syncthreads();

			if (threadIdx.x == 0) {

				_nBins[iter * 1 + pert] =
						(nOutBins > _maxOutBin) ? _maxOutBin : nOutBins;
//			printf("nBins %d, nElements: %d \n ", nOutBins, nElements);
			}
		}
	}
}

__global__ void selectBinKernelFast2(uint* _assign, uint* _nBins,
		const float *_assignVal, const uint* _assignIdx,
		const uint* _nElemPerBin, uint _Arows, uint _Brows, uint _p, uint _vl,
		uint _nClusters1, uint _nClusters2, uint _k1, uint _k, uint _maxTrials,
		uint _maxOutBin, uint _c1scale, const uint *_distSeq, uint _numDistSeq,
		uint _distCluster, uint _nBinsPerDB) {

// instead of the Dijkstra do the brute-force thing
	extern __shared__ float shm[];

	float* shmIter = shm;

	float* val = shmIter;
	shmIter += _p * _k1 * _Arows;
	uint* idx = (uint*) shmIter;
	shmIter += _p * _k1 * _Arows;
	uint* numbers = (uint*) shmIter;
	shmIter += _p;
	uint* denom = (uint*) shmIter;
	shmIter += _p;

	uint* outIdx = (uint*) shmIter;
	shmIter += blockDim.x;

	float* dist = (float*) shmIter;
	shmIter += blockDim.x;

	uint* nElem = (uint*) shmIter;
	shmIter += blockDim.x;

	uint& nOutBins = *(uint*) shmIter;
	shmIter += 1;

	uint& nElements = *(uint*) shmIter;
	shmIter += 1;

	uint& nIter = *(uint*) shmIter;
	shmIter += 1;

	if (threadIdx.x < _p) {
		numbers[threadIdx.x] = _distCluster;
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		denom[0] = 1;
		for (int i = 1; i < _p; i++) {
			denom[i] = denom[i - 1] * numbers[i - 1];
		}
	}

	__syncthreads();

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

//		if (threadIdx.x == 0)
//			printf("iter %d \n", iter);
//
//		__syncthreads();

		for (uint pert = 0; pert < 1; pert++) {
			__syncthreads();
			// read the sorted assignment
			for (int p = 0; p < _p; p++) {
				if (threadIdx.x < _k1 * _Arows) {
					val[threadIdx.x + p * _k1 * _Arows] = _assignVal[(iter
							* 1 + pert) * _k1 * _p * _Arows
							+ p * _k1 * _Arows + threadIdx.x];
					idx[threadIdx.x + p * _k1 * _Arows] = _assignIdx[(iter
							* 1 + pert) * _k1 * _p * _Arows
							+ p * _k1 * _Arows + threadIdx.x];

				}
			}

			// TODO loop multiple times to include sufficiently many bins at the end

			if (threadIdx.x == 0) {
				nOutBins = 0;
				nElements = 0;
				nIter = 0;

			}

			__syncthreads();

			while ((nElements < _k) && (nIter < _maxTrials)
					&& (nOutBins < _maxOutBin)) {

// generate all possible bins within the bounds given by numbers[]
// calc the corresponding binIdx in the DB and the distance to the cluster center
//				dist[threadIdx.x] = 0.;

// TODO fix 4
// uint bin[4];				// maximum number for p

				float ddd = 0.;
				uint oIdx = 0;
				for (int p = 0; p < _p; p++) {
					uint bp = (_distSeq[nIter * blockDim.x + threadIdx.x]
							/ denom[p]) % numbers[p];

					ddd += val[bp + p * _k1 * _Arows];
					oIdx = (oIdx * _nClusters1 * _nClusters2)
							+ idx[bp + p * _k1 * _Arows];
				}

				dist[threadIdx.x] = ddd;
				outIdx[threadIdx.x] = oIdx % HASH_SIZE;

				__syncthreads();

// sort all cluster centers based on the distance
//				bitonic3(dist, outIdx, blockDim.x);

//	if (outIdx[threadIdx.x] < )
//				nElem[threadIdx.x] = _nElemPerBin[outIdx[threadIdx.x]];
//				nElem[threadIdx.x] = 1;

#if 1
				uint nElReg = _nElemPerBin[oIdx % HASH_SIZE];

				if (nElReg)
					nElem[threadIdx.x] = 1;
				else
					nElem[threadIdx.x] = 0;

				__syncthreads();
				scan_block2(nElem, true);

				if (nElReg) {
					uint pos = nElem[threadIdx.x] + nOutBins;
					if (pos < _maxOutBin)
						_assign[(iter * 1 + pert) * _maxOutBin
								+ pos] = outIdx[threadIdx.x];
				}

				__syncthreads();

				if (threadIdx.x == 0) {
					nOutBins += nElem[blockDim.x - 1];

//					nOutBins += blockDim.x;

					nIter++;
				}
#else

				uint pos = threadIdx.x + nOutBins;
				if (pos < _maxOutBin)
				_assign[(iter * 1 + pert) * _maxOutBin + pos] =
				outIdx[threadIdx.x];

				__syncthreads();
				if (threadIdx.x == 0) {
					nOutBins += blockDim.x;

					nIter++;
				}
#endif
				__syncthreads();
			}

			__syncthreads();

			if (threadIdx.x == 0) {

				_nBins[iter * 1 + pert] =
						(nOutBins > _maxOutBin) ? _maxOutBin : nOutBins;
//			printf("nBins %d, nElements: %d \n ", nOutBins, nElements);
			}
		}
	}
}

void PerturbationProTree::getBins(uint *_bins, uint *_nBins,
		const float *_assignVal, const uint *_assignIdx, uint _N, uint _k1,
		uint _k2, uint _maxBins) {

//	uint nThreads = 64; // 32;
	uint nThreads = 1024;

	dim3 block(nThreads, 1, 1);
	dim3 grid((_N < 1024) ? _N : 1024, 1, 1);

	cudaMemset(_bins, 0, 1 * _N * _maxBins * sizeof(uint));

	cudaMemset(_nBins, 0, 1 * _N * sizeof(uint));

//	uint maxTrials = 20 * idiv(_maxBins, nThreads);

	uint maxTrials = 5 * idiv(_maxBins, nThreads);

	maxTrials = 16;				// TODO

	uint shm = (2 * d_p * _k1 * d_nClusters2 + 2 * d_p + 3 * nThreads + nThreads
			+ 3 + 4 * nThreads) * sizeof(float);

	uint c1scale = d_nClusters2;

	std::cout << "getBins shmSize " << shm << std::endl;
//	selectBinKernel<<<grid, block, shm>>>(_bins, _nBins, _assignVal, _assignIdx,
//			d_binCounts, d_nClusters2, _N, d_p, d_vl, d_nClusters, d_nClusters2,
//			_k1, _k2, maxTrials, _maxBins, c1scale, d_distSeq, d_numDistSeq,
//			d_distCluster, d_nDBs, d_nBins);

	shm = (2 * d_p * _k1 * d_nClusters2 + 2 * d_p + 3 * nThreads + 3)
			* sizeof(float);
	selectBinKernelFast2<<<grid, block, shm>>>(_bins, _nBins, _assignVal,
			_assignIdx, d_binCounts, d_nClusters2, _N, d_p, d_vl, d_nClusters,
			d_nClusters2, _k1, _k2, maxTrials, _maxBins, c1scale, d_distSeq,
			d_numDistSeq, d_distCluster, d_nBins);

	checkCudaErrors(cudaDeviceSynchronize());

}
void PerturbationProTree::getBIGBins(uint *_bins, uint *_nBins,
		const float *_assignVal, const uint *_assignIdx, uint _N, uint _k1,
		uint _k2, uint _maxBins) {

//	uint nThreads = 64; // 32;
	uint nThreads = 1024;
//	uint nThreads = 512;

	dim3 block(nThreads, 1, 1);
	dim3 grid((_N < 1024) ? _N : 1024, 1, 1);

	cudaMemset(_bins, 0, 1 * _N * _maxBins * sizeof(uint));

	cudaMemset(_nBins, 0, 1 * _N * sizeof(uint));

//	uint maxTrials = 20 * idiv(_maxBins, nThreads);

	uint maxTrials = 5 * idiv(_maxBins, nThreads);

	maxTrials = 16;

	maxTrials = 64;
	maxTrials = 256;

	uint shm = (2 * d_p * _k1 * d_nClusters2 + 2 * d_p + 3 * nThreads + nThreads
			+ 3 + 4 * nThreads) * sizeof(float);

	uint c1scale = d_nClusters2;

	std::cout << "getBins shmSize " << shm << std::endl;
	selectBinKernel<<<grid, block, shm>>>(_bins, _nBins, _assignVal, _assignIdx,
			d_binCounts, d_nClusters2, _N, d_p, d_vl, d_nClusters, d_nClusters2,
			_k1, _k2, maxTrials, _maxBins, c1scale, d_distSeq, d_numDistSeq,
			d_distCluster, d_nBins);

//	shm = (2 * d_p * _k1 * d_nClusters2 + 2 * d_p + 3 * nThreads + 3)
//			* sizeof(float);
//	selectBIGBinKernelFast2<<<grid, block, shm>>>(_bins, _nBins, _assignVal,
//			_assignIdx, d_binCounts, d_nClusters2, _N, d_p, d_vl, d_nClusters,
//			d_nClusters2, _k1, _k2, maxTrials, _maxBins, c1scale, d_distSeq,
//			d_numDistSeq, d_distCluster, d_nDBs, d_nBins);

	checkCudaErrors(cudaDeviceSynchronize());

//	outputVecUint("origBin: ", _bins, 200);

	countZeros("origBins: ", _bins, _N * _maxBins);

}

void PerturbationProTree::getBIGBinsSorted(uint *_bins, uint *_nBins,
		const float *_assignVal, const uint *_assignIdx, uint _N, uint _k1,
		uint _k2, uint _maxBins) {

//	uint nThreads = 64; // 32;

	uint nThreads = 1024;
//	uint nThreads = 512;

	dim3 block(nThreads, 1, 1);
	dim3 grid((_N < 1024) ? _N : 1024, 1, 1);

	float* dists;
	cudaMalloc(&dists, 1 * _N * _maxBins * sizeof(uint));

	cudaMemset(_bins, 0, 1 * _N * _maxBins * sizeof(uint));

	cudaMemset(_nBins, 0, 1 * _N * sizeof(uint));

//	uint maxTrials = 20 * idiv(_maxBins, nThreads);

	uint maxTrials = 5 * idiv(_maxBins, nThreads);

	maxTrials = 16;

	maxTrials = 64;
	maxTrials = 2560;

	uint kMax = 32;

	uint shm = (2 * d_p * kMax + 2 * d_p + 3 * nThreads + nThreads + 3
			+ 4 * nThreads) * sizeof(float);

	uint c1scale = d_nClusters2;

	std::cout << "getBins shmSize " << shm << std::endl;
	selectBinKernelUnsorted<<<grid, block, shm>>>(_bins, dists, _nBins,
			_assignVal, _assignIdx, d_binCounts, d_nClusters2, _N, d_p, d_vl,
			d_nClusters, d_nClusters2, _k1, kMax, _k2, maxTrials, _maxBins,
			c1scale, d_distSeq, d_numDistSeq, d_distCluster, d_nBins);

	checkCudaErrors(cudaDeviceSynchronize());

	std::cout << "now sorting " << std::endl;
	block = dim3(1024, 1, 1);
	shm = (2 * 4096) * sizeof(float);

	binSortKernel<<<grid, block, shm>>>(_bins, dists, _nBins, _maxBins, _N);

	checkCudaErrors(cudaDeviceSynchronize());

	outputVec("origDist: ", dists, 20);
	outputVecUint("origBin: ", _bins, 200);

	countZeros("origBins: ", _bins, _N * _maxBins);

	cudaFree(dists);

}

void PerturbationProTree::getBIGBins2D(uint *_bins, uint *_nBins,
		const float *_assignVal, const uint *_assignIdx, uint _N, uint _k1,
		uint _k2, uint _maxBins) {

//	uint nThreads = 64; // 32;

	uint nThreads = 1024;
//	uint nThreads = 512;

	dim3 block(nThreads, 1, 1);
	dim3 grid((_N < 1024) ? _N : 1024, 1, 1);

	float* dists;
	cudaMalloc(&dists, 1 * _N * _maxBins * sizeof(uint));

	cudaMemset(_bins, 0, 1 * _N * _maxBins * sizeof(uint));

	cudaMemset(_nBins, 0, 1 * _N * sizeof(uint));

//	uint maxTrials = 20 * idiv(_maxBins, nThreads);

	uint maxTrials = 5 * idiv(_maxBins, nThreads);

	maxTrials = 16;

	maxTrials = 64;
	maxTrials = 2560;

	uint kMax = 64;

	uint shm = (4 * kMax + 2 * nThreads + 2) * sizeof(float);

//	uint c1scale = d_nClusters2;

	uint nIntermediateBin = 256;
//	uint nIntermediateBin = 1024;

	std::cout << "getBins2D shmSize " << shm << std::endl;
	selectBinKernel2D2Parts<<<grid, block, shm>>>(_bins, dists, _maxBins,
			nIntermediateBin, _assignVal, _assignIdx, d_nClusters2, _N, d_p,
			d_nClusters, d_nClusters2, _k1, kMax, d_distSeq, d_distCluster);

	checkCudaErrors(cudaDeviceSynchronize());

//	outputVecUint("origBin: ", _bins, 20);
//	outputVec("dists2D ", dists, nIntermediateBin);
//	outputVec("dists2D -1 ", dists + nIntermediateBin, nIntermediateBin);

	nThreads = 1024;
	block = dim3(nThreads, 1, 1);

	shm = (4 * nIntermediateBin + 3 * nThreads + 5) * sizeof(float);

	std::cout << "selectBinKernel2DFinal shm: " << shm << std::endl;

//	_k2 = 4096;
	std::cout << "k2: " << _k2 << " maxBins: " << _maxBins << std::endl;

	selectBinKernel2DFinal<<<grid, block, shm>>>(_bins, dists, _nBins,
			d_binCounts, nIntermediateBin, _N, d_nClusters, d_nClusters2,
			maxTrials, _k2, _maxBins, d_distSeq, d_distCluster,
			d_nBins);

	checkCudaErrors(cudaDeviceSynchronize());

	outputVecUint("finalBin: ", _bins, 20);
//	outputVec("dists2D - final", dists, (_maxBins > 500) ? 500 : _maxBins);

//	outputVec("dists2D - final", dists, 2000);

//	outputVecUint("origBin: ", _bins, 200);

	countZeros("origBins: ", _bins, _N * _maxBins);

	cudaFree(dists);

}

__global__ void getKVectorIDsKernel(uint* _bestIdx, uint* _nVec,
		const uint* _dbIdx, const uint* _binPrefix, const uint* _binCounts,
		uint _nDBBins, const float* _Q, const uint* _assignedBins,
		const uint* _assignedNBins, uint _QN, uint _dim, uint _maxBins, uint _k,
		uint _maxVecConsider, uint _maxVecOut, 
		uint _maxNVecPerBin) {

	extern __shared__ float shm[];

	float* shmIter = shm;

	uint* nBins = (uint*) shmIter;
	shmIter += 1;

	uint &currentBin(*(uint*) shmIter);
	shmIter++;
	uint &nVec(*(uint*) shmIter);
	shmIter++;

	uint* idx = (uint*) shmIter;
	shmIter += _maxVecConsider;

	uint* val = (uint*) shmIter;
	shmIter += _maxVecConsider;

	uint* hash = val;

// in shm;

	uint count;

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _QN; iter += gridDim.x) {

		__syncthreads();

		if (threadIdx.x < 1) {
			nBins[threadIdx.x] = _assignedNBins[iter * 1
					+ threadIdx.x];
			nBins[threadIdx.x] =
					(nBins[threadIdx.x] < _maxBins) ?
							nBins[threadIdx.x] : _maxBins;
		}

		val[threadIdx.x] = _maxVecConsider;

		__syncthreads();

		// loop over the best assigned bins
		// do round robin over the different permutations
		uint _maxVecConsiderPerPert = _maxVecConsider / 1;

		uint offs = 0;

		for (int pert = 0; pert < 1; pert++) {
//		for (int pert = 1; pert < 2; pert++) {
//						for (int pert = 0; pert < 1; pert++) {

			count = 0;

			for (int bin = 0;
//					(bin < _maxBins) && (count < _maxVecConsiderPerPert);
					(bin < nBins[pert]) && (count < _maxVecConsiderPerPert);
					bin++) {
//
//				if (count >= _maxVecConsiderPerPert)
//					break;
//				if (bin >= nBins[pert])
//					continue;

				if (threadIdx.x == 0) {

					currentBin = _assignedBins[(iter * 1 + pert)
							* _maxBins + bin];

					nVec = _binCounts[currentBin];
					nVec = (nVec > _maxNVecPerBin) ? _maxNVecPerBin : nVec;

					if ((count + nVec) >= _maxVecConsiderPerPert)
						nVec = _maxVecConsiderPerPert - count - 1;

				}

				__syncthreads();

				// fetch all the vector indices for the selected bin
				for (uint v = threadIdx.x; v < nVec; v += blockDim.x) {

					idx[offs + count + v] = _dbIdx[_binPrefix[currentBin] + v];
					val[offs + count + v] = (count + v) * 1
							+ pert;
				}

				count += nVec;

				__syncthreads();
			}

			offs += count;
		}

		count = offs;

		__syncthreads();

//		if (threadIdx.x == 0) {
//			for (int i = 0; i < _maxVecConsider; i++) {
//				printf("prev: %d %d %d \n", i, idx[i], val[i]);
//			}
//		}

		// resort the array
		uint myIdx = idx[threadIdx.x];
		uint myVal = val[threadIdx.x];

		__syncthreads();

		val[threadIdx.x] = 0;

		__syncthreads();

		// write in correct order
		if (myVal < _maxVecConsider) {
			idx[myVal] = myIdx;
			val[myVal] = 2;
		}

		__syncthreads();

//		if (threadIdx.x == 0) {
//			for (int i = 0; i < _maxVecConsider; i++) {
//				printf( "num: %d %d %d \n", i, idx[i], val[i]);
//			}
//		}

		myIdx = idx[threadIdx.x];
		myVal = val[threadIdx.x];

		uint maxHash = 2048;

		uint myHash = myIdx % maxHash;

		// remove duplicates using hash
		do {
			__syncthreads();

			currentBin = 0;

			// everyone tries to write to the hash at the same time
			if (myVal == 2) {
				hash[myHash] = threadIdx.x;
			}

			// make sure the smallest index is placed in the hash map
			for (int j = 0; j < 1 - 1; j++) {
				__syncthreads();
				if (myVal == 2) {
					if (idx[hash[myHash]] == myIdx) {
						if (hash[myHash] > threadIdx.x)
							hash[myHash] = threadIdx.x;
					}
				}
				__syncthreads();
			}

			if (myVal == 2) {
				// if threadIdx.x was the smallest, keep the value
				if (hash[myHash] == threadIdx.x) {
					myVal = 1;
				} else if ((hash[myHash] < threadIdx.x)
						&& (idx[hash[myHash]] == myIdx))
					myVal = 0; // otherwise it's a duplicate
			}

			// check if there is still something to be done
			uint vote = popsift::any(myVal == 2);
			if (vote && (threadIdx.x % 32 == 0))
				currentBin = 1;

			__syncthreads();

		} while (currentBin);

//		if (myVal == 2) {
//			printf( "not yet done %d \n", threadIdx.x);
//		}

		__syncthreads();

		val[threadIdx.x] = myVal;

//		__syncthreads();
//
//		if (threadIdx.x == 0) {
//			for (int i = 0; i < _maxVecConsider; i++) {
//				printf("after: %d %d %d \n", i, idx[i], val[i]);
//			}
//
//		}

		__syncthreads();

		scan_block2(val, false);

		if ((myVal == 1) && (val[threadIdx.x] < _maxVecOut)) {
			_bestIdx[iter * _maxVecOut + val[threadIdx.x]] = myIdx;
		}

		if (threadIdx.x == 0) {
			count = (val[blockDim.x - 1] < _maxVecOut) ?
					val[blockDim.x - 1] : _maxVecOut;
			_nVec[iter] = count;
		}

#if 0

//		if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
//				for (int i = 0; i < count; i++)
//					printf("presort: %d %d \n", i, idx[i]);
//			}

// sort the results
		if ((threadIdx.x >= count) && (threadIdx.x < _maxVecConsider))
		val[threadIdx.x] = 999999999.;

		__syncthreads();

// sort all vectors by vector ID
		bitonic3(val, idx, _maxVecConsider);

// each vector should only appear at maximum _nPerturbation times
// count occurences
		if (threadIdx.x < count) {
			val[threadIdx.x] = 1.;
		} else if (threadIdx.x < _maxVecConsider)
		val[threadIdx.x] = 0.;

		for (int db = 1; db < 1; db++) {
			if ((threadIdx.x + db) < count) {
				if (idx[threadIdx.x] == idx[threadIdx.x + db]) {
					val[threadIdx.x]++;
				}
			}
		}

		__syncthreads();

// make sure to consider only the first occurence of each vectorID
// (remove duplicates)
		if ((threadIdx.x + 1) < count) {
			if (idx[threadIdx.x] == idx[threadIdx.x + 1]) {
				val[threadIdx.x + 1] = 0.;
			}
		}

// sort all vectorIDs descending by occurence
		bitonic3Descending(val, idx, _maxVecConsider);

		if (threadIdx.x == 0) {
			count = (count < _maxVecOut) ? count : _maxVecOut;
		}

		if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
			for (int i = 0; i < count; i++)
			printf("i: %d %d %d \n", i, idx[i], val[i]);
		}

		__syncthreads();

		if (threadIdx.x < count) {
			if (val[threadIdx.x] > 0.) {
				_bestIdx[iter * _maxVecOut + threadIdx.x] = idx[threadIdx.x];
				val[threadIdx.x] = 1.;
			}
		} else if (threadIdx.x < _maxVecConsider)
		val[threadIdx.x] = 0.;

// count number of valid vectors (reduction)
		for (uint stride = _maxVecConsider >> 1; stride > 0; stride >>= 1) {
			__syncthreads();

			if (threadIdx.x < stride)
			val[threadIdx.x] += val[threadIdx.x + stride];
		}
		__syncthreads();

		if (threadIdx.x == 0) {
			_nVec[iter] = (uint) val[0];

		}
#endif

	}
}

__global__ void getKVectorIDsKernelLarge(uint* _bestIdx, uint* _nVec,
		const uint* _dbIdx, const uint* _binPrefix, const uint* _binCounts,
		uint _nDBBins, const float* _Q, const uint* _assignedBins,
		const uint* _assignedNBins, uint _QN, uint _dim, uint _maxBins, uint _k,
		uint _maxVecConsider, uint _maxVecOut, 
		uint _maxNVecPerBin) {

	extern __shared__ float shm[];

	float* shmIter = shm;

	uint* nBins = (uint*) shmIter;
	shmIter += 1;

	uint &currentBin(*(uint*) shmIter);
	shmIter++;
	uint &nVec(*(uint*) shmIter);
	shmIter++;

	uint* idx = (uint*) shmIter;
	shmIter += _maxVecConsider;

	uint* val = (uint*) shmIter;
	shmIter += _maxVecConsider;

	uint* hash = val;

// in shm;

	uint count;

	uint nIter = _k / blockDim.x;

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _QN; iter += gridDim.x) {

		__syncthreads();

		if (threadIdx.x < 1) {
			nBins[threadIdx.x] = _assignedNBins[iter * 1
					+ threadIdx.x];
			nBins[threadIdx.x] =
					(nBins[threadIdx.x] < _maxBins) ?
							nBins[threadIdx.x] : _maxBins;
		}

		for (int i = threadIdx.x; i < _maxVecConsider; i += blockDim.x)
			val[i] = _maxVecConsider;

		__syncthreads();

		// loop over the best assigned bins
		// do round robin over the different permutations
		uint _maxVecConsiderPerPert = _maxVecConsider / 1;

		uint offs = 0;

		for (int pert = 0; pert < 1; pert++) {

			count = 0;

			for (int bin = 0;
					(bin < nBins[pert]) && (count < _maxVecConsiderPerPert);
					bin++) {

				if (threadIdx.x == 0) {

					currentBin = _assignedBins[(iter * 1 + pert)
							* _maxBins + bin];

					nVec = _binCounts[currentBin];
					nVec = (nVec > _maxNVecPerBin) ? _maxNVecPerBin : nVec;

					if ((count + nVec) >= _maxVecConsiderPerPert)
						nVec = _maxVecConsiderPerPert - count - 1;

				}

				__syncthreads();

				// fetch all the vector indices for the selected bin
				for (uint v = threadIdx.x; v < nVec; v += blockDim.x) {

					idx[offs + count + v] = _dbIdx[_binPrefix[currentBin] + v];
					val[offs + count + v] = (count + v) * 1
							+ pert;
				}

				count += nVec;

				__syncthreads();
			}

			offs += count;
		}

		count = offs;

		__syncthreads();

		// resort the array
		uint myIdx[4];
		uint myVal[4];

		for (int i = 0; i < nIter; i++) {
			myIdx[i] = idx[threadIdx.x + i * blockDim.x];
			myVal[i] = val[threadIdx.x + i * blockDim.x];
			val[threadIdx.x + i * blockDim.x] = 0;
		}

		__syncthreads();

		// write in correct order
		for (int i = 0; i < nIter; i++) {
			if (myVal[i] < _maxVecConsider) {
				idx[myVal[i]] = myIdx[i];
				val[myVal[i]] = 2;
			}
		}

		__syncthreads();

//		if (threadIdx.x == 0) {
//			for (int i = 0; i < _maxVecConsider; i++) {
//				printf("num: %d %d %d \n", i, idx[i], val[i]);
//			}
//		}

		for (int i = 0; i < nIter; i++) {
			myIdx[i] = idx[threadIdx.x + i * blockDim.x];
			myVal[i] = val[threadIdx.x + i * blockDim.x];
		}

		uint maxHash = 2048;

		uint myHash[4];
		for (int i = 0; i < nIter; i++) {
			myHash[i] = myIdx[i] % maxHash;
		}

#if 1
		// remove duplicates using hash
		do {
			__syncthreads();

			currentBin = 0;

			// everyone tries to write to the hash at the same time
			for (int i = 0; i < nIter; i++) {
				if (myVal[i] == 2) {
					hash[myHash[i]] = threadIdx.x + i * blockDim.x;
				}
			}

			// make sure the smallest index is placed in the hash map
			for (int j = 0; j < 1 - 1; j++) {
				__syncthreads();
				for (int i = 0; i < nIter; i++) {
					if (myVal[i] == 2) {
						if (idx[hash[myHash[i]]] == myIdx[i]) {
							if (hash[myHash[i]]
									> (threadIdx.x + i * blockDim.x))
								hash[myHash[i]] =
										(threadIdx.x + i * blockDim.x);
						}
					}
				}
				__syncthreads();
			}

			for (int i = 0; i < nIter; i++) {
				if (myVal[i] == 2) {
					// if threadIdx.x was the smallest, keep the value
					if (hash[myHash[i]] == (threadIdx.x + i * blockDim.x)) {
						myVal[i] = 1;
					} else if ((hash[myHash[i]] < (threadIdx.x + i * blockDim.x))
							&& (idx[hash[myHash[i]]] == myIdx[i]))
						myVal[i] = 0; // otherwise it's a duplicate
				}
			}

			// check if there is still something to be done
			uint vote = popsift::any(myVal[0] == 2);
			for (int i = 1; i < nIter; i++) {
				vote = vote || popsift::any(myVal[i] == 2);
			}

			if (vote && (threadIdx.x % 32 == 0))
				currentBin = 1;

			__syncthreads();

		} while (currentBin);
#endif

//		if (myVal == 2) {
//			printf( "not yet done %d \n", threadIdx.x);
//		}

		__syncthreads();

		for (int i = 0; i < nIter; i++)
			val[threadIdx.x + i * blockDim.x] = myVal[i];

		__syncthreads();

		scan_blockLarge(val, false, _maxVecConsider);

//		if (threadIdx.x == 0) {
//			for (int i = 0; i < _maxVecConsider; i++) {
//				printf("after: %d %d %d \n", i, idx[i], val[i]);
//			}
//
//		}
//
//		__syncthreads();

		for (int i = 0; i < nIter; i++)
			if ((myVal[i] == 1)
					&& (val[threadIdx.x + i * blockDim.x] < _maxVecOut)) {
				_bestIdx[iter * _maxVecOut + val[threadIdx.x + i * blockDim.x]] =
						myIdx[i];
			}

		if (threadIdx.x == 0) {
			count = (val[_maxVecConsider - 1] < _maxVecOut) ?
					val[_maxVecConsider - 1] : _maxVecOut;
			_nVec[iter] = count;
		}
	}
}

#if 1
__global__ void getKVectorIDsKernelFast(uint* _bestIdx, uint* _nVec,
		const uint* _dbIdx, const uint* _binPrefix, const uint* _binCounts,
		uint _nDBBins, const float* _Q, const uint* _assignedBins,
		const uint* _assignedNBins, uint _QN, uint _dim, uint _maxBins, uint _k,
		uint _maxVecConsider, uint _maxVec, 
		uint _maxNVecPerBin) {

	extern __shared__ float shm[];

	float* shmIter = shm;

	uint *currentBin = (uint*) shmIter;
	shmIter += blockDim.x;
	uint *nVec = (uint*) shmIter;
	shmIter += blockDim.x;
	uint *pos = (uint*) shmIter;
	shmIter += blockDim.x;

	uint &nBins(*(uint*) shmIter);
	shmIter++;

	uint &binsLeft(*(uint*) shmIter);
	shmIter++;

	uint &binIter(*(uint*) shmIter);
	shmIter++;

	uint &offset(*(uint*) shmIter);
	shmIter++;

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _QN; iter += gridDim.x) {

		__syncthreads();

		if (threadIdx.x == 0) {

			nBins = _assignedNBins[iter];
			nBins = (nBins < _maxBins) ? nBins : _maxBins;

			binIter = nBins / blockDim.x + 1;
			offset = 0;
		}

		__syncthreads();

		for (uint bIter = 0; bIter < binIter; bIter++) {
			uint bin = bIter * blockDim.x + threadIdx.x;

			uint nV = 0;
			if (bin < nBins) {
				currentBin[threadIdx.x] = _assignedBins[iter * _maxBins + bin];
				nV = (_binCounts[currentBin[threadIdx.x]] < _maxNVecPerBin) ?
						_binCounts[currentBin[threadIdx.x]] : _maxNVecPerBin;

			}

			pos[threadIdx.x] = nV;

			if (threadIdx.x == 0) {

				binsLeft =
						(nBins <= (blockDim.x + bin)) ?
								(nBins - bin) : blockDim.x;

//				if (blockIdx.x == 2) {
//					printf("currentBin: %d counts: %d limited: %d \n", currentBin[threadIdx.x], _binCounts[currentBin[threadIdx.x]], nV );
//				}

//				printf("maxVec: %d offset: %d pos: %d \n", _maxVec, offset,
//						pos[0]);
			}

			__syncthreads();

			scan_block2(pos, false);

			pos[threadIdx.x] += offset;

			if ((pos[threadIdx.x] + nV) > _maxVec) {
				nV = (pos[threadIdx.x] >= _maxVec) ?
						0 : (_maxVec - pos[threadIdx.x]);
			}

			nVec[threadIdx.x] = nV;

			__syncthreads();

//			if (threadIdx.x == 0) {
//				for (int i = 0; i < ((nBins < blockDim.x) ? nBins : blockDim.x); i++)
//					printf(" %5d ", nVec[i]);
//				printf("\n");
//			}

			if (bin < nBins)
				for (int v = 0; v < nVec[threadIdx.x]; v++) {
					_bestIdx[iter * _maxVec + pos[threadIdx.x] + v] =
							_dbIdx[_binPrefix[currentBin[threadIdx.x]] + v];
				}

			__syncthreads();

			if (threadIdx.x == 0) {
				offset = pos[blockDim.x - 1] + nVec[blockDim.x - 1];
			}
		}

		if (threadIdx.x == 0) {
			_nVec[iter] = (offset > _maxVec) ? _maxVec : offset;
		}
	}
}

#else
__global__ void getKVectorIDsKernelFast(uint* _bestIdx, uint* _nVec,
		const uint* _dbIdx, const uint* _binPrefix, const uint* _binCounts,
		uint _nDBBins, const float* _Q, const uint* _assignedBins,
		const uint* _assignedNBins, uint _QN, uint _dim, uint _maxBins, uint _k,
		uint _maxVecConsider, uint _maxVec, uint 1,
		uint _maxNVecPerBin) {

	extern __shared__ float shm[];

	const uint laneSize = 32;

	float* shmIter = shm;

	uint *currentBin = (uint*) shmIter;
	shmIter += blockDim.x;
	uint *nVec = (uint*) shmIter;
	shmIter += blockDim.x;
	uint *pos = (uint*) shmIter;
	shmIter += blockDim.x;

	volatile uint *binProc = (uint*) shmIter;
	shmIter += blockDim.x / laneSize;

	uint &nBins(*(uint*) shmIter);
	shmIter++;

	uint &binsLeft(*(uint*) shmIter);
	shmIter++;

	uint &binIter(*(uint*) shmIter);
	shmIter++;

	uint &offset(*(uint*) shmIter);
	shmIter++;

	uint* nProcessed = (uint*) shmIter;
	shmIter++;

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _QN; iter += gridDim.x) {

		__syncthreads();

		if (threadIdx.x == 0) {

			nBins = _assignedNBins[iter];
			nBins = (nBins < _maxBins) ? nBins : _maxBins;

			binIter = nBins / blockDim.x + 1;
			offset = 0;
		}

		__syncthreads();

		uint laneIdx = threadIdx.x % laneSize;
		uint lane = threadIdx.x / laneSize;

		for (uint bIter = 0; bIter < binIter; bIter++) {
			uint bin = bIter * blockDim.x + threadIdx.x;

			uint nV = 0;
			if (bin < nBins) {
				currentBin[threadIdx.x] = _assignedBins[iter * _maxBins + bin];
				nV = (_binCounts[currentBin[threadIdx.x]] < _maxNVecPerBin) ? _binCounts[currentBin[threadIdx.x]] : _maxNVecPerBin;

			}

			pos[threadIdx.x] = nV;

			if (threadIdx.x == 0) {
				nProcessed = 0;
				binsLeft =
				(nBins <= (blockDim.x + bin)) ?
				(nBins - bin) : blockDim.x;
			}

			__syncthreads();

			scan_block2(pos, false);

			pos[threadIdx.x] += offset;

			if ((pos[threadIdx.x] + nV) > _maxVec) {
				nV = (pos[threadIdx.x] >= _maxVec) ?
				0 : (_maxVec - pos[threadIdx.x]);
			}

			nVec[threadIdx.x] = nV;

			__syncthreads();

#if 1
			if (bin < nBins)
			for (int v = 0; v < nVec[threadIdx.x]; v++) {
				_bestIdx[iter * _maxVec + pos[threadIdx.x] + v] =
				_dbIdx[_binPrefix[currentBin[threadIdx.x]] + v];
			}
#else

			// each lane is now responsible to copy the vectorIDs for one bin
			do {
				// fetch next bin to be processed
				if (laneIdx == 0) {
					binProc[lane] = atomicInc(nProcessed, 100000);

//					if ((iter ==2 ) && (binProc[lane] < binsLeft)) {
//						printf("%d %d %d \n", binProc[lane], currentBin[binProc[lane]], nVec[binProc[lane]]);
//					}
				}
				if (binProc[lane] < binsLeft) {
					for (int v = laneIdx; v < nVec[binProc[lane]]; v +=
							laneSize)
					_bestIdx[iter * _maxVec + pos[binProc[lane]] + v] =
					_dbIdx[_binPrefix[currentBin[binProc[lane]]] + v];
				}

			}while (binProc[lane] < binsLeft);
#endif
			__syncthreads();

			if (threadIdx.x == 0) {
				offset = pos[blockDim.x - 1] + nVec[blockDim.x - 1];
			}
		}

		if (threadIdx.x == 0) {
			_nVec[iter] = (offset > _maxVec) ? _maxVec : offset;
		}
	}
}
#endif

__global__ void getKBinVectorIDsKernelFast(uint* _bestIdx, uint* _nVec,
		const float *_assignVal, const uint* _assignIdx, uint _p, uint _k1,
		uint _nClusters1, uint _nClusters2, const uint* _dbIdx,
		const uint* _binPrefix, const uint* _binCounts, uint _nDBBins,
		const uint *_distSeq, uint _numDistSeq, uint _distCluster, uint _QN,
		uint _maxBins, uint _maxVec, uint _maxNVecPerBin) {

	extern __shared__ float shm[];

	float* shmIter = shm;

//	float* val = shmIter;
//	shmIter += _p * _k1 * _Arows;
	uint* idx = (uint*) shmIter;
	shmIter += _p * _k1 * _nClusters2;
	uint* numbers = (uint*) shmIter;
	shmIter += _p;
	uint* denom = (uint*) shmIter;
	shmIter += _p;

	uint *nVec = (uint*) shmIter;
	shmIter += blockDim.x;
	uint *pos = (uint*) shmIter;
	shmIter += blockDim.x;

//	uint &nBins(*(uint*) shmIter);
//	shmIter++;

//	uint &binsLeft(*(uint*) shmIter);
//	shmIter++;

//	uint &binIter(*(uint*) shmIter);
//	shmIter++;

	uint &offset(*(uint*) shmIter);
	shmIter++;

	if (threadIdx.x < _p) {
		numbers[threadIdx.x] = _distCluster;
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		denom[0] = 1;
		for (int i = 1; i < _p; i++) {
			denom[i] = denom[i - 1] * numbers[i - 1];
		}
	}

	__syncthreads();

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _QN; iter += gridDim.x) {

		__syncthreads();

		for (int p = 0; p < _p; p++) {

			// load the best indices
			if (threadIdx.x < _k1 * _nClusters2) {
//				val[threadIdx.x + p * _k1 * _Arows] = _assignVal[(iter) * _k1
//						* _p * _Arows + p * _k1 * _Arows + threadIdx.x];
				idx[threadIdx.x + p * _k1 * _nClusters2] = _assignIdx[(iter)
						* _k1 * _p * _nClusters2 + p * _k1 * _nClusters2
						+ threadIdx.x];

			}
		}

		if (threadIdx.x == 0) {

			offset = 0;
		}

		__syncthreads();

		for (uint bIter = 0; bIter < _maxBins; bIter += blockDim.x) {
			if (offset > _maxVec)
				break;

			__syncthreads();

			uint bin = bIter + threadIdx.x;

			uint nV = 0;

			uint currentBin;
//			if (bin < nBins)
			{

//				float ddd = 0.;
				uint oIdx = 0;
				for (int p = 0; p < _p; p++) {
					uint bp = (_distSeq[bin] / denom[p]) % numbers[p];

//					ddd += val[bp + p * _k1 * _Arows];
					oIdx = (oIdx * _nClusters1 * _nClusters2)
							+ idx[bp + p * _k1 * _nClusters2];
				}

//				dist[threadIdx.x] = ddd;
				currentBin = oIdx % HASH_SIZE;

				nV = (_binCounts[currentBin] < _maxNVecPerBin) ?
						_binCounts[currentBin] : _maxNVecPerBin;

//				if (iter == 0)
//					printf( "%d %d %d %d \n", bIter, offset, nV, oIdx);

			}

			pos[threadIdx.x] = nV;

			__syncthreads();

			scan_block2(pos, false);

			pos[threadIdx.x] += offset;

			if ((pos[threadIdx.x] + nV) > _maxVec) {
				nV = (pos[threadIdx.x] >= _maxVec) ?
						0 : (_maxVec - pos[threadIdx.x]);
			}

			nVec[threadIdx.x] = nV;

//			if (bin < nBins)
			for (int v = 0; v < nVec[threadIdx.x]; v++) {
				_bestIdx[iter * _maxVec + pos[threadIdx.x] + v] =
						_dbIdx[_binPrefix[currentBin] + v];
			}

			__syncthreads();

			if (threadIdx.x == 0) {
				offset = pos[blockDim.x - 1] + nVec[blockDim.x - 1];
			}
		}

		__syncthreads();
		if (threadIdx.x == 0) {
			_nVec[iter] = (offset > _maxVec) ? _maxVec : offset;
		}
	}
}

__global__ void getPerturbationKBestVectorsKernel(float *_bestDist,
		uint* _bestIdx, const uint* _inIdx, const uint* _nVec, uint _maxVecIn,
		const float* _dbVec, const float* _Q, uint _QN, uint _dim,
		uint _maxBins, uint _k, uint _maxVec) {

	extern __shared__ float shm[];

	float* shmIter = shm + _dim;

	float* val = shmIter;
	shmIter += _maxVec;
	uint* idx = (uint*) shmIter;
	shmIter += _maxVec;

// in shm;
	uint &nVec(*(uint*) shmIter);
	shmIter++;

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _QN; iter += gridDim.x) {

		__syncthreads();

		if (threadIdx.x == 0) {
			nVec = _nVec[iter];
			nVec = (nVec < _maxVec) ? nVec : _maxVec;

			if (iter % 1000 == 0) {
				printf("nVec: %d \n", nVec);
			}
		}
		__syncthreads();

		// load query vector
		float b;
		if (threadIdx.x < _dim)
			b = _Q[iter * _dim + threadIdx.x];

		// load all indices
		for (int a = threadIdx.x; a < nVec; a += blockDim.x) {
			idx[a] = _inIdx[iter * _maxVecIn + a];

			if (idx[a] >= 1000000) {
				printf("panic: %d %d %d %d \n ", idx[a], iter, a, nVec);
			}
		}

		__syncthreads();

//		if (threadIdx.x == 0) {
//			for (int a = 0; a < nVec; a++) {
//				printf("idx: %d %d \n", a, idx[a]);
//			}
//		}

		// loop over all selected vectors
		for (int a = 0; a < nVec; a++) {

			// compute the distance to the vector
//			if (threadIdx.x < _dim) {
//				uint loc = idx[a] * _dim + threadIdx.x;
//				float v = _dbVec[loc];
//
////				if ((blockIdx.x == 90) && (a == 110))
////					printf( "got: %d %d %f \n", loc, idx[a], v);
//
//				shm[threadIdx.x] = sqr( b - v );
//			}
			if (threadIdx.x < _dim) {
//				if (idx[a] < 1000000)

				shm[threadIdx.x] = sqr(b - _dbVec[idx[a] * _dim + threadIdx.x]);
			}

			for (uint stride = _dim >> 1; stride > 0; stride >>= 1) {
				__syncthreads();

				if (threadIdx.x < stride)
					shm[threadIdx.x] += shm[threadIdx.x + stride];
			}
			__syncthreads();

			// store the result
			if (threadIdx.x == 0) {

				val[a] = shm[0];

//				printf("idx: %d dist: %f \n", idx[a], val[a]);

			}
			__syncthreads();
		}

		// sort the results
		if ((threadIdx.x >= nVec) && (threadIdx.x < _maxVec))
			val[threadIdx.x] = 10000000.;

		__syncthreads();

		bitonic3(val, idx, _maxVec);

		if ((threadIdx.x >= nVec) && (threadIdx.x < _maxVec))
			val[threadIdx.x] = 0.;

		if (threadIdx.x < _k) {
			_bestDist[iter * _k + threadIdx.x] = val[threadIdx.x];
			_bestIdx[iter * _k + threadIdx.x] = idx[threadIdx.x];
		}

		__syncthreads();

	}
}

__global__ void getPerturbationKBestVectorsKernelLarge(float *_bestDist,
		uint* _bestIdx, const uint* _inIdx, const uint* _nVec, uint _maxVecIn,
		const float* _dbVec, const float* _Q, uint _QN, uint _dim,
		uint _maxBins, uint _k, uint _maxVec) {

	extern __shared__ float shm[];

	float* shmIter = shm + _dim;

	float* val = shmIter;
	shmIter += _maxVec;
	uint* idx = (uint*) shmIter;
	shmIter += _maxVec;

// in shm;
	uint &nVec(*(uint*) shmIter);
	shmIter++;

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _QN; iter += gridDim.x) {

		__syncthreads();

		if (threadIdx.x == 0) {
			nVec = _nVec[iter];
			nVec = (nVec < _maxVec) ? nVec : _maxVec;

			if (iter % 1000 == 0) {
				printf("nVec: %d \n", nVec);
			}
		}
		__syncthreads();

		// load query vector
		float b;
		if (threadIdx.x < _dim)
			b = _Q[iter * _dim + threadIdx.x];

		// load all indices
		for (int a = threadIdx.x; a < nVec; a += blockDim.x) {
			idx[a] = _inIdx[iter * _maxVecIn + a];

			if (idx[a] >= 1000000) {
				printf("panic: %d %d %d %d \n ", idx[a], iter, a, nVec);
			}
		}

		__syncthreads();

//		if (threadIdx.x == 0) {
//			for (int a = 0; a < nVec; a++) {
//				printf("idx: %d %d \n", a, idx[a]);
//			}
//		}

		// loop over all selected vectors
		for (int a = 0; a < nVec; a++) {

			// compute the distance to the vector
//			if (threadIdx.x < _dim) {
//				uint loc = idx[a] * _dim + threadIdx.x;
//				float v = _dbVec[loc];
//
////				if ((blockIdx.x == 90) && (a == 110))
////					printf( "got: %d %d %f \n", loc, idx[a], v);
//
//				shm[threadIdx.x] = sqr( b - v );
//			}
			if (threadIdx.x < _dim) {
//				if (idx[a] < 1000000)

				shm[threadIdx.x] = sqr(b - _dbVec[idx[a] * _dim + threadIdx.x]);
			}

			for (uint stride = _dim >> 1; stride > 0; stride >>= 1) {
				__syncthreads();

				if (threadIdx.x < stride)
					shm[threadIdx.x] += shm[threadIdx.x + stride];
			}
			__syncthreads();

			// store the result
			if (threadIdx.x == 0) {

				val[a] = shm[0];

//				printf("idx: %d dist: %f \n", idx[a], val[a]);

			}
			__syncthreads();
		}

		// sort the results
		for (int i = threadIdx.x; i < _maxVec; i += blockDim.x)
			if (i >= nVec)
				val[i] = 10000000.;

		__syncthreads();

		bitonicLarge(val, idx, _maxVec);

//		for (int i  = threadIdx.x; i < _maxVec; i += blockDim)
//		if (threadIdx.x >= nVec)
//			val[i] = 0.;

		for (uint i = threadIdx.x; i < _k; i += blockDim.x)
			if (i < _k) {
				_bestDist[iter * _k + i] = val[i];
				_bestIdx[iter * _k + i] = idx[i];
			}

		__syncthreads();

	}
}

void PerturbationProTree::getKBestVectors(float *_bestDist, uint *_bestIdx,
		const uint *_bins, const uint *_nBins, uint _maxBins, const float* _Q,
		uint _QN, uint _k) {

	uint nnn = log2(_k);

//	nnn = 1024;

	std::cout << "nnn: " << nnn << std::endl;

	uint maxVecConsider = nnn;
	uint maxVecOut = nnn;
	uint nThreads = (maxVecConsider < 1024) ? maxVecConsider : 1024;
	dim3 block(nThreads, 1, 1);
	dim3 grid((_QN < 1024) ? _QN : 1024, 1, 1);

	uint hash = 2048;

	uint shmSize = (maxVecConsider
			+ ((hash > maxVecConsider) ? hash : maxVecConsider) + 1 + 10)
			* sizeof(float);

	uint *selectIdx; // array for storing nn vector IDs (size maxVecOut * _QN)
	uint *nVec;

	cudaMalloc(&selectIdx, _QN * _k * sizeof(uint));
	cudaMalloc(&nVec, _QN * sizeof(uint));

	if ((!selectIdx) || (!nVec)) {
		std::cout << "getKBestVectors: did not get memory !" << std::endl;

		exit(1);
	}

	cudaMemset(selectIdx, 0, _QN * _k * sizeof(uint));

	getKVectorIDsKernel<<<grid, block, shmSize>>>(selectIdx, nVec, d_dbIdx,
			d_binPrefix, d_binCounts, d_nBins, _Q, _bins, _nBins, _QN, d_dim,
			_maxBins, _k, maxVecConsider, maxVecOut, 80);

	checkCudaErrors(cudaDeviceSynchronize());

//	std::cout << "multi Vector IDs done" << std::endl;
////	_QN = 1;
//
//	outputVecUint("selectIdx", selectIdx, 100);

//	uint maxVec = 2 * log2(_k);
	uint maxVec = log2(_k);

	nThreads = (maxVec > d_dim) ? maxVec : d_dim;

	nThreads = (nThreads < 1024) ? nThreads : 1024;

	block = dim3(nThreads, 1, 1);

	shmSize = (d_dim + 2 * maxVec + 10) * sizeof(float);

	std::cout << "maxVec: " << maxVec << " shm: " << shmSize << std::endl;

	getPerturbationKBestVectorsKernel<<<grid, block, shmSize>>>(_bestDist,
			_bestIdx, selectIdx, nVec, maxVecOut, d_dbVec, _Q, _QN, d_dim,
			_maxBins, _k, maxVec);

	checkCudaErrors(cudaDeviceSynchronize());

	std::cout << "multiKBestVectors done " << std::endl;

	cudaFree(nVec);
	cudaFree(selectIdx);

}

void PerturbationProTree::getKBestVectorsLarge(float *_bestDist, uint *_bestIdx,
		const uint *_bins, const uint *_nBins, uint _maxBins, const float* _Q,
		uint _QN, uint _k) {

	uint nnn = log2(_k);

	std::cout << "large: nnn: " << nnn << " k:  " << _k << std::endl;

	uint maxVecConsider = nnn;
	uint maxVecOut = nnn;
	uint nThreads = (maxVecConsider < 1024) ? maxVecConsider : 1024;
	dim3 block(nThreads, 1, 1);
	dim3 grid((_QN < 1024) ? _QN : 1024, 1, 1);

	uint hash = 2048;

	uint shmSize = (maxVecConsider
			+ ((hash > maxVecConsider) ? hash : maxVecConsider) + 1 + 10)
			* sizeof(float);

	uint *selectIdx; // array for storing nn vector IDs (size maxVecOut * _QN)
	uint *nVec;

	cudaMalloc(&selectIdx, _QN * _k * sizeof(uint));
	cudaMalloc(&nVec, _QN * sizeof(uint));

	if ((!selectIdx) || (!nVec)) {
		std::cout << "getKBestVectors: did not get memory !" << std::endl;

		exit(1);
	}

	cudaMemset(selectIdx, 0, _QN * _k * sizeof(uint));

//	uint* binStart;
//	cudaMalloc(&binStart, d_nDBs * sizeof(uint));
//	cudaMemset(binStart, 0, d_nDBs * sizeof(uint));

	getKVectorIDsKernelLarge<<<grid, block, shmSize>>>(selectIdx, nVec, d_dbIdx,
			d_binPrefix, d_binCounts, d_nBins, _Q, _bins, _nBins, _QN, d_dim,
			_maxBins, _k, maxVecConsider, maxVecOut, 80);

	checkCudaErrors(cudaDeviceSynchronize());

//	std::cout << "multi Vector IDs done" << std::endl;
////	_QN = 1;
//
//	outputVecUint("selectIdx", selectIdx, 1751);

//	uint maxVec = 2 * log2(_k);
	uint maxVec = log2(_k);

	nThreads = (maxVec > d_dim) ? maxVec : d_dim;

	nThreads = (nThreads < 1024) ? nThreads : 1024;

	block = dim3(nThreads, 1, 1);

	shmSize = (d_dim + 2 * maxVec + 10) * sizeof(float);

	std::cout << "maxVec: " << maxVec << " shm: " << shmSize << std::endl;

	getPerturbationKBestVectorsKernelLarge<<<grid, block, shmSize>>>(_bestDist,
			_bestIdx, selectIdx, nVec, maxVecOut, d_dbVec, _Q, _QN, d_dim,
			_maxBins, _k, maxVec);

	checkCudaErrors(cudaDeviceSynchronize());

	std::cout << "multiKBestVectors done " << std::endl;

	cudaFree(nVec);
	cudaFree(selectIdx);

}

__global__ void rerankKernel(float* _bestDist, uint* _bestIdx,
		const uint* _inIdx, const uint* _nVec, uint _maxVecIn,
		const float* _cbDist, uint _nClusters, uint _lineParts,
		const float* _lineLambda, const uint* _lineP1, const uint* _lineP2,
		const float* _queryL1, uint _QN, uint _dim, uint _maxBins, uint _k,
		uint _maxVec) {

	extern __shared__ float shm[];

	float* shmIter = shm;

	float* queryDist = shmIter;
	shmIter += _lineParts * _nClusters;

	uint* idx = (uint*) shmIter;
	shmIter += _maxVec;
	float* val = shmIter;
	shmIter += _maxVec;

	uint &nVec(*(uint*) shmIter);
	shmIter++;

	for (uint iter = blockIdx.x; iter < _QN; iter += gridDim.x) {

		__syncthreads();

		// load queryDistance
		for (int p = 0; p < _lineParts; p++) {
			if (threadIdx.x < _nClusters)
				queryDist[p * _nClusters + threadIdx.x] =
						_queryL1[iter * _lineParts * _nClusters + p * _nClusters
								+ threadIdx.x];
//
//			if (threadIdx.x == 0) {
//				for (int c = 0; c < _nClusters; c++)
//					printf("\t %f", queryDist[p * _nClusters + c]);
//				printf("\n");
//			}
		}

		if (threadIdx.x == 0) {
			nVec = _nVec[iter];
			nVec = (nVec < _maxVec) ? nVec : _maxVec;

		}
		__syncthreads();

		// compute the distances to all line approximations
		for (int a = threadIdx.x; a < nVec; a += blockDim.x) {
			idx[a] = _inIdx[iter * _maxVecIn + a];

			float totalDist = 0.;

			for (uint p = 0; p < _lineParts; p++) {
				uint l1 = _lineP1[idx[a] * _lineParts + p];
				uint l2 = _lineP2[idx[a] * _lineParts + p];
				float lambda = _lineLambda[idx[a] * +_lineParts + p];

				float c2 = _cbDist[l2 * _nClusters * _lineParts
						+ l1 * _lineParts + p];

				float d = dist(queryDist[p * _nClusters + l1],
						queryDist[p * _nClusters + l2], c2, lambda);

				totalDist += d;

				if (!isTriangle(queryDist[p * _nClusters + l1],
						queryDist[p * _nClusters + l2], c2))
					printf("non-triangle: l1/l2 %d %d === %f %f %f = %f %f \n",
							l1, l2, queryDist[p * _nClusters + l1],
							queryDist[p * _nClusters + l2], c2, d, lambda);
			}

			val[a] = totalDist;
		}

		// sort the results
		for (int i = threadIdx.x; i < _maxVec; i += blockDim.x)
			if (i >= nVec)
				val[i] = 10000000.;

		__syncthreads();

		if (_maxVec <= blockDim.x)
			bitonic3(val, idx, _maxVec);
		else
			bitonicLarge(val, idx, _maxVec);

		for (uint i = threadIdx.x; i < _k; i += blockDim.x)
			if (i < _k) {
				_bestDist[iter * _k + i] = val[i];
				_bestIdx[iter * _k + i] = idx[i];
			}

		__syncthreads();

	}
}

__device__ void myAdd(volatile float* _a, volatile float* _b) {
	*_a += *_b;
}

__inline__ __device__ float warpReduceSum(float _val, int _p) {
	for (int stride = _p >> 1; stride > 0; stride >>= 1)
		_val += popsift::shuffle_down(_val, stride);
	return _val;
}

__global__ void rerankKernelFast(float* _bestDist, uint* _bestIdx,
		const uint* _inIdx, const uint* _nVec, uint _maxVecIn,
		const float* _cbDist, uint _nClusters, uint _lineParts,
		const float* _lineLambda, const float* _queryL1, uint _QN, uint _dim,
		uint _maxBins, uint _k, uint _maxVec) {

	extern __shared__ float shm[];

	float* shmIter = shm;

	float* queryDist = shmIter;
	shmIter += _lineParts * _nClusters;

	uint* idx = (uint*) shmIter;
	shmIter += _maxVec;
	float* val = shmIter;
	shmIter += _maxVec;

	uint* laneA = (uint*) shmIter;
	shmIter += blockDim.x / _lineParts;

//	volatile float* d = shmIter;
//	shmIter += blockDim.x;

	uint &nVec(*(uint*) shmIter);
	shmIter++;

	uint* nProcessed = (uint*) shmIter;
	shmIter++;

	for (uint iter = blockIdx.x; iter < _QN; iter += gridDim.x) {

		__syncthreads();

		// load queryDistance
		for (int p = 0; p < _lineParts; p++) {
			if (threadIdx.x < _nClusters)
				queryDist[p * _nClusters + threadIdx.x] =
						_queryL1[iter * _lineParts * _nClusters + p * _nClusters
								+ threadIdx.x];

		}

		if (threadIdx.x == 0) {
			nVec = _nVec[iter];
			nVec = (nVec < _maxVec) ? nVec : _maxVec;

			*nProcessed = 0;

		}
		__syncthreads();

#if 0

		// compute the distances to all line approximations
		for (int a = threadIdx.x; a < nVec; a += blockDim.x) {
			idx[a] = _inIdx[iter * _maxVecIn + a];

			float totalDist = 0.;

			for (uint p = 0; p < _lineParts; p++) {

				float l = _lineLambda[idx[a] * +_lineParts + p];
				lineDescr& line( *( (lineDescr*)&l));

				uint l1 = line.p1;
				uint l2 = line.p2;
				float lambda = toFloat(line.lambda);

				float c2 = _cbDist[l2 * _nClusters * _lineParts
				+ l1 * _lineParts + p];

				float d = dist(queryDist[p * _nClusters + l1],
						queryDist[p * _nClusters + l2], c2, lambda);

				totalDist += d;
//
//				if (!isTriangle(queryDist[p * _nClusters + l1],
//								queryDist[p * _nClusters + l2], c2))
//				printf("non-triangle: l1/l2 %d %d === %f %f %f = %f %f \n",
//						l1, l2, queryDist[p * _nClusters + l1],
//						queryDist[p * _nClusters + l2], c2, d, lambda);
			}

//			if (iter == 0)
//			printf("%d %f \n", a, totalDist);
			val[a] = totalDist;
		}
#else

		// compute the distance in parallel
		// p threads work on one proposed vectorID
		uint p = threadIdx.x % _lineParts;
		uint lane = threadIdx.x / _lineParts;

		// fetch next a;
		if (p == 0) {
			laneA[lane] = atomicInc(nProcessed, 100000);
			if (laneA[lane] < nVec)
				idx[laneA[lane]] = _inIdx[iter * _maxVecIn + laneA[lane]];
		}

//		__syncthreads();

		float ddd = 0.;
		if (laneA[lane] < nVec)
			do {

				float l = _lineLambda[idx[laneA[lane]] * _lineParts + p];
				lineDescr& line(*((lineDescr*) &l));

				uint l1 = line.p1;
				uint l2 = line.p2;
				float lambda = toFloat(line.lambda);

				float c2 = _cbDist[l2 * _nClusters * _lineParts
						+ l1 * _lineParts + p];

				ddd = dist(queryDist[p * _nClusters + l1],
						queryDist[p * _nClusters + l2], c2, lambda);

				ddd = warpReduceSum(ddd, _lineParts);

				// store result
				if (p == 0) {

					val[laneA[lane]] = ddd;
//
//					if (iter == 0)
//							printf("%d %f \n", laneA[lane], ddd );

					laneA[lane] = atomicInc(nProcessed, 100000);
					if (laneA[lane] < nVec)
						idx[laneA[lane]] =
								_inIdx[iter * _maxVecIn + laneA[lane]];

				}
			} while (laneA[lane] < nVec);
#endif
//		__syncthreads();

		// sort the results
		for (int i = threadIdx.x; i < _maxVec; i += blockDim.x)
			if (i >= nVec)
				val[i] = 10000000.;

		__syncthreads();

		if (_maxVec <= blockDim.x)
			bitonic3(val, idx, _maxVec);
		else
			bitonicLarge(val, idx, _maxVec);

		for (uint i = threadIdx.x; i < _k; i += blockDim.x)
			if (i < _k) {
				_bestDist[iter * _k + i] = val[i];
				_bestIdx[iter * _k + i] = idx[i];
			}

		__syncthreads();

	}
}

__global__ void rerankBIGKernelFast(float* _bestDist, uint* _bestIdx,
		const uint* _inIdx, const uint* _nVec, uint _maxVecIn,
		const float* _cbDist, uint _nClusters, uint _lineParts,
		const float* _lineLambda, const float* _queryL1, uint _QN, uint _dim,
		uint _maxBins, uint _k, uint _maxVec) {

	extern __shared__ float shm[];

	float* shmIter = shm;

	float* queryDist = shmIter;
	shmIter += _lineParts * _nClusters;

	uint* idx = (uint*) shmIter;
	shmIter += _maxVec;
	float* val = shmIter;
	shmIter += _maxVec;

	uint* laneA = (uint*) shmIter;
	shmIter += blockDim.x / _lineParts;

//	volatile float* d = shmIter;
//	shmIter += blockDim.x;

	uint &nVec(*(uint*) shmIter);
	shmIter++;

	uint* nProcessed = (uint*) shmIter;
	shmIter++;

	for (uint iter = blockIdx.x; iter < _QN; iter += gridDim.x) {

		__syncthreads();

		// load queryDistance
		for (int p = 0; p < _lineParts; p++) {
			if (threadIdx.x < _nClusters)
				queryDist[p * _nClusters + threadIdx.x] =
						_queryL1[iter * _lineParts * _nClusters + p * _nClusters
								+ threadIdx.x];

		}

		if (threadIdx.x == 0) {
			nVec = _nVec[iter];
			nVec = (nVec < _maxVec) ? nVec : _maxVec;

			*nProcessed = 0;

		}
		__syncthreads();

#if 0

		// compute the distances to all line approximations
		for (int a = threadIdx.x; a < nVec; a += blockDim.x) {
			idx[a] = _inIdx[iter * _maxVecIn + a];

			float totalDist = 0.;

			for (uint p = 0; p < _lineParts; p++) {

				float l = _lineLambda[idx[a] * +_lineParts + p];
				lineDescr& line( *( (lineDescr*)&l));

				uint l1 = line.p1;
				uint l2 = line.p2;
				float lambda = toFloat(line.lambda);

				float c2 = _cbDist[l2 * _nClusters * _lineParts
				+ l1 * _lineParts + p];

				float d = dist(queryDist[p * _nClusters + l1],
						queryDist[p * _nClusters + l2], c2, lambda);

				totalDist += d;
//
//				if (!isTriangle(queryDist[p * _nClusters + l1],
//								queryDist[p * _nClusters + l2], c2))
//				printf("non-triangle: l1/l2 %d %d === %f %f %f = %f %f \n",
//						l1, l2, queryDist[p * _nClusters + l1],
//						queryDist[p * _nClusters + l2], c2, d, lambda);
			}

//			if (iter == 0)
//			printf("%d %f \n", a, totalDist);
			val[a] = totalDist;
		}
#else

		// compute the distance in parallel
		// p threads work on one proposed vectorID
		uint p = threadIdx.x % _lineParts;
		uint lane = threadIdx.x / _lineParts;

		// fetch next a;
		if (p == 0) {
			laneA[lane] = atomicInc(nProcessed, 100000);
			if (laneA[lane] < nVec)
				idx[laneA[lane]] = _inIdx[iter * _maxVecIn + laneA[lane]];
		}

//		__syncthreads();

#if 1
		float ddd = 0.;
		if (laneA[lane] < nVec)
			do {

				size_t offset = idx[laneA[lane]];
				offset *= _lineParts;
				offset += p;

//				float l = _lineLambda[idx[laneA[lane]] * _lineParts + p];
				float l = _lineLambda[offset];

				lineDescr& line(*((lineDescr*) &l));

				uint l1 = line.p1;
				uint l2 = line.p2;
				float lambda = toFloat(line.lambda);

//				if (threadIdx.x == 0) {
//				if ((p ==0) && (iter == 0)) {
//					printf("lam%f l1%d l2%d \n", lambda, l1, l2);
//				}

				float c2 = _cbDist[l2 * _nClusters * _lineParts
						+ l1 * _lineParts + p];

				ddd = dist(queryDist[p * _nClusters + l1],
						queryDist[p * _nClusters + l2], c2, lambda);

				ddd = warpReduceSum(ddd, _lineParts);

				// store result
				if (p == 0) {

					val[laneA[lane]] = ddd;

//					if (iter == 0)
//							printf("%d %f \n", laneA[lane], ddd );

					laneA[lane] = atomicInc(nProcessed, 100000);
					if (laneA[lane] < nVec)
						idx[laneA[lane]] =
								_inIdx[iter * _maxVecIn + laneA[lane]];

				}
			} while (laneA[lane] < nVec);
#endif

#endif
//		__syncthreads();

		// sort the results
		for (int i = threadIdx.x; i < _maxVec; i += blockDim.x)
			if (i >= nVec)
				val[i] = 10000000.;

		__syncthreads();

		if (_maxVec <= blockDim.x)
			bitonic3(val, idx, _maxVec);
		else
			bitonicLarge(val, idx, _maxVec);

		for (uint i = threadIdx.x; i < _k; i += blockDim.x)
			if (i < _k) {
				_bestDist[iter * _k + i] = val[i];
				_bestIdx[iter * _k + i] = idx[i];
			}

		__syncthreads();

	}
}

/** assumes vectors in pinned memory, only outputs the best result */
__global__ void rerankBIGKernelPerfect(float* _bestDist, uint* _bestIdx,
		const uint* _inIdx, const uint* _nVec, uint _maxVecIn, const float* _Q, uint _QN, const float* _dbVec,
		uint _dim, uint _maxBins, uint _k, uint _maxVec) {

	extern __shared__ float shm[];

	float* shmIter = shm;

	uint8_t* vec = (uint8_t*) shmIter;
	float* vecf = (float*) vec;
	shmIter += _dim / 4;

	float* dist = shmIter;
	shmIter += blockDim.x;

	uint &nVec(*(uint*) shmIter);
	shmIter++;



	for (uint iter = blockIdx.x; iter < _QN; iter += gridDim.x) {

		float minD = 99999999999.;
		uint minIdx = 0;


		__syncthreads();

		// load query vector
		float v = _Q[iter * _dim + threadIdx.x];

		if (threadIdx.x == 0) {
			nVec = _nVec[iter];
			nVec = (nVec < _maxVec) ? nVec : _maxVec;
		}
		__syncthreads();



		for (int i = 0; i < nVec; i++) {
			__syncthreads();

			// load from CPU to shm (assumes uint8_t[_dim])
			if (threadIdx.x < _dim / 4)
				vecf[threadIdx.x] = _dbVec[_inIdx[iter * _maxVecIn + i]
						* _dim / 4 + threadIdx.x];

			__syncthreads();

			// compute distance
			float d = v - (float) vec[threadIdx.x];
			dist[threadIdx.x] = d * d;

			// reduction
			for (int stride = _dim >> 1; stride > 0; stride >>= 1) {
				__syncthreads();
				if (threadIdx.x < stride)
					dist[threadIdx.x] += dist[threadIdx.x + stride];
			}

			__syncthreads();


			if (threadIdx.x == 0) {

				if ((blockIdx.x == 0) && (iter == 0)) {
					printf( "%d: processing %d ( %d %d %d ) = %f\n", i, _inIdx[iter* _maxVecIn +i], vec[0], vec[1], vec[2], dist[0]);
				}
				if (dist[0] < minD) {
					minD = dist[0];
					minIdx = _inIdx[iter * _maxVecIn + i];
				}
			}
		}

		__syncthreads();


		// store single output
		if (threadIdx.x == 0) {
			printf("%d: %d %f \n", (iter+9000), minIdx, minD );
			_bestDist[iter * _k] = minD;
			_bestIdx[iter * _k] = minIdx;
		}
		__syncthreads();

	}
}


__global__ void rerankKernelDirectFast(float* _bestDist, uint* _bestIdx,
		const uint* _inIdx, const uint* _nVec, uint _maxVecIn,
		const float* _cbDist, uint _nClusters, uint _lineParts,
		const float* _lineLambda, const float* _queryL1, uint _QN, uint _dim,
		uint _maxBins, uint _k, uint _maxVec) {

	extern __shared__ float shm[];

	float* shmIter = shm;

	float* queryDist = shmIter;
	shmIter += _lineParts * _nClusters;

	uint* idx = (uint*) shmIter;
	shmIter += _maxVec;
	float* val = shmIter;
	shmIter += _maxVec;

//	uint* laneA = (uint*) shmIter;
	shmIter += blockDim.x / _lineParts;

//	volatile float* d = shmIter;
//	shmIter += blockDim.x;

	uint &nVec(*(uint*) shmIter);
	shmIter++;

	uint* nProcessed = (uint*) shmIter;
	shmIter++;

	for (uint iter = blockIdx.x; iter < _QN; iter += gridDim.x) {

		__syncthreads();

		// load queryDistance
		for (int p = 0; p < _lineParts; p++) {
			if (threadIdx.x < _nClusters)
				queryDist[p * _nClusters + threadIdx.x] =
						_queryL1[iter * _lineParts * _nClusters + p * _nClusters
								+ threadIdx.x];

		}

		if (threadIdx.x == 0) {
			nVec = _nVec[iter];
			nVec = (nVec < _maxVec) ? nVec : _maxVec;

			*nProcessed = 0;

		}
		__syncthreads();

		// compute the distance in parallel
		// p threads work on one proposed vectorID
		uint p = threadIdx.x % _lineParts;
		uint lane = threadIdx.x / _lineParts;

#if 0
		// fetch next a;
		if (p == 0) {
			laneA[lane] = atomicInc(nProcessed, 100000);
			if (laneA[lane] < nVec)
			idx[laneA[lane]] = _inIdx[iter * _maxVecIn + laneA[lane]];
		}

//		__syncthreads();

		float ddd = 0.;
		if (laneA[lane] < nVec)

		do {

//				uint ii = iter * _maxVecIn + laneA[lane];
//				float l = _lineLambda[ii * _lineParts + p];
//				lineDescr& line(*((lineDescr*) &l));
//
//				uint l1 = line.p1;
//				uint l2 = line.p2;
//				float lambda = toFloat(line.lambda);
//
//				float c2 = _cbDist[l2 * _nClusters * _lineParts
//						+ l1 * _lineParts + p];
//
//				ddd = dist(queryDist[p * _nClusters + l1],
//						queryDist[p * _nClusters + l2], c2, lambda);
//
//				ddd = warpReduceSum(ddd, _lineParts);

			// store result
			if (p == 0) {

				if (laneA[lane] == 3494) {
					printf("3494: %d \n", idx[laneA[lane]]);
				}

				val[laneA[lane]] = ddd;
//
//					if (iter == 0)
//							printf("%d %f \n", laneA[lane], ddd );

				laneA[lane] = atomicInc(nProcessed, 100000);
				if (laneA[lane] < nVec)
				idx[laneA[lane]] =
				_inIdx[iter * _maxVecIn + laneA[lane]];

			}
		}while (laneA[lane] < nVec);

#endif
		__syncthreads();
		uint la;
//		float ddd = 0.;
		for (la = lane; la < nVec; la += blockDim.x / _lineParts) {
			if (p == 0) {

//				printf("lane: %d %d %d \n", lane, la, nVec);
//				if ((la == 3494)) {
//					printf("3494 in: %d %d \n", iter, _inIdx[ iter * _maxVecIn + la]);
//				}

//				val[la] = ddd;
				//
				//					if (iter == 0)
				//							printf("%d %f \n", laneA[lane], ddd );

				idx[la] = _inIdx[iter * _maxVecIn + la];

			}

		}

		__syncthreads();

		// sort the results
//		for (int i = threadIdx.x; i < _maxVec; i += blockDim.x)
//			if (i >= nVec)
//				val[i] = 10000000.;

		__syncthreads();

//		if (_maxVec <= blockDim.x)
//			bitonic3(val, idx, _maxVec);
//		else
//			bitonicLarge(val, idx, _maxVec);

		for (uint i = threadIdx.x; i < nVec; i += blockDim.x) {
			_bestDist[iter * _k + i] = val[i];
			_bestIdx[iter * _k + i] = idx[i];

			if ((iter * _k + i) == 3494) {
				printf("3494 out: %d \n", idx[i]);
			}

		}

		__syncthreads();

	}
}

__global__ void rerankKernelFastLoop(float* _bestDist, uint* _bestIdx,
		const uint* _inIdx, const uint* _nVec, uint _maxVecIn,
		const float* _cbDist, uint _nClusters, uint _lineParts,
		const float* _lineLambda, const float* _queryL1, uint _QN, uint _dim,
		uint _maxBins, uint _k, uint _maxVec) {

	extern __shared__ float shm[];

	float* shmIter = shm;

	float* queryDist = shmIter;
	shmIter += _lineParts * _nClusters;

	uint* idx = (uint*) shmIter;
	shmIter += 4096;
	float* val = shmIter;
	shmIter += 4096;

	uint* laneA = (uint*) shmIter;
	shmIter += blockDim.x / _lineParts;

//	volatile float* d = shmIter;
//	shmIter += blockDim.x;

	uint &nVec(*(uint*) shmIter);
	shmIter++;

	uint* nProcessed = (uint*) shmIter;
	shmIter++;

	for (uint iter = blockIdx.x; iter < _QN; iter += gridDim.x) {

		__syncthreads();

		// load queryDistance
		for (int p = 0; p < _lineParts; p++) {
			if (threadIdx.x < _nClusters)
				queryDist[p * _nClusters + threadIdx.x] =
						_queryL1[iter * _lineParts * _nClusters + p * _nClusters
								+ threadIdx.x];

		}

		for (uint vIter = 0; vIter < _maxVec / 4096; vIter++) {
			__syncthreads();

			uint vOffs = vIter * 4096;

			if (threadIdx.x == 0) {
				nVec = _nVec[iter] - vOffs;
				nVec = (nVec < _maxVec) ? nVec : _maxVec;

				nVec = (nVec < 4096) ? nVec : 4096;

				*nProcessed = 0;

			}
			__syncthreads();

#if 0

			// compute the distances to all line approximations
			for (int a = threadIdx.x; a < nVec; a += blockDim.x) {
				idx[a] = _inIdx[iter * _maxVecIn + a + vOffs];

				float totalDist = 0.;

				for (uint p = 0; p < _lineParts; p++) {

					float l = _lineLambda[idx[a] * +_lineParts + p];
					lineDescr& line( *( (lineDescr*)&l));

					uint l1 = line.p1;
					uint l2 = line.p2;
					float lambda = toFloat(line.lambda);

					float c2 = _cbDist[l2 * _nClusters * _lineParts
					+ l1 * _lineParts + p];

					float d = dist(queryDist[p * _nClusters + l1],
							queryDist[p * _nClusters + l2], c2, lambda);

					totalDist += d;
//
//				if (!isTriangle(queryDist[p * _nClusters + l1],
//								queryDist[p * _nClusters + l2], c2))
//				printf("non-triangle: l1/l2 %d %d === %f %f %f = %f %f \n",
//						l1, l2, queryDist[p * _nClusters + l1],
//						queryDist[p * _nClusters + l2], c2, d, lambda);
				}

//			if (iter == 0)
//			printf("%d %f \n", a, totalDist);
				val[a] = totalDist;
			}
#else

			// compute the distance in parallel
			// p threads work on one proposed vectorID
			uint p = threadIdx.x % _lineParts;
			uint lane = threadIdx.x / _lineParts;

			// fetch next a;
			if (p == 0) {
				laneA[lane] = atomicInc(nProcessed, 100000);
				if (laneA[lane] < nVec)
					idx[laneA[lane]] = _inIdx[iter * _maxVecIn + laneA[lane]
							+ vOffs];
			}

//		__syncthreads();

			float ddd = 0.;
			if (laneA[lane] < nVec)
				do {

					float l = _lineLambda[idx[laneA[lane]] * _lineParts + p];
					lineDescr& line(*((lineDescr*) &l));

					uint l1 = line.p1;
					uint l2 = line.p2;
					float lambda = toFloat(line.lambda);

					float c2 = _cbDist[l2 * _nClusters * _lineParts
							+ l1 * _lineParts + p];

					ddd = dist(queryDist[p * _nClusters + l1],
							queryDist[p * _nClusters + l2], c2, lambda);

					ddd = warpReduceSum(ddd, _lineParts);

					// store result
					if (p == 0) {

						val[laneA[lane]] = ddd;
//
//					if (iter == 0)
//							printf("%d %f \n", laneA[lane], ddd );

						laneA[lane] = atomicInc(nProcessed, 100000);
						if (laneA[lane] < nVec)
							idx[laneA[lane]] = _inIdx[iter * _maxVecIn
									+ laneA[lane]];

					}
				} while (laneA[lane] < nVec);
#endif
//		__syncthreads();

			// sort the results
			for (int i = threadIdx.x; i < 4096; i += blockDim.x)
				if (i >= nVec)
					val[i] = 10000000.;

			__syncthreads();

//			bitonicLarge(val, idx, 4096);

			for (uint i = threadIdx.x; i < 4096; i += blockDim.x) {
				_bestDist[iter * _k + i + vOffs] = val[i];
				_bestIdx[iter * _k + i + vOffs] = idx[i];
			}

			__syncthreads();

		}
	}
}

#if 1
__global__ void rerankPlusVecKernelFast(float* _bestDist, uint* _bestIdx,
		const uint* _dbIdx, const uint* _binPrefix, const uint* _binCounts,
		const uint* _assignedBins, const uint* _assignedNBins, uint _maxBins,
		uint _maxNVecPerBin,

		const float* _cbDist, uint _nClusters, uint _lineParts,
		const float* _lineLambda, const float* _queryL1, uint _QN, uint _dim,
		uint _k, uint _maxVec) {

	extern __shared__ float shm[];

	float* shmIter = shm;

	float* queryDist = shmIter;
	shmIter += _lineParts * _nClusters;

	uint* idx = (uint*) shmIter;
	shmIter += _maxVec;
	float* val = shmIter;
	shmIter += _maxVec;

//	uint* laneA = (uint*) shmIter;
	shmIter += blockDim.x / _lineParts;

//	volatile float* d = shmIter;
//	shmIter += blockDim.x;

	uint &nVec(*(uint*) shmIter);
	shmIter++;

	uint &nBins(*(uint*) shmIter);
	shmIter++;

	uint &currentBin(*(uint*) shmIter);
	shmIter++;

	uint &totalVec(*(uint*) shmIter);
	shmIter++;

	uint* nProcessed = (uint*) shmIter;
	shmIter++;

	for (uint iter = blockIdx.x; iter < _QN; iter += gridDim.x) {

		__syncthreads();

		// load queryDistance
		for (int p = 0; p < _lineParts; p++) {
			if (threadIdx.x < _nClusters)
				queryDist[p * _nClusters + threadIdx.x] =
						_queryL1[iter * _lineParts * _nClusters + p * _nClusters
								+ threadIdx.x];

		}

		if (threadIdx.x == 0) {

			nBins = _assignedNBins[iter];

			totalVec = 0;
		}
		__syncthreads();

		// loop over the best assigned bins
		for (int bin = 0; (bin < nBins) && (totalVec < _maxVec); bin++) {

			if (threadIdx.x == 0) {
				currentBin = _assignedBins[iter * _maxBins + bin];

				nVec = _binCounts[currentBin];

				if ((totalVec + nVec) >= _maxVec)
					nVec = _maxVec - totalVec;

				*nProcessed = 0;

			}
			__syncthreads();

			if (nVec == 0)
				continue;

			uint* inIdx = idx + totalVec;

			for (int v = threadIdx.x; v < nVec; v += blockDim.x)
				inIdx[v] = _dbIdx[_binPrefix[currentBin] + v];

			__syncthreads();

#if 1

			// compute the distances to all line approximations
			for (int a = threadIdx.x; a < nVec; a += blockDim.x) {

				float totalDist = 0.;

				for (uint p = 0; p < _lineParts; p++) {

					float l = _lineLambda[inIdx[a] * +_lineParts + p];
					lineDescr& line(*((lineDescr*) &l));

					uint l1 = line.p1;
					uint l2 = line.p2;
					float lambda = toFloat(line.lambda);

					float c2 = _cbDist[l2 * _nClusters * _lineParts
							+ l1 * _lineParts + p];

					float d = dist(queryDist[p * _nClusters + l1],
							queryDist[p * _nClusters + l2], c2, lambda);

					totalDist += d;
				}

				val[totalVec + a] = totalDist;
			}
#else

			// compute the distance in parallel
			// p threads work on one proposed vectorID
			uint p = threadIdx.x % _lineParts;
			uint lane = threadIdx.x / _lineParts;

			// fetch next a;
			if (p == 0) {
				laneA[lane] = atomicInc(nProcessed, 100000);
			}

//		__syncthreads();

			float ddd = 0.;
			if (laneA[lane] < nVec)
			do {

				float l = _lineLambda[inIdx[laneA[lane]] * _lineParts + p];
				lineDescr& line(*((lineDescr*) &l));

				uint l1 = line.p1;
				uint l2 = line.p2;
				float lambda = toFloat(line.lambda);

				float c2 = _cbDist[l2 * _nClusters * _lineParts
				+ l1 * _lineParts + p];

				ddd = dist(queryDist[p * _nClusters + l1],
						queryDist[p * _nClusters + l2], c2, lambda);

				ddd = warpReduceSum(ddd, _lineParts);

				// store result
				if (p == 0) {

					val[totalVec + laneA[lane]] = ddd;
//
//					if (iter == 0)
//							printf("%d %f \n", laneA[lane], ddd );

					laneA[lane] = atomicInc(nProcessed, 100000);
					if (laneA[lane] < nVec)
					idx[laneA[lane]] =
					inIdx[laneA[lane]];

				}
			}while (laneA[lane] < nVec);
#endif
			__syncthreads();

			if (threadIdx.x == 0)
				totalVec += nVec;

		}

		if (threadIdx.x == 0)
			nVec = totalVec;

		__syncthreads();

		// sort the results
		for (int i = threadIdx.x; i < _maxVec; i += blockDim.x)
			if (i >= nVec)
				val[i] = 10000000.;

		__syncthreads();

		if (_maxVec <= blockDim.x)
			bitonic3(val, idx, _maxVec);
		else
			bitonicLarge(val, idx, _maxVec);

		for (uint i = threadIdx.x; i < _k; i += blockDim.x)
			if (i < _k) {
				_bestDist[iter * _k + i] = val[i];
				_bestIdx[iter * _k + i] = idx[i];
			}

		__syncthreads();

	}
}

#endif

#if 1
void PerturbationProTree::rerankKBestVectors(float *_bestDist, uint *_bestIdx,
		const float* _queryL1, const uint *_bins, const uint *_nBins,
		uint _maxBins, const float* _Q, uint _QN, uint _k) {

	uint nnn = log2(_k);

//	nnn = 1024;

	std::cout << "nnn: " << nnn << std::endl;

	uint maxVecConsider = nnn;
	uint maxVecOut = nnn;
	uint nThreads = (maxVecConsider < 1024) ? maxVecConsider : 1024;
	dim3 block(nThreads, 1, 1);
	dim3 grid((_QN < 1024) ? _QN : 1024, 1, 1);

	uint hash = 2048;

	uint shmSize = (maxVecConsider
			+ ((hash > maxVecConsider) ? hash : maxVecConsider) + 1 + 10)
			* sizeof(float);

	uint *selectIdx;// array for storing nn vector IDs (size maxVecOut * _QN)
	uint *nVec;

	cudaMalloc(&selectIdx, _QN * _k * sizeof(uint));
	cudaMalloc(&nVec, _QN * sizeof(uint));

	if ((!selectIdx) || (!nVec)) {
		std::cout << "getKBestVectors: did not get memory !" << std::endl;

		exit(1);
	}

	cudaMemset(selectIdx, 0, _QN * _k * sizeof(uint));

#if 0
	if (_k < 1024)
	getKVectorIDsKernel<<<grid, block, shmSize>>>(selectIdx, nVec, d_dbIdx,
			d_binPrefix, d_binCounts, d_nBins, _Q, _bins, _nBins, _QN,
			d_dim, _maxBins, _k, maxVecConsider, maxVecOut, d_nDBs, 280);
	else
	getKVectorIDsKernelLarge<<<grid, block, shmSize>>>(selectIdx, nVec,
			d_dbIdx, d_binPrefix, d_binCounts, d_nBins, _Q, _bins, _nBins,
			_QN, d_dim, _maxBins, _k, maxVecConsider, maxVecOut, d_nDBs,
			280);

#else
//	shmSize = (3* nThreads + nThreads / 32 + 5) * sizeof(float);
	shmSize = (3 * nThreads + 4) * sizeof(float);

	getKVectorIDsKernelFast<<<grid, block, shmSize>>>(selectIdx, nVec, d_dbIdx,
			d_binPrefix, d_binCounts, d_nBins, _Q, _bins, _nBins, _QN, d_dim,
			_maxBins, _k, maxVecConsider, maxVecOut, 2800);

#endif
	checkCudaErrors(cudaDeviceSynchronize());

//	outputVecUint("selectIdx", selectIdx, 1024);

	outputVecUint("nVec", nVec, 100);

	std::cout << "multi Vector IDs done" << std::endl;
////	_QN = 1;
//
//	outputVecUint("selectIdx", selectIdx, 100);

//	uint maxVec = 2 * log2(_k);
	uint maxVec = log2(_k);

	nThreads = (maxVec > d_dim) ? maxVec : d_dim;

	nThreads = (nThreads < 1024) ? nThreads : 1024;

	block = dim3(nThreads, 1, 1);

//	grid = dim3(1, 1, 1);

//	shmSize = (d_dim + d_nClusters * d_lineParts + 2 * maxVec + 2)
//			* sizeof(float);

//	rerankKernel<<<grid, block, shmSize>>>(_bestDist, _bestIdx, selectIdx, nVec,
//			maxVecOut, d_codeBookDistL1L2, d_nClusters, d_lineParts,
//			d_lineLambda, d_lineP1, d_lineP2, _queryL1, _QN, d_dim, _maxBins,
//			_k, maxVec);

	shmSize = (d_dim + d_nClusters * d_lineParts + 2 * maxVec + nThreads
			+ nThreads / d_lineParts + 2) * sizeof(float);

	if (maxVec <= 4096) {
		shmSize = (d_nClusters * d_lineParts + 2 * maxVec
				+ nThreads / d_lineParts + 2) * sizeof(float);

		std::cout << "rerank: maxVec: " << maxVec << " shm: " << shmSize << std::endl;

		rerankKernelFast<<<grid, block, shmSize>>>(_bestDist, _bestIdx,
				selectIdx, nVec, maxVecOut, d_codeBookDistL1L2, d_nClusters,
				d_lineParts, d_lineLambda, _queryL1, _QN, d_dim, _maxBins, _k,
				maxVec);
	} else {

		shmSize = (d_nClusters * d_lineParts + 2 * 4096 + nThreads / d_lineParts
				+ 2) * sizeof(float);

		std::cout << "rerank: maxVec: " << maxVec << " k: " << _k << " shm: "
				<< shmSize << std::endl;

		rerankKernelFastLoop<<<grid, block, shmSize>>>(_bestDist, _bestIdx,
				selectIdx, nVec, maxVecOut, d_codeBookDistL1L2, d_nClusters,
				d_lineParts, d_lineLambda, _queryL1, _QN, d_dim, _maxBins, _k,
				maxVec);
//
//		checkCudaErrors(cudaDeviceSynchronize());

		thrust::device_ptr<float> bdt(_bestDist);
		thrust::device_ptr<uint> bit(_bestIdx);

		for (int i = 0; i < _QN; i++)
			thrust::sort_by_key(bdt + i * _k, bdt + i * _k + _k, bit + i * _k);

		std::cout << "done sort by key" << std::endl;

//		shmSize = (6 * 1024 + 2) * sizeof(float);
//
//		std::cout << "mergeKernel: shm: " << shmSize << std::endl;
//
//		mergeKernel<<<grid, block, shmSize>>>(_bestDist, _bestIdx, _QN, _k);
	}

	checkCudaErrors(cudaDeviceSynchronize());

	std::cout << "rerankKBestVectors done " << std::endl;

	outputVec("BestDist", _bestDist, 1000);

	cudaFree(nVec);
	cudaFree(selectIdx);

}
#else
void PerturbationProTree::rerankKBestVectors(float *_bestDist, uint *_bestIdx,
		const float* _queryL1, const uint *_bins, const uint *_nBins,
		uint _maxBins, const float* _Q, uint _QN, uint _k) {

	uint maxVec = log2(_k);

	uint nThreads = (maxVec > d_dim) ? maxVec : d_dim;

	nThreads = (nThreads < 1024) ? nThreads : 1024;

	dim3 block = dim3(nThreads, 1, 1);
	dim3 grid((_QN < 1024) ? _QN : 1024, 1, 1);

	uint shmSize = (d_dim + d_nClusters * d_lineParts + 2 * maxVec
			+ nThreads / d_lineParts + 10) * sizeof(float);
	std::cout << "maxVec: " << maxVec << " shm: " << shmSize << std::endl;

//	getKVectorIDsKernel<<<grid, block, shmSize>>>(selectIdx, nVec, d_dbIdx,
//			d_binPrefix, d_binCounts, d_nBins, _Q, _bins, _nBins, _QN, d_dim,
//			_maxBins, _k, maxVecConsider, maxVecOut, d_nDBs, 280);

	rerankPlusVecKernelFast<<<grid, block, shmSize>>>(_bestDist, _bestIdx,
			d_dbIdx, d_binPrefix, d_binCounts, _bins, _nBins, _maxBins, 280,
			d_codeBookDistL1L2, d_nClusters, d_lineParts, d_lineLambda,
			_queryL1, _QN, d_dim, _k, maxVec);

	checkCudaErrors(cudaDeviceSynchronize());

	std::cout << "rerankKBestVectors done " << std::endl;

	outputVec("BestDist", _bestDist, 1000);

}
#endif

void PerturbationProTree::rerankBIGKBestVectors(float *_bestDist,
		uint *_bestIdx, const float* _queryL1, const uint *_bins,
		const uint *_nBins, uint _maxBins, const float* _Q, uint _QN, uint _k) {

	uint nnn = log2(_k);
//	nnn = 1024;

	std::cout << "nnn: " << nnn << std::endl;

	uint maxVecConsider = nnn;
	uint maxVecOut = nnn;
	uint nThreads = (maxVecConsider < 1024) ? maxVecConsider : 1024;
	dim3 block(nThreads, 1, 1);
	dim3 grid((_QN < 1024) ? _QN : 1024, 1, 1);

	uint hash = 2048;

	uint shmSize = (maxVecConsider
			+ ((hash > maxVecConsider) ? hash : maxVecConsider) + 1 + 10)
			* sizeof(float);

	uint *selectIdx;// array for storing nn vector IDs (size maxVecOut * _QN)
	uint *nVec;

	cudaMalloc(&selectIdx, _QN * _k * sizeof(uint));
	cudaMalloc(&nVec, _QN * sizeof(uint));

	if ((!selectIdx) || (!nVec)) {
		std::cout << "getKBestVectors: did not get memory !" << std::endl;

		exit(1);
	}

	cudaMemset(selectIdx, 0, _QN * _k * sizeof(uint));

#if 0
	if (_k < 1024)
	getKVectorIDsKernel<<<grid, block, shmSize>>>(selectIdx, nVec, d_dbIdx,
			d_binPrefix, d_binCounts, d_nBins, _Q, _bins, _nBins, _QN,
			d_dim, _maxBins, _k, maxVecConsider, maxVecOut, d_nDBs, 280);
	else
	getKVectorIDsKernelLarge<<<grid, block, shmSize>>>(selectIdx, nVec,
			d_dbIdx, d_binPrefix, d_binCounts, d_nBins, _Q, _bins, _nBins,
			_QN, d_dim, _maxBins, _k, maxVecConsider, maxVecOut, d_nDBs,
			280);

#else
//	shmSize = (3* nThreads + nThreads / 32 + 5) * sizeof(float);
	shmSize = (3 * nThreads + 4) * sizeof(float);

	getKVectorIDsKernelFast<<<grid, block, shmSize>>>(selectIdx, nVec, d_dbIdx,
			d_binPrefix, d_binCounts, d_nBins, _Q, _bins, _nBins, _QN, d_dim,
			_maxBins, _k, maxVecConsider, maxVecOut, nnn);

//	getKVectorIDsKernelFast<<<grid, block, shmSize>>>(selectIdx, nVec, d_dbIdx,
//			d_binPrefix, d_binCounts, d_nBins, _Q, _bins, _nBins, _QN, d_dim,
//			_maxBins, _k, maxVecConsider, maxVecOut, d_nDBs, nnn);

//	getKVectorIDsKernelFast<<<grid, block, shmSize>>>(selectIdx, nVec, d_dbIdx,
//				d_binPrefix, d_binCounts, d_nBins, _Q, _bins, _nBins, _QN, d_dim,
//				_maxBins, _k, maxVecConsider, maxVecOut, d_nDBs, 4096);

//	getKVectorIDsKernelFast<<<grid, block, shmSize>>>(selectIdx, nVec, d_dbIdx,
//			d_binPrefix, d_binCounts, d_nBins, _Q, _bins, _nBins, _QN, d_dim,
//			_maxBins, _k, maxVecConsider, maxVecOut, d_nDBs, 280);

	// 280 for most results
#endif
	checkCudaErrors(cudaDeviceSynchronize());

//	outputVecUint("selectIdx", selectIdx, 1024);
//
//	outputVecUint("nVec", nVec, 100);

	std::cout << "multi Vector IDs done" << std::endl;
////	_QN = 1;
//
//	outputVecUint("selectIdx", selectIdx, 100);

//	uint maxVec = 2 * log2(_k);
	uint maxVec = log2(_k);

	nThreads = (maxVec > d_dim) ? maxVec : d_dim;

	nThreads = (nThreads < 1024) ? nThreads : 1024;

	block = dim3(nThreads, 1, 1);

//	grid = dim3(1, 1, 1);

//	shmSize = (d_dim + d_nClusters * d_lineParts + 2 * maxVec + 2)
//			* sizeof(float);

//	rerankKernel<<<grid, block, shmSize>>>(_bestDist, _bestIdx, selectIdx, nVec,
//			maxVecOut, d_codeBookDistL1L2, d_nClusters, d_lineParts,
//			d_lineLambda, d_lineP1, d_lineP2, _queryL1, _QN, d_dim, _maxBins,
//			_k, maxVec);

	cudaMemcpy(_bestIdx, selectIdx, maxVec * _QN * sizeof(uint),
			cudaMemcpyDeviceToDevice);

	shmSize = (d_dim + d_nClusters * d_lineParts + 2 * maxVec + nThreads
			+ nThreads / d_lineParts + 2) * sizeof(float);

#if 0
	if (maxVec <= 4096) {
		shmSize = (d_nClusters * d_lineParts + 2 * maxVec
				+ nThreads / d_lineParts + 2) * sizeof(float);

		std::cout << "rerank: maxVec: " << maxVec << " shm: " << shmSize << std::endl;

		rerankKernelFast<<<grid, block, shmSize>>>(_bestDist, _bestIdx,
				selectIdx, nVec, maxVecOut, d_codeBookDistL1L2, d_nClusters,
				d_lineParts, d_lineLambda, _queryL1, _QN, d_dim, _maxBins, _k,
				maxVec);
	} else {

		shmSize = (d_nClusters * d_lineParts + 2 * 4096 + nThreads / d_lineParts
				+ 2) * sizeof(float);

		std::cout << "rerank: maxVec: " << maxVec << " k: " << _k << " shm: "
		<< shmSize << std::endl;

		rerankKernelFastLoop<<<grid, block, shmSize>>>(_bestDist, _bestIdx,
				selectIdx, nVec, maxVecOut, d_codeBookDistL1L2, d_nClusters,
				d_lineParts, d_lineLambda, _queryL1, _QN, d_dim, _maxBins, _k,
				maxVec);
//
//		checkCudaErrors(cudaDeviceSynchronize());

		thrust::device_ptr<float> bdt( _bestDist );
		thrust::device_ptr<uint> bit( _bestIdx );

		for (int i =0; i < _QN; i++)
		thrust::sort_by_key( bdt + i * _k , bdt + i* _k + _k, bit + i * _k);

		std::cout << "done sort by key" << std::endl;

//		shmSize = (6 * 1024 + 2) * sizeof(float);
//
//		std::cout << "mergeKernel: shm: " << shmSize << std::endl;
//
//		mergeKernel<<<grid, block, shmSize>>>(_bestDist, _bestIdx, _QN, _k);
	}

	checkCudaErrors(cudaDeviceSynchronize());

	std::cout << "rerankKBestVectors done " << std::endl;
	outputVec("BestDist", _bestDist, 1000);
#endif

	cudaFree(nVec);
	cudaFree(selectIdx);

}

/** assume _hLines in pinned memory */
void PerturbationProTree::rerankBIGKBestVectors2(float *_bestDist,
		uint *_bestIdx, const float* _queryL1, const uint *_bins,
		const uint *_nBins, uint _maxBins, const float* _Q, uint _QN, uint _k,
		const float* _hLines) {

	uint nnn = log2(_k);
//	nnn = 1024;

	std::cout << "nnn: " << nnn << std::endl;

	uint maxVecConsider = nnn;
	uint maxVecOut = nnn;
	uint nThreads = (maxVecConsider < 1024) ? maxVecConsider : 1024;
	dim3 block(nThreads, 1, 1);
	dim3 grid((_QN < 1024) ? _QN : 1024, 1, 1);

	uint hash = 2048;

	uint shmSize = (maxVecConsider
			+ ((hash > maxVecConsider) ? hash : maxVecConsider) + 1 + 10)
			* sizeof(float);

	uint *selectIdx;// array for storing nn vector IDs (size maxVecOut * _QN)
	uint *nVec;

	cudaMalloc(&selectIdx, _QN * _k * sizeof(uint));
	cudaMalloc(&nVec, _QN * sizeof(uint));

	if ((!selectIdx) || (!nVec)) {
		std::cout << "getKBestVectors: did not get memory !" << std::endl;

		exit(1);
	}

	cudaMemset(selectIdx, 0, _QN * _k * sizeof(uint));

//	shmSize = (3* nThreads + nThreads / 32 + 5) * sizeof(float);
	shmSize = (3 * nThreads + 4) * sizeof(float);

	getKVectorIDsKernelFast<<<grid, block, shmSize>>>(selectIdx, nVec, d_dbIdx,
			d_binPrefix, d_binCounts, d_nBins, _Q, _bins, _nBins, _QN, d_dim,
			_maxBins, _k, maxVecConsider, maxVecOut, nnn);

	checkCudaErrors(cudaDeviceSynchronize());

	std::cout << "multi Vector IDs done" << std::endl;

//	rerankKernel<<<grid, block, shmSize>>>(_bestDist, _bestIdx, selectIdx, nVec,
//			maxVecOut, d_codeBookDistL1L2, d_nClusters, d_lineParts,
//			d_lineLambda, d_lineP1, d_lineP2, _queryL1, _QN, d_dim, _maxBins,
//			_k, maxVec);

	uint maxVec = log2(_k);

	nThreads = (maxVec > d_dim) ? maxVec : d_dim;

	nThreads = (nThreads < 1024) ? nThreads : 1024;

	block = dim3(nThreads, 1, 1);

//	cudaMemcpy(_bestIdx, selectIdx, maxVec * _QN * sizeof(uint),
//			cudaMemcpyDeviceToDevice);

	shmSize = (d_dim + d_nClusters * d_lineParts + 2 * maxVec + nThreads
			+ nThreads / d_lineParts + 2) * sizeof(float);

#if 1
	if (maxVec <= 4096) {
		shmSize = (d_nClusters * d_lineParts + 2 * maxVec
				+ nThreads / d_lineParts + 2) * sizeof(float);

		std::cout << "rerank: maxVec: " << maxVec << " shm: " << shmSize << std::endl;

		rerankBIGKernelFast<<<grid, block, shmSize>>>(_bestDist, _bestIdx,
				selectIdx, nVec, maxVecOut, d_codeBookDistL1L2, d_nClusters,
				d_lineParts, _hLines, _queryL1, _QN, d_dim, _maxBins, _k,
				maxVec);
	} else {

		shmSize = (d_nClusters * d_lineParts + 2 * 4096 + nThreads / d_lineParts
				+ 2) * sizeof(float);

		std::cout << "rerank: maxVec: " << maxVec << " k: " << _k << " shm: "
				<< shmSize << std::endl;

		rerankKernelFastLoop<<<grid, block, shmSize>>>(_bestDist, _bestIdx,
				selectIdx, nVec, maxVecOut, d_codeBookDistL1L2, d_nClusters,
				d_lineParts, d_lineLambda, _queryL1, _QN, d_dim, _maxBins, _k,
				maxVec);
//
//		checkCudaErrors(cudaDeviceSynchronize());

		thrust::device_ptr<float> bdt(_bestDist);
		thrust::device_ptr<uint> bit(_bestIdx);

		for (int i = 0; i < _QN; i++)
			thrust::sort_by_key(bdt + i * _k, bdt + i * _k + _k, bit + i * _k);

		std::cout << "done sort by key" << std::endl;

//		shmSize = (6 * 1024 + 2) * sizeof(float);
//
//		std::cout << "mergeKernel: shm: " << shmSize << std::endl;
//
//		mergeKernel<<<grid, block, shmSize>>>(_bestDist, _bestIdx, _QN, _k);
	}

	checkCudaErrors(cudaDeviceSynchronize());

	std::cout << "rerankKBestVectors done " << std::endl;
//	outputVec("BestDist", _bestDist, 1000);
#endif

	cudaFree(nVec);
	cudaFree(selectIdx);

}

/** assume _hLines in pinned memory */
void PerturbationProTree::rerankBIGKBestVectorsPerfect(float *_bestDist,
		uint *_bestIdx, const uint *_bins, const uint *_nBins, uint _maxBins,
		const float* _Q, uint _QN, uint _k, const float* _hLines) {

	uint nnn = log2(_k);
//	nnn = 1024;

	std::cout << "nnn: " << nnn << std::endl;

	uint maxVecConsider = nnn;
	uint maxVecOut = nnn;
	uint nThreads = (maxVecConsider < 1024) ? maxVecConsider : 1024;
	dim3 block(nThreads, 1, 1);
	dim3 grid((_QN < 1024) ? _QN : 1024, 1, 1);

	uint hash = 2048;

	uint shmSize = (maxVecConsider
			+ ((hash > maxVecConsider) ? hash : maxVecConsider) + 1 + 10)
			* sizeof(float);

	uint *selectIdx;// array for storing nn vector IDs (size maxVecOut * _QN)
	uint *nVec;

	cudaMalloc(&selectIdx, _QN * _k * sizeof(uint));
	cudaMalloc(&nVec, _QN * sizeof(uint));

	if ((!selectIdx) || (!nVec)) {
		std::cout << "getKBestVectors: did not get memory !" << std::endl;

		exit(1);
	}

	cudaMemset(selectIdx, 0, _QN * _k * sizeof(uint));

//	shmSize = (3* nThreads + nThreads / 32 + 5) * sizeof(float);
	shmSize = (3 * nThreads + 4) * sizeof(float);

	getKVectorIDsKernelFast<<<grid, block, shmSize>>>(selectIdx, nVec, d_dbIdx,
			d_binPrefix, d_binCounts, d_nBins, _Q, _bins, _nBins, _QN, d_dim,
			_maxBins, _k, maxVecConsider, maxVecOut, nnn);

	checkCudaErrors(cudaDeviceSynchronize());


		cudaMemcpy(_bestIdx, selectIdx, _k * _QN * sizeof(uint),
				cudaMemcpyDeviceToDevice);

	std::cout << "multi Vector IDs done" << std::endl;

	uint maxVec = log2(_k);

	nThreads = d_dim;

	block = dim3(nThreads, 1, 1);

	shmSize = (d_dim / 4 + d_dim + 1) * sizeof(float);

	std::cout << "rerankPerfect: maxVec: " << maxVec << " shm: " << shmSize << std::endl;

	rerankBIGKernelPerfect<<<grid, block, shmSize>>>(_bestDist, _bestIdx,
			selectIdx, nVec, maxVecOut, _Q, _QN, _hLines, d_dim, _maxBins, _k,
			maxVec);

	checkCudaErrors(cudaDeviceSynchronize());

	std::cout << "rerankKBestVectors done " << std::endl;
//	outputVec("BestDist", _bestDist, 1000);

	cudaFree(nVec);
	cudaFree(selectIdx);

}

/** with line reranking given CPU _hlines */
void PerturbationProTree::rerankBIGKBestVectors(vector<uint>& _resIdx,
		float *_bestDist, uint *_bestIdx, const float* _queryL1,
		const uint *_bins, const uint *_nBins, uint _maxBins, const float* _Q,
		uint _QN, uint _k, const float *_hLines) {

	uint nnn = log2(_k);

//	nnn = 1024;

	std::cout << "nnn: " << nnn << std::endl;

	uint maxVecConsider = nnn;
	uint maxVecOut = nnn;
	uint nThreads = (maxVecConsider < 1024) ? maxVecConsider : 1024;
	dim3 block(nThreads, 1, 1);
	dim3 grid((_QN < 1024) ? _QN : 1024, 1, 1);

	uint hash = 2048;

	uint shmSize = (maxVecConsider
			+ ((hash > maxVecConsider) ? hash : maxVecConsider) + 1 + 10)
			* sizeof(float);

	uint *selectIdx;// array for storing nn vector IDs (size maxVecOut * _QN)
	uint *nVec;

	cudaMalloc(&selectIdx, _QN * _k * sizeof(uint));
	cudaMalloc(&nVec, _QN * sizeof(uint));

	if ((!selectIdx) || (!nVec)) {
		std::cout << "getKBestVectors: did not get memory !" << std::endl;

		exit(1);
	}

	cudaMemset(selectIdx, 0, _QN * _k * sizeof(uint));

#if 0
	if (_k < 1024)
	getKVectorIDsKernel<<<grid, block, shmSize>>>(selectIdx, nVec, d_dbIdx,
			d_binPrefix, d_binCounts, d_nBins, _Q, _bins, _nBins, _QN,
			d_dim, _maxBins, _k, maxVecConsider, maxVecOut, d_nDBs, 280);
	else
	getKVectorIDsKernelLarge<<<grid, block, shmSize>>>(selectIdx, nVec,
			d_dbIdx, d_binPrefix, d_binCounts, d_nBins, _Q, _bins, _nBins,
			_QN, d_dim, _maxBins, _k, maxVecConsider, maxVecOut, d_nDBs,
			280);

#else
//	shmSize = (3* nThreads + nThreads / 32 + 5) * sizeof(float);
	shmSize = (3 * nThreads + 4) * sizeof(float);

	getKVectorIDsKernelFast<<<grid, block, shmSize>>>(selectIdx, nVec, d_dbIdx,
			d_binPrefix, d_binCounts, d_nBins, _Q, _bins, _nBins, _QN, d_dim,
			_maxBins, _k, maxVecConsider, maxVecOut, 128);
	// 280 for most results
#endif
	checkCudaErrors(cudaDeviceSynchronize());

//	outputVecUint("selectIdx", selectIdx, 1024);
//
//	outputVecUint("nVec", nVec, 100);

	std::cout << "multi Vector IDs done" << std::endl;
////	_QN = 1;
//
//	outputVecUint("selectIdx", selectIdx, 100);

//	uint maxVec = 2 * log2(_k);
	uint maxVec = log2(_k);

	nThreads = (maxVec > d_dim) ? maxVec : d_dim;

	nThreads = (nThreads < 1024) ? nThreads : 1024;

	block = dim3(nThreads, 1, 1);

	cudaMemset(selectIdx, 0, maxVec * _QN * sizeof(uint));

//	cudaMemcpy(_bestIdx, selectIdx, maxVec * _QN * sizeof(uint),
//			cudaMemcpyDeviceToDevice);
//
//	cudaMemcpy(&_resIdx[0], selectIdx, _QN * maxVec * sizeof(uint),
//			cudaMemcpyDeviceToHost);

//	checkCudaErrors(cudaDeviceSynchronize());

	std::cout << "copy down done " << std::endl;

	// now assemble lines on CPU:

	float *cLines = new float[_QN * maxVec * d_lineParts];

	if (!cLines) {
		cerr << "did not get cLines memory" << std::endl;
		exit(1);
	}

	for (int c = 0, i = 0; i < _QN * maxVec; i++) {
		size_t idx = _resIdx[i];
		idx *= 16;
//		idx = idx % 1000000000;

		for (int k = 0; k < d_lineParts; k++) {
			cLines[c++] = _hLines[idx + k];
		}

	}
//
//	for (int k = 0; k < 1600; k++) {
//		std::cout << " " << cLines[k] << std::endl;
//	}
	std::cout << "CPU assembled" << std::endl;

	cudaMemcpy(d_lineLambda, cLines, _QN * maxVec * d_lineParts * sizeof(float),
			cudaMemcpyHostToDevice);

	std::cout << "done line assembly" << std::endl;

	outputVecUint("before", selectIdx, 200);
#if 1
	if (maxVec <= 4096) {

		std::cout << "output maxVec: " << maxVec << std::endl;
		std::cout << "nThreads:  " << nThreads << std::endl;

		std::cout << "d_nClusters: " << d_nClusters << std::endl;
		std::cout << "d_lineParts: " << d_lineParts << std::endl;

		shmSize = (d_nClusters * d_lineParts + 2 * maxVec
				+ nThreads / d_lineParts + 2) * sizeof(float);

		std::cout << "shmSize rerank: " << shmSize << std::endl;

		std::cout << "maxVec " << maxVec << " k: " << _k << std::endl;

		cudaMemset(_bestIdx, 0, maxVec * _QN * sizeof(uint));
		rerankKernelDirectFast<<<grid, block, shmSize>>>(_bestDist, _bestIdx,
				selectIdx, nVec, maxVecOut, d_codeBookDistL1L2, d_nClusters,
				d_lineParts, d_lineLambda, _queryL1, _QN, d_dim, _maxBins, _k,
				maxVec);

//		outputVecUint("after", _bestIdx, 3496);

		uint* sel = new uint[maxVec * _QN];
		uint* best = new uint[maxVec * _QN];

		cudaMemcpy(sel, selectIdx, maxVec * _QN * sizeof(uint),
				cudaMemcpyDeviceToHost);
//		cudaMemcpy(_bestIdx, selectIdx, maxVec * _QN * sizeof(uint), cudaMemcpyDeviceToDevice);

		cudaMemcpy(best, _bestIdx, maxVec * _QN * sizeof(uint),
				cudaMemcpyDeviceToHost);
		bool same = true;
		for (int i = 0; i < maxVec * _QN; i++) {
			if (sel[i] != best[i]) {
				same = false;
				std::cout << i << ": " << sel[i] << " " << best[i] << std::endl;
			}

		}

		std::cout << "comoparison: " << ((same) ? " same" : "different") << std::endl;

		delete[] best;
		delete[] sel;

	} else {

//		shmSize = (d_nClusters * d_lineParts + 2 * 4096 + nThreads / d_lineParts
//				+ 2) * sizeof(float);
//
//		std::cout << "rerank: maxVec: " << maxVec << " k: " << _k << " shm: "
//		<< shmSize << std::endl;
//
//		rerankKernelFastLoop<<<grid, block, shmSize>>>(_bestDist, _bestIdx,
//				selectIdx, nVec, maxVecOut, d_codeBookDistL1L2, d_nClusters,
//				d_lineParts, d_lineLambda, _queryL1, _QN, d_dim, _maxBins, _k,
//				maxVec);
////
////		checkCudaErrors(cudaDeviceSynchronize());
//
//		thrust::device_ptr<float> bdt( _bestDist );
//		thrust::device_ptr<uint> bit( _bestIdx );
//
//		for (int i =0; i < _QN; i++)
//		thrust::sort_by_key( bdt + i * _k , bdt + i* _k + _k, bit + i * _k);
//
//		std::cout << "done sort by key" << std::endl;

//		shmSize = (6 * 1024 + 2) * sizeof(float);
//
//		std::cout << "mergeKernel: shm: " << shmSize << std::endl;
//
//		mergeKernel<<<grid, block, shmSize>>>(_bestDist, _bestIdx, _QN, _k);

	}
#endif

	checkCudaErrors(cudaDeviceSynchronize());

	std::cout << "rerankKBestVectors done " << std::endl;
//	outputVec("BestDist", _bestDist, 1000);

	delete[] cLines;

	cudaFree(nVec);
	cudaFree(selectIdx);

}

void PerturbationProTree::rerankKBestBinVectors(float *_bestDist,
		uint *_bestIdx, const float* _queryL1, const float* _assignVal,
		const uint* _assignIdx, uint _maxBins, uint _k1, uint k2,
		const float* _Q, uint _QN, uint _k) {

	uint nnn = log2(_k);

//	nnn = 1024;

	std::cout << "nnn: " << nnn << std::endl;

	uint maxVecConsider = nnn;
	uint maxVecOut = nnn;
	uint nThreads = (maxVecConsider < 1024) ? maxVecConsider : 1024;
	dim3 block(nThreads, 1, 1);
	dim3 grid((_QN < 1024) ? _QN : 1024, 1, 1);

	uint *selectIdx;// array for storing nn vector IDs (size maxVecOut * _QN)
	uint *nVec;

	cudaMalloc(&selectIdx, _QN * _k * sizeof(uint));
	cudaMalloc(&nVec, _QN * sizeof(uint));

	if ((!selectIdx) || (!nVec)) {
		std::cout << "getKBestVectors: did not get memory !" << std::endl;

		exit(1);
	}

	cudaMemset(selectIdx, 0, _QN * _k * sizeof(uint));

//	shmSize = (3* nThreads + nThreads / 32 + 5) * sizeof(float);
	uint shmSize = (d_p * _k1 * d_nClusters2 + 2 * d_p + 2 * nThreads + 4)
			* sizeof(float);

	getKBinVectorIDsKernelFast<<<grid, block, shmSize>>>(selectIdx, nVec,
			_assignVal, _assignIdx, d_p, _k1, d_nClusters, d_nClusters2,
			d_dbIdx, d_binPrefix, d_binCounts, d_nBins, d_distSeq, d_numDistSeq,
			d_distCluster, _QN, _maxBins, maxVecOut, 280);

	checkCudaErrors(cudaDeviceSynchronize());

//	outputVecUint("selectIdx", selectIdx, 1024);

	outputVecUint("nVec", nVec, 100);

//	std::cout << "multi Vector IDs done" << std::endl;
////	_QN = 1;
//
//	outputVecUint("selectIdx", selectIdx, 100);

//	uint maxVec = 2 * log2(_k);
	uint maxVec = log2(_k);

	nThreads = (maxVec > d_dim) ? maxVec : d_dim;

	nThreads = (nThreads < 1024) ? nThreads : 1024;

	block = dim3(nThreads, 1, 1);

//	grid = dim3(1, 1, 1);

//	shmSize = (d_dim + d_nClusters * d_lineParts + 2 * maxVec + 2)
//			* sizeof(float);

	std::cout << "maxVec: " << maxVec << " shm: " << shmSize << std::endl;

//	rerankKernel<<<grid, block, shmSize>>>(_bestDist, _bestIdx, selectIdx, nVec,
//			maxVecOut, d_codeBookDistL1L2, d_nClusters, d_lineParts,
//			d_lineLambda, d_lineP1, d_lineP2, _queryL1, _QN, d_dim, _maxBins,
//			_k, maxVec);

	shmSize = (d_dim + d_nClusters * d_lineParts + 2 * maxVec + nThreads
			+ nThreads / d_lineParts + 2) * sizeof(float);
	rerankKernelFast<<<grid, block, shmSize>>>(_bestDist, _bestIdx, selectIdx,
			nVec, maxVecOut, d_codeBookDistL1L2, d_nClusters, d_lineParts,
			d_lineLambda, _queryL1, _QN, d_dim, _maxBins, _k, maxVec);

	checkCudaErrors(cudaDeviceSynchronize());

	std::cout << "rerankKBestVectors done " << std::endl;

	outputVec("BestDist", _bestDist, 1000);

	cudaFree(nVec);
	cudaFree(selectIdx);

}

#if 0
__global__ void getBestLineVectorsKernel(float *_bestDist, uint* _bestIdx,
		const uint* _inIdx, const uint* _nVec, uint _maxVecIn,
		const float* _lambda, const uint* _l1Idx, const uint* _l2Idx,
		const float *_Ql1dist, const float *_Ql2dist, uint _QN, uint d_p,
		const float* _codeBookDistL1L2, uint _maxBins, uint _k, uint _maxVec) {

	extern __shared__ float shm[];

	float* shmIter = shm + _dim;

	float* val = shmIter;
	shmIter += _maxVec;
	uint* idx = (uint*) shmIter;
	shmIter += _maxVec;

// in shm;
	uint &nVec(*(uint*) shmIter);
	shmIter++;

// loop over all corresponding vectors in the query
	for (int iter = blockIdx.x; iter < _QN; iter += gridDim.x) {

		__syncthreads();

		if (threadIdx.x == 0) {
			nVec = _nVec[iter];
			nVec = (nVec < _maxVec) ? nVec : _maxVec;

			if (iter % 1000 == 0) {
				printf("nVec: %d \n", nVec);
			}
		}
		__syncthreads();

		// load query vector
		float b;
		if (threadIdx.x < _dim)
		b = _Q[iter * _dim + threadIdx.x];

		// load all indices
		for (int a = threadIdx.x; a < nVec; a += blockDim.x) {
			idx[a] = _inIdx[iter * _maxVecIn + a];

			if (idx[a] >= 1000000) {
				printf("panic: %d %d %d %d \n ", idx[a], iter, a, nVec);
			}
		}

		__syncthreads();

//		if (threadIdx.x == 0) {
//			for (int a = 0; a < nVec; a++) {
//				printf("idx: %d %d \n", a, idx[a]);
//			}
//		}

		// loop over all selected vectors
		for (int a = 0; a < nVec; a++) {

			// compute the distance to the vector
//			if (threadIdx.x < _dim) {
//				uint loc = idx[a] * _dim + threadIdx.x;
//				float v = _dbVec[loc];
//
////				if ((blockIdx.x == 90) && (a == 110))
////					printf( "got: %d %d %f \n", loc, idx[a], v);
//
//				shm[threadIdx.x] = sqr( b - v );
//			}
			if (threadIdx.x < _dim) {
//				if (idx[a] < 1000000)

				shm[threadIdx.x] = sqr(b - _dbVec[idx[a] * _dim + threadIdx.x]);
			}

			for (uint stride = _dim >> 1; stride > 0; stride >>= 1) {
				__syncthreads();

				if (threadIdx.x < stride)
				shm[threadIdx.x] += shm[threadIdx.x + stride];
			}
			__syncthreads();

			// store the result
			if (threadIdx.x == 0) {

				val[a] = shm[0];

//				printf("idx: %d dist: %f \n", idx[a], val[a]);

			}
			__syncthreads();
		}

		// sort the results
		if ((threadIdx.x >= nVec) && (threadIdx.x < _maxVec))
		val[threadIdx.x] = 10000000.;

		__syncthreads();

		bitonic3(val, idx, _maxVec);

		if ((threadIdx.x >= nVec) && (threadIdx.x < _maxVec))
		val[threadIdx.x] = 0.;

		if (threadIdx.x < _k) {
			_bestDist[iter * _k + threadIdx.x] = val[threadIdx.x];
			_bestIdx[iter * _k + threadIdx.x] = idx[threadIdx.x];
		}

		__syncthreads();

	}
}

#endif

// each block is responsible for one vector, blockDim.x should be _dim
// requires _dim * float shared memory
// looping multiple times to process all B vectors
// _vl is the length of the _p vector segments (should be 2^n)
// 			 k0			k1			k2		k3		k4
// output    p0,p1,..   p0,p1,..	..
__global__ void assignPerturbationKBestClusterKernel(uint *_assign,
		const float* _A, const float* _B, uint _Arows, uint _Brows, uint _dim,
		uint _p, uint _vl, uint _k, uint _NP2, uint _dimBits) {

	extern __shared__ float shmb[];

	float* shm = shmb + _dim;

	float* shmIter = shm;
	shmIter += _NP2;
	uint* shmIdx = (uint*) shmIter;
	shmIter += _NP2;

	uint offs = (2 * _NP2 > _dim) ? 2 * _NP2 : _dim;

	shmIter = shm + offs;

	float* val = shmIter;
	shmIter += _p * _Arows;
	uint* idx = (uint*) (shmIter);

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		__syncthreads();

		if (threadIdx.x < _dim)
			shmb[threadIdx.x] = _B[iter * _dim + threadIdx.x];

		for (uint pert = 0; pert < 1; pert++) {

			__syncthreads();

			uint pIdx = pertIdx(threadIdx.x, _dimBits, pert);
			// load perturbed vector
			float b = shmb[pIdx];

			const float* A = _A + pert * _Arows * _dim;

			// loop over all vectors of A
			for (int a = 0; a < _Arows; a++) {
				if (threadIdx.x < _dim)
					shm[threadIdx.x] = sqr(b - A[a * _dim + threadIdx.x]);

				// compute the sum of differences for the vector segments
				// i.e. all _p segments in parallel
				for (uint stride = _vl >> 1; stride > 0; stride >>= 1) {
					__syncthreads();

					if ((threadIdx.x) < _p * stride) {
						uint p = threadIdx.x / stride * _vl;
						uint bias = threadIdx.x % stride;
						shm[p + bias] += shm[p + bias + stride];
					}
				}
				__syncthreads();

				// store the result
				if (threadIdx.x < _p) {

					val[a + threadIdx.x * _Arows] = shm[threadIdx.x * _vl];
					idx[a + threadIdx.x * _Arows] = a;

//				if ((threadIdx.x == 0)) //  && (a < 10))
//					printf("idx %d %f \n", idx[a], val[a]);

				}

				__syncthreads();
			}

			// sort the results;
			__syncthreads();

			for (int i = 0; i < _p; i++) {

				if (threadIdx.x < _NP2)
					shm[threadIdx.x] = 10000000.;

				// copy to original shm
				if (threadIdx.x < _Arows) {
					shm[threadIdx.x] = val[threadIdx.x + i * _Arows];
					shmIdx[threadIdx.x] = idx[threadIdx.x + i * _Arows];
				}
				__syncthreads();

				bitonic3(shm, shmIdx, _NP2);

				if (threadIdx.x < _Arows) {
					val[threadIdx.x + i * _Arows] = shm[threadIdx.x];
					idx[threadIdx.x + i * _Arows] = shmIdx[threadIdx.x];
				}

				__syncthreads();

			}

			// write out decision
			for (int k = 0; k < _k; k++) {
				if (threadIdx.x < _p) {
					_assign[(iter * 1 + pert) * _k * _p + k * _p
							+ threadIdx.x] = idx[threadIdx.x * _Arows + k];

				}
			}
		}
	} // iter
}

// each block is responsible for one vector, blockDim.x should be _dim
// requires _dim * float shared memory
// looping multiple times to process all B vectors
// _vl is the length of the _p vector segments (should be 2^n)
// 			 k0			k1			k2		k3		k4
// output    p0,p1,..   p0,p1,..	..
__global__ void assignPerturbationKBestLineClusterKernel(uint *_assign,
		float* _l1Dist, const float* _A, const float* _B, uint _Arows,
		uint _Brows, uint _dim, uint _p, uint _vl, uint _k, uint _NP2,
		uint _dimBits) {

	extern __shared__ float shmb[];

	float* shm = shmb + _dim;

	float* shmIter = shm;
	shmIter += _NP2;
	uint* shmIdx = (uint*) shmIter;
	shmIter += _NP2;

	uint offs = (2 * _NP2 > _dim) ? 2 * _NP2 : _dim;

	shmIter = shm + offs;

	float* val = shmIter;
	shmIter += _p * _Arows;
	uint* idx = (uint*) (shmIter);

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		__syncthreads();

		if (threadIdx.x < _dim)
			shmb[threadIdx.x] = _B[iter * _dim + threadIdx.x];

		for (uint pert = 0; pert < 1; pert++) {

			__syncthreads();

			uint pIdx = pertIdx(threadIdx.x, _dimBits, pert);
			// load perturbed vector
			float b = shmb[pIdx];

			const float* A = _A + pert * _Arows * _dim;

			// loop over all vectors of A
			for (int a = 0; a < _Arows; a++) {
				if (threadIdx.x < _dim)
					shm[threadIdx.x] = sqr(b - A[a * _dim + threadIdx.x]);

				// compute the sum of differences for the vector segments
				// i.e. all _p segments in parallel
				for (uint stride = _vl >> 1; stride > 0; stride >>= 1) {
					__syncthreads();

					if ((threadIdx.x) < _p * stride) {
						uint p = threadIdx.x / stride * _vl;
						uint bias = threadIdx.x % stride;
						shm[p + bias] += shm[p + bias + stride];
					}
				}
				__syncthreads();

				// store the result
				if (threadIdx.x < _p) {

					val[a + threadIdx.x * _Arows] = shm[threadIdx.x * _vl];
					idx[a + threadIdx.x * _Arows] = a;

//				if ((threadIdx.x == 0)) //  && (a < 10))
//					printf("idx %d %f \n", idx[a], val[a]);

				}

				__syncthreads();
			}

			// sort the results;
			__syncthreads();

			for (int i = 0; i < _p; i++) {

				if (threadIdx.x < _NP2)
					shm[threadIdx.x] = 10000000.;

				// copy to original shm
				if (threadIdx.x < _Arows) {
					shm[threadIdx.x] = val[threadIdx.x + i * _Arows];
					shmIdx[threadIdx.x] = idx[threadIdx.x + i * _Arows];
				}
				__syncthreads();

				bitonic3(shm, shmIdx, _NP2);

				if (threadIdx.x < _Arows) {
					val[threadIdx.x + i * _Arows] = shm[threadIdx.x];
					idx[threadIdx.x + i * _Arows] = shmIdx[threadIdx.x];
				}

				__syncthreads();

			}

			// write out l1 distance
			if ((threadIdx.x < _p) && (pert == 0))
				_l1Dist[(iter * _p) + threadIdx.x] = val[threadIdx.x * _Arows];

			// write out decision
			for (int k = 0; k < _k; k++) {
				if (threadIdx.x < _p) {
					_assign[(iter * 1 + pert) * _k * _p + k * _p
							+ threadIdx.x] = idx[threadIdx.x * _Arows + k];
				}

			}
		}
	} // iter
}

__global__ void lineClusterKernel(float *_lineLambda, uint *_lineP1,
		uint *_lineP2, float* _lineDist, const float* _cbDist, const float* _cb,
		uint _nClusters, const float* _B, uint _Brows, uint _dim, uint _p,
		uint _vl) {

	extern __shared__ float shm[];

	float* shmIter = shm;
	shmIter += _dim;

	float* val = shmIter;
	shmIter += _p * _nClusters;

	float* minD = shmIter;
	shmIter += _p;
	uint* minIdx = (uint*) shmIter;
	shmIter += _p;

	float* lambda = shmIter;
	shmIter += _p * _nClusters;
	float* dist = shmIter;
	shmIter += _p * _nClusters;
	float* l1Idx = shmIter;
	shmIter += _p * _nClusters;
	float* l2Idx = shmIter;
	shmIter += _p * _nClusters;

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		__syncthreads();

		float b;
		// load vector
		if (threadIdx.x < _dim)
			b = _B[iter * _dim + threadIdx.x];

		const float* A = _cb;

		// loop over all vectors of A
		for (int a = 0; a < _nClusters; a++) {
			if (threadIdx.x < _dim)
				shm[threadIdx.x] = sqr(b - A[a * _dim + threadIdx.x]);

			// compute the sum of differences for the vector segments
			// i.e. all _p segments in parallel
			for (uint stride = _vl >> 1; stride > 0; stride >>= 1) {
				__syncthreads();

				if ((threadIdx.x) < _p * stride) {
					uint p = threadIdx.x / stride * _vl;
					uint bias = threadIdx.x % stride;
					shm[p + bias] += shm[p + bias + stride];
				}
			}
			__syncthreads();

			// store the result
			if (threadIdx.x < _p) {
				float v = shm[threadIdx.x * _vl];
				val[a + threadIdx.x * _nClusters] = v;

				// determine closest center
				if ((a == 0) || (minD[threadIdx.x] > v)) {
					minD[threadIdx.x] = v;
					minIdx[threadIdx.x] = a;
				}

			}

			__syncthreads();
		}

		__syncthreads();

//		if ((threadIdx.x == 0)) {
//
//			printf("yea yea yea \n");
//
//			for (int p = 0; p < _p; p++) {
//				for (int a = 0; a < _nClusters; a++)
//					printf(" %.3f ", val[a + p * _nClusters]);
//
//				printf( "   min: %d %.3f", minIdx[p], minD[p]);
//				printf("\n");
//			}
//
//
//		}
//
//		__syncthreads();

		uint p = threadIdx.x / _nClusters;
		uint cIdx = threadIdx.x % _nClusters;

#if 0
		if (p < _p) {

			float c2 = _cbDist[minIdx[p] * _nClusters * _p + cIdx * _p + p];

			lambda[threadIdx.x] = project(val[threadIdx.x], minD[p], c2,
					dist[threadIdx.x]);

			if (cIdx == minIdx[p] )
			dist[threadIdx.x] = 999999999999.;

			if (!isTriangle(val[threadIdx.x], minD[p], c2))
			printf(
					"non-triangle: p %d pIdx %d: minIdx: %d === %f %f %f = %f %f \n",
					p, cIdx, minIdx[p], val[threadIdx.x], minD[p], c2,
					dist[threadIdx.x], lambda[threadIdx.x]);

			lIdx[threadIdx.x] = cIdx;

		}
#endif

		uint bestIdx1;
		uint bestIdx2;
		for (int minId = 0; minId < _nClusters; minId++) {

			if (p < _p) {

				float c2 = _cbDist[minId * _nClusters * _p + cIdx * _p + p];

				float d;
				float l;
				l = project(val[threadIdx.x], val[p * _nClusters + minId], c2,
						d);

				if (cIdx == minId)
					d = 999999999999.;

				if ((minId == 0) || (d < dist[threadIdx.x])) {
					dist[threadIdx.x] = d;
					lambda[threadIdx.x] = l;
					bestIdx1 = cIdx;
					bestIdx2 = minId;
				}

				if (!isTriangle(val[threadIdx.x], val[p * _nClusters + minId],
						c2))
					printf(
							"non-triangle: p %d pIdx %d: minIdx: %d === %f %f %f = %f %f \n",
							p, cIdx, minIdx[p], val[threadIdx.x], minD[p], c2,
							dist[threadIdx.x], lambda[threadIdx.x]);

			}
		}

		if (p < _p) {
			l1Idx[threadIdx.x] = bestIdx1;
			l2Idx[threadIdx.x] = bestIdx2;
		}

		__syncthreads();
#if 1
		// reduction to find best axis
		for (int stride = _nClusters >> 1; stride > 0; stride >>= 1) {
			__syncthreads();
			if ((p < _p) && (cIdx < stride)) {
				if (dist[threadIdx.x] > dist[threadIdx.x + stride]) {
					dist[threadIdx.x] = dist[threadIdx.x + stride];
					l1Idx[threadIdx.x] = l1Idx[threadIdx.x + stride];
					l2Idx[threadIdx.x] = l2Idx[threadIdx.x + stride];
					lambda[threadIdx.x] = lambda[threadIdx.x + stride];
				}
			}
		}

		__syncthreads();
#endif

		// write results
		if (threadIdx.x < _p) {
			_lineP1[iter * _p + threadIdx.x] = l1Idx[threadIdx.x * _nClusters];
			_lineP2[iter * _p + threadIdx.x] = l2Idx[threadIdx.x * _nClusters];
			_lineLambda[iter * _p + threadIdx.x] = lambda[threadIdx.x
					* _nClusters];
		}

		if (_lineDist) {
			if (threadIdx.x == 0) {
				float d = 0.;
				for (int i = 0; i < _p; i++)
					d += dist[i * _nClusters];
				_lineDist[iter] = d;
			}
		}

	} // iter
}

__global__ void lineClusterKernelFast(float *_lineLambda, float* _lineDist,
		const float* _cbDist, const float* _cb, uint _nClusters,
		const float* _B, uint _Brows, uint _dim, uint _p, uint _vl) {

	extern __shared__ float shm[];

	float* shmIter = shm;
	shmIter += _dim;

	float* val = shmIter;
	shmIter += _p * _nClusters;

//	float* minD = shmIter;
//	shmIter += _p;
//	uint* minIdx = (uint*) shmIter;
//	shmIter += _p;

	float* lambda = shmIter;
	shmIter += _p * _nClusters;
	float* dist = shmIter;
	shmIter += _p * _nClusters;
//	float* l1Idx = shmIter;
//	shmIter += _p * _nClusters;
//	float* l2Idx = shmIter;
//	shmIter += _p * _nClusters;

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		__syncthreads();

		float b;
		// load vector
		if (threadIdx.x < _dim)
			b = _B[iter * _dim + threadIdx.x];

		const float* A = _cb;

		// loop over all vectors of A
		for (int a = 0; a < _nClusters; a++) {
			if (threadIdx.x < _dim)
				shm[threadIdx.x] = sqr(b - A[a * _dim + threadIdx.x]);

			// compute the sum of differences for the vector segments
			// i.e. all _p segments in parallel
			for (uint stride = _vl >> 1; stride > 0; stride >>= 1) {
				__syncthreads();

				if ((threadIdx.x) < _p * stride) {
					uint p = threadIdx.x / stride * _vl;
					uint bias = threadIdx.x % stride;
					shm[p + bias] += shm[p + bias + stride];
				}
			}
			__syncthreads();

			// store the result
			if (threadIdx.x < _p) {
				float v = shm[threadIdx.x * _vl];
				val[a + threadIdx.x * _nClusters] = v;

//				// determine closest center
//				if ((a == 0) || (minD[threadIdx.x] > v)) {
//					minD[threadIdx.x] = v;
//					minIdx[threadIdx.x] = a;
//				}

			}

			__syncthreads();
		}

		__syncthreads();

		uint p = threadIdx.x / _nClusters;
		uint cIdx = threadIdx.x % _nClusters;

		for (int minId = 0; minId < _nClusters; minId++) {

			if (p < _p) {

				float c2 = _cbDist[minId * _nClusters * _p + cIdx * _p + p];

				float d;
				float l;
				l = project(val[threadIdx.x], val[p * _nClusters + minId], c2,
						d);

				if (cIdx == minId)
					d = 999999999999.;

				if ((minId == 0) || (d < dist[threadIdx.x])) {
					dist[threadIdx.x] = d;
					lineDescr& line(*((lineDescr*) (lambda + threadIdx.x)));

					line.p1 = cIdx;
					line.p2 = minId;
					line.lambda = toUShort(l);
				}

			}
		}

		__syncthreads();

		// reduction to find best axis
		for (int stride = _nClusters >> 1; stride > 0; stride >>= 1) {
			__syncthreads();
			if ((p < _p) && (cIdx < stride)) {
				if (dist[threadIdx.x] > dist[threadIdx.x + stride]) {
					dist[threadIdx.x] = dist[threadIdx.x + stride];
					lambda[threadIdx.x] = lambda[threadIdx.x + stride];
				}
			}
		}

		__syncthreads();

		// write results
		if (threadIdx.x < _p) {
			_lineLambda[iter * _p + threadIdx.x] = lambda[threadIdx.x
					* _nClusters];
		}

		if (_lineDist) {
			if (threadIdx.x == 0) {
				float d = 0.;
				for (int i = 0; i < _p; i++)
					d += dist[i * _nClusters];
				_lineDist[iter] = d;
			}
		}

	} // iter
}

void PerturbationProTree::lineDist(const float* _DB, uint _N) {

	d_lineParts = 16;

	//TODO
//	d_lineParts = 32;

	if (!d_codeBookDistL1L2)
		computeCBL1L1Dist(d_lineParts);

	d_lineP1 = NULL;
	d_lineP2 = NULL;

//	_N = 1000;

	float* dist = NULL;

	if (!d_lineLambda)
		cudaMalloc(&d_lineLambda, d_lineParts * _N * sizeof(float));
//
//	if (!d_lineP1)
//		cudaMalloc(&d_lineP1, d_lineParts * _N * sizeof(uint));
//	if (!d_lineP2)
//		cudaMalloc(&d_lineP2, d_lineParts * _N * sizeof(uint));

	if (!d_lineLambda) {
		std::cout << "line Dist: did not get memory " << std::endl;
	}

	cudaMalloc(&dist, _N * sizeof(float));

	uint nLines = d_lineParts * d_nClusters;
	uint nThreads = (d_dim > nLines) ? d_dim : nLines;

	dim3 block(nThreads, 1, 1);
	dim3 grid((_N > 1024) ? 1024 : _N, 1, 1);

//	dim3 grid(1,1,1);
	uint shmSize = (d_lineParts * d_nClusters + 2 * d_lineParts + d_dim
			+ 4 * nLines) * sizeof(float);

	std::cout << "shmSize: " << shmSize << std::endl;
//	lineClusterKernel<<<grid, block, shmSize>>>(d_lineLambda, d_lineP1,
//			d_lineP2, dist, d_codeBookDistL1L2, d_multiCodeBook, d_nClusters,
//			_DB, _N, d_dim, d_lineParts, d_dim / d_lineParts);

	lineClusterKernelFast<<<grid, block, shmSize>>>(d_lineLambda, dist,
			d_codeBookDistL1L2, d_multiCodeBook, d_nClusters, _DB, _N, d_dim,
			d_lineParts, d_dim / d_lineParts);

	checkCudaErrors(cudaDeviceSynchronize());

//	outputVec("Dist ", dist, 1000);
//
//	outputVec("Lambda", d_lineLambda, 1000);

	float minD, maxD, avgD;
	float* ldist = new float[_N];
	cudaMemcpy(ldist, dist, _N * sizeof(float), cudaMemcpyDeviceToHost);

	minD = ldist[0];
	maxD = ldist[0];
	avgD = 0;
	for (int i = 0; i < _N; i++) {
		if (ldist[i] < minD)
			minD = ldist[i];
		if (ldist[i] > maxD)
			maxD = ldist[i];
		avgD += ldist[i];
	}
	std::cout << "line dist (min, max, avg)    " << minD << " " << maxD << " "
			<< (avgD / _N) << std::endl;

	cudaFree(dist);
}

__global__ void lineAssignmentKernel(float *_queryDist, const float* _cb,
		uint _nClusters, const float* _B, uint _Brows, uint _dim, uint _p,
		uint _vl) {

	extern __shared__ float shm[];

	float* shmIter = shm;
	shmIter += _dim;

	float* val = shmIter;
	shmIter += _p * _nClusters;

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		__syncthreads();

		float b;
		// load vector
		if (threadIdx.x < _dim)
			b = _B[iter * _dim + threadIdx.x];

		const float* A = _cb;

		// loop over all vectors of A
		for (int a = 0; a < _nClusters; a++) {
			if (threadIdx.x < _dim)
				shm[threadIdx.x] = sqr(b - A[a * _dim + threadIdx.x]);

			// compute the sum of differences for the vector segments
			// i.e. all _p segments in parallel
			for (uint stride = _vl >> 1; stride > 0; stride >>= 1) {
				__syncthreads();

				if ((threadIdx.x) < _p * stride) {
					uint p = threadIdx.x / stride * _vl;
					uint bias = threadIdx.x % stride;
					shm[p + bias] += shm[p + bias + stride];
				}
			}
			__syncthreads();

			// store the result
			if (threadIdx.x < _p) {
				float v = shm[threadIdx.x * _vl];
				val[a + threadIdx.x * _nClusters] = v;
			}

			__syncthreads();
		}

		__syncthreads();

		if (threadIdx.x < _nClusters) {
			for (int p = 0; p < _p; p++)
				_queryDist[iter * _nClusters * _p + p * _nClusters + threadIdx.x] =
						val[p * _nClusters + threadIdx.x];
		}

	}
}

void PerturbationProTree::getLineAssignment(float* _queryL1Dist,
		const float* _Q, uint _QN) {

	uint nLines = d_lineParts * d_nClusters;
	uint nThreads = (d_dim > nLines) ? d_dim : nLines;

	dim3 block(nThreads, 1, 1);
	dim3 grid((_QN > 1024) ? 1024 : _QN, 1, 1);

	uint shmSize = (d_dim + d_lineParts * d_nClusters) * sizeof(float);

	std::cout << "shmSize: " << shmSize << std::endl;
	lineAssignmentKernel<<<grid, block, shmSize>>>(_queryL1Dist,
			d_multiCodeBook, d_nClusters, _Q, _QN, d_dim, d_lineParts,
			d_dim / d_lineParts);

	checkCudaErrors(cudaDeviceSynchronize());

//	outputVec("qL1", _queryL1Dist, d_nClusters * d_lineParts);
}

__global__ void assignPerturbationKBestClusterKernelSingleP(uint *_assign,
		const float* _A, const float* _B, uint _Arows, uint _Brows, uint _dim,
		uint _p, uint _vl, uint _k, uint _NP2, uint _dimBits) {

	extern __shared__ float shmb[];

	float* shm = shmb + _dim;

	uint* shmIdx = (uint*) (shm + _NP2);

	float val[16];

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		__syncthreads();

		if (threadIdx.x < _dim)
			shmb[threadIdx.x] = _B[iter * _dim + threadIdx.x];

		for (uint pert = 0; pert < 1; pert++) {
			__syncthreads();

			uint pIdx = pertIdx(threadIdx.x, _dimBits, pert);
			// load perturbed vector
			float b = shmb[pIdx];

			const float* A = _A + pert * _Arows * _dim;

			// loop over all vectors of A
			for (int a = 0; a < _Arows; a++) {

				if (threadIdx.x < _dim)
					shm[threadIdx.x] = sqr(b - A[a * _dim + threadIdx.x]);

				__syncthreads();

				// store the result
				if (threadIdx.x == a) {
					for (int p = 0; p < _p; p++) {
						val[p] = 0;
						for (int i = 0; i < _vl; i++) {
							val[p] += shm[p * _vl + i];
						}
					}
				}

				__syncthreads();
			}

			// sort the results;

			for (int i = 0; i < _p; i++) {

				if (threadIdx.x < _NP2)
					shm[threadIdx.x] = 10000000.;

				// copy to original shm
				if (threadIdx.x < _Arows) {
					shm[threadIdx.x] = val[i];
					shmIdx[threadIdx.x] = threadIdx.x;
				}
				__syncthreads();

				bitonic3(shm, shmIdx, _NP2);

				val[i] = shmIdx[threadIdx.x];

				__syncthreads();

			}

			// TOOD optimize
			// write out decision
			for (int k = 0; k < _k; k++) {
				if (threadIdx.x == k) {
					for (int i = 0; i < _p; i++)
						shmIdx[i] = val[i];
				}
				__syncthreads();
				if (threadIdx.x < _p) {
					_assign[(iter * 1 + pert) * _k * _p + k * _p
							+ threadIdx.x] = shmIdx[threadIdx.x];

				}
				__syncthreads();
			}

		}
	} // iter
}

void PerturbationProTree::getKBestAssignment(uint *_assign, const float* _A,
		const float* _B, uint _Arows, uint _Brows, uint _k) const {

	if (_Arows <= 1024) {

		uint NP2 = log2(_Arows);

		uint nThreads = (NP2 > d_dim) ? NP2 : d_dim;

		dim3 block(nThreads, 1, 1);
		dim3 grid((_Brows < 1024) ? _Brows : 1024, 1, 1);

		uint shm = (2 * nThreads + 2 * _Arows * d_p) * sizeof(float);

		if (shm > 32000) {
			shm = (nThreads + 2 * _Arows) * sizeof(float);
			//	std::cout << "kbest single p : shm " << shm << std::endl;
			assignPerturbationKBestClusterKernelSingleP<<<grid, block, shm>>>(
					_assign, _A, _B, _Arows, _Brows, d_dim, d_p, d_vl, _k, NP2,
					d_dimBits);

		} else {

			// std::cout << "kbest: shm " << shm << std::endl;
			assignPerturbationKBestClusterKernel<<<grid, block, shm>>>(_assign,
					_A, _B, _Arows, _Brows, d_dim, d_p, d_vl, _k, NP2,
					d_dimBits);
		}
	} else {

		if (d_p > 1) {
			std::cout << "not implemented";
			return;
		}
		float* resd;
		uint* idxD;

		cudaMalloc(&resd, _Arows * _Brows * sizeof(float));
		cudaMalloc(&idxD, _Arows * _Brows * sizeof(uint));
		calcDist(resd, _A, _B, _Arows, _Brows);

		// initialize the key array to the trivial list 0,1, ... _Arows, 0, 1, ....
		uint* idx = new uint[_Arows * _Brows];
		uint h = 0;
		for (int j = 0; j < _Brows; j++)
			for (int i = 0; i < _Arows; i++, h++)
				idx[h] = i;
		cudaMemcpy(idxD, idx, _Arows * _Brows * sizeof(uint),
				cudaMemcpyHostToDevice);

		bitonicSort(resd, _assign, resd, idxD, _Brows, _Arows, 1);

		delete[] idx;
		cudaFree(idxD);
		cudaFree(resd);

	}
	checkCudaErrors(cudaDeviceSynchronize());
}

void PerturbationProTree::getKBestLineAssignment(uint *_assign, float* _l1Dist,
		const float* _A, const float* _B, uint _Arows, uint _Brows,
		uint _k) const {

	uint NP2 = log2(_Arows);

	uint nThreads = (NP2 > d_dim) ? NP2 : d_dim;

	dim3 block(nThreads, 1, 1);
	dim3 grid((_Brows < 1024) ? _Brows : 1024, 1, 1);

	uint shm = (2 * nThreads + 2 * _Arows * d_p) * sizeof(float);

	if (shm > 32000) {
		shm = (nThreads + 2 * _Arows) * sizeof(float);
		//	std::cout << "kbest single p : shm " << shm << std::endl;
		assignPerturbationKBestClusterKernelSingleP<<<grid, block, shm>>>(
				_assign, _A, _B, _Arows, _Brows, d_dim, d_p, d_vl, _k, NP2,
				d_dimBits);

		std::cout << "!!!!!!!!! not supported !!!!!!!!!!" << std::endl;

	} else {

		// std::cout << "kbest: shm " << shm << std::endl;
//		assignPerturbationKBestClusterKernel<<<grid, block, shm>>>(_assign, _A,
//				_B, _Arows, _Brows, d_dim, d_p, d_vl, _k, NP2, d_dimBits,
//				d_nDBs);

		assignPerturbationKBestLineClusterKernel<<<grid, block, shm>>>(_assign,
				_l1Dist, _A, _B, _Arows, _Brows, d_dim, d_p, d_vl, _k, NP2,
				d_dimBits);
	}

	checkCudaErrors(cudaDeviceSynchronize());
}

void PerturbationProTree::testKNN(const float* _Q, uint _QN) {

//	outputVecUint("prefix", d_binPrefix + 816791, 20);
//	outputVecUint("prefix", d_binCounts + 816791, 20);
//	outputVecUint("dbidx", d_dbIdx + 789975, 9);

	uint k1 = 4;

	prepareDistSequence(d_nClusters2 * k1, d_p);

	uint* assignd;
	uint* assignd2;
	cudaMalloc(&assignd, 1 * k1 * d_p * _QN * sizeof(uint));
	cudaMalloc(&assignd2, 1 * k1 * d_p * _QN * sizeof(uint));

	outputVec("mcodebook", d_multiCodeBook, 256);

	outputVec("mcodebook2", d_multiCodeBook2, 256);
	outputVec("mcodebook2 -p2",
			d_multiCodeBook2 + d_nClusters * d_nClusters2 * d_dim, 256);

	getKBestAssignment(assignd, d_multiCodeBook, _Q, d_nClusters, _QN, k1);

	outputVecUint("assign1 pert0: ", assignd, k1 * d_p);

	outputVecUint("assign1 pert1: ", assignd + k1 * d_p, k1 * d_p);

	float *assignVal;
	uint *assignIdx;

	cudaMalloc(&assignVal,
			_QN * 1 * d_p * k1 * d_nClusters2 * sizeof(float));
	cudaMalloc(&assignIdx,
			_QN * 1 * d_p * k1 * d_nClusters2 * sizeof(uint));

	getKBestAssignment2(assignVal, assignIdx, d_multiCodeBook2, _Q,
			d_nClusters2, _QN, assignd, d_nClusters, k1);

	outputVecUint("assignIdx 1 - p0", assignIdx, k1 * d_nClusters2);
	outputVecUint("assignIdx 1 - p1", assignIdx + k1 * d_nClusters2 * d_p,
			k1 * d_nClusters2);

	uint *idx = new uint[d_p * k1 * d_nClusters2];
	float *val = new float[d_p * k1 * d_nClusters2];

	cudaMemcpy(val, assignVal, d_p * k1 * d_nClusters2 * sizeof(float),
			cudaMemcpyDeviceToHost);

	for (int p = 0; p < d_p; p++) {
		for (int i = 0; i < k1 * d_nClusters2; i++)
			std::cout << val[p * k1 * d_nClusters2 + i] << " ";
		std::cout << std::endl << std::endl;
	}

	uint k2 = 40;
	k2 = 40960;
//	uint maxBins = 40;
	uint maxBins = 13000;

	uint* nBins;
	uint* bins;

	cudaMalloc(&nBins, 1 * _QN * sizeof(uint));
	cudaMalloc(&bins, 1 * _QN * maxBins * sizeof(uint));

//	getBins(bins, nBins, assignVal, assignIdx, _QN, k1, k2, maxBins);
	getBins(bins, nBins, assignVal, assignIdx, 1, k1, k2, maxBins);

	outputVecUint("received nBins: ", nBins, 1);

	std::cout << "done with bins!!!!!!" << std::endl;
	k2 = 256;
//	k2 = 512;
//	k2 = 64;
//	k2 = 4096;

	uint maxVec = k2;
	float* bestDist;
	uint* bestIdx;
	cudaMalloc(&bestDist, _QN * maxVec * sizeof(float));
	cudaMalloc(&bestIdx, _QN * maxVec * sizeof(uint));

//	getKBestVectors(bestDist, bestIdx, bins, nBins, maxBins, _Q, _QN, k2);

	if (k2 <= 1024)
		getKBestVectors(bestDist, bestIdx, bins, nBins, maxBins, _Q, 1, k2);
	else
		getKBestVectorsLarge(bestDist, bestIdx, bins, nBins, maxBins, _Q, 1,
				k2);

	uint* bestIdxh = new uint[maxVec];
	float* bestDisth = new float[maxVec];

	cudaMemcpy(bestIdxh, bestIdx, maxVec * sizeof(uint),
			cudaMemcpyDeviceToHost);
	cudaMemcpy(bestDisth, bestDist, maxVec * sizeof(float),
			cudaMemcpyDeviceToHost);

	for (int i = 0; i < maxVec; i++) {
		std::cout << i << " " << bestIdxh[i] << "  " << bestDisth[i] << std::endl;
	}

	std::cout << std::endl;

	float* resd;
	cudaMalloc(&resd, d_p * d_NdbVec * sizeof(float));
	calcDist(resd, d_dbVec, _Q, d_NdbVec, 1);

//outputVec("Res:", resd, 20);

	float* resh = new float[d_p * d_NdbVec];

	cudaMemcpy(resh, resd, d_p * d_NdbVec * sizeof(float),
			cudaMemcpyDeviceToHost);

	vector<pair<float, uint> > ddd;
	ddd.resize(d_NdbVec);

	for (int i = 0; i < d_NdbVec; i++) {
		float val = 0.;
		for (int p = 0; p < d_p; p++)
			val += resh[i * d_p + p];
		ddd[i] = pair<float, uint>(val, i);
	}

	sort(ddd.begin(), ddd.end());

	getAssignment(assignd, d_multiCodeBook, _Q, d_nClusters, 1);
	outputVecUint("assign: ", assignd, 4);

	getKBestAssignment(assignd, d_multiCodeBook, _Q, d_nClusters, 1, k1);
	outputVecUint("assign: ", assignd, k1 * d_p);
	outputVecUint("assignIdx2-1: ", assignIdx, k1);
	outputVecUint("assignIdx2-2: ", assignIdx + k1 * d_nClusters2, k1);

	std::cout << "distance by brute-force search: " << std::endl;
	for (int i = 0; i < 20; i++) {
		std::cout << i << "  " << ddd[i].first << "  " << ddd[i].second << std::endl;
		getKBestAssignment(assignd, d_multiCodeBook,
				d_dbVec + ddd[i].second * d_dim, d_nClusters, 1, k1);
//		outputVecUint("assign: ", assignd, k1 * d_p);
		getAssignment2(assignd2, d_multiCodeBook2,
				d_dbVec + ddd[i].second * d_dim, d_nClusters2, 1, assignd,
				d_nClusters);
//		outputVecUint("assign2: ", assignd2, d_p);
		getKBestAssignment2(assignVal, assignIdx, d_multiCodeBook2,
				d_dbVec + ddd[i].second * d_dim, d_nClusters2, 1, assignd,
				d_nClusters, k1);
		outputVecUint("assignIdx2-1: ", assignIdx, k1);
		outputVecUint("assignIdx2-2: ", assignIdx + k1 * d_nClusters2, k1);
//		outputVecUint("", assignd + 4, 4);
//		outputVecUint("", assignd + 8, 4);
	}

	cudaFree(bestIdx);
	cudaFree(bestDist);

	cudaFree(bins);
	cudaFree(nBins);
	cudaFree(assignIdx);
	cudaFree(assignVal);
	cudaFree(assignd2);
	cudaFree(assignd);

	delete[] val;
	delete[] idx;
}

void PerturbationProTree::queryKNN(vector<uint>& _resIdx,
		vector<float>& _resDist, const float* _Q, uint _QN, uint _kVec) {

	_resIdx.resize(_QN * _kVec);
	_resDist.resize(_QN * _kVec);

	uint k1 = 1;

	k1 = 8;

//	k1 = 16;

	prepareDistSequence(d_nClusters2 * k1, d_p);

//	k1 = 16;

//	prepareDistSequence(14, d_p);

	uint* assignd;
	float *assignVal;
	uint *assignIdx;

	cudaMalloc(&assignd, 1 * k1 * d_p * _QN * sizeof(uint));

	float *queryL1;
	cudaMalloc(&queryL1, d_nClusters * d_lineParts * _QN * sizeof(uint));

	cudaMalloc(&assignVal,
			_QN * 1 * d_p * k1 * d_nClusters2 * sizeof(float));
	cudaMalloc(&assignIdx,
			_QN * 1 * d_p * k1 * d_nClusters2 * sizeof(uint));

	uint k2 = 40;
	k2 = 4096;

//	k2 = 8192;

	//	k2 = _kVec;

	uint maxBins = 4096;

//	uint maxBins = 4* 8192;
//	uint maxBins = 2048;

	uint* nBins;
	uint* bins;

	cudaMalloc(&nBins, _QN * 1 * sizeof(uint));
	cudaMalloc(&bins, _QN * maxBins * 1 * sizeof(uint));

	if (!nBins || !bins) {
		std::cout << "Bins: did not get memory" << std::endl;
		exit(1);
	}

	uint maxVec = _kVec;
	float* bestDist;
	uint* bestIdx;
	cudaMalloc(&bestDist, _QN * maxVec * sizeof(float));
	cudaMalloc(&bestIdx, _QN * maxVec * sizeof(uint));

	for (int i = 0; i < 1; i++) {
		getKBestAssignment(assignd, d_multiCodeBook, _Q, d_nClusters, _QN, k1);

		getLineAssignment(queryL1, _Q, _QN);

		getKBestAssignment2(assignVal, assignIdx, d_multiCodeBook2, _Q,
				d_nClusters2, _QN, assignd, d_nClusters, k1);

//		outputVec("assignVal: ", assignVal, 200);
//		outputVecUint("assignIdx: ", assignIdx, 200);

		std::cout << "done assignements " << std::endl;

#if 1
		getBins(bins, nBins, assignVal, assignIdx, _QN, k1, k2, maxBins);

	// outputVecUint("Bins", bins, 2000);

		std::cout << "done bins " << std::endl;

//		if (maxVec <= 1024)
//		getKBestVectors(bestDist, bestIdx, bins, nBins, maxBins, _Q, _QN,
//					maxVec);
//		else
//			getKBestVectorsLarge(bestDist, bestIdx, bins, nBins, maxBins, _Q,
//					_QN, maxVec);

		rerankKBestVectors(bestDist, bestIdx, queryL1, bins, nBins, maxBins, _Q,
				_QN, maxVec);

#else
		rerankKBestBinVectors(bestDist, bestIdx, queryL1, assignVal, assignIdx,
				maxBins, k1, k2, _Q, _QN, maxVec);
#endif

		std::cout << "done vectors " << _QN << std::endl;
	}

	cudaMemcpy(&_resIdx[0], bestIdx, _QN * maxVec * sizeof(uint),
			cudaMemcpyDeviceToHost);
	cudaMemcpy(&_resDist[0], bestDist, _QN * maxVec * sizeof(float),
			cudaMemcpyDeviceToHost);

#if 0
	if (maxVec > 4096) {
		// sort the results on host
		vector< pair<float,uint> > svec;
		svec.resize(_QN*maxVec);

		std::cout << "dist before: ";
		for (int i = 0; i < 100; i++) {
			std::cout << "\t" << _resDist[i];
		}
		std::cout << std::endl;

		for (int i = 0; i < _QN * maxVec; i++) {
			svec[i] = pair<float,uint>(_resDist[i], _resIdx[i]);
		}
		for (int i = 0; i < _QN; i++) {
			sort(svec.begin() + i * maxVec, svec.begin() + i * maxVec + maxVec);
		}
		for (int i = 0; i < _QN * maxVec; i++) {
			_resIdx[i] = svec[i].second;
			_resDist[i] = svec[i].first;
		}

		std::cout << "dist: ";
		for (int i = 0; i < 00; i++) {
			std::cout << "\t" << _resDist[i];
		}
		std::cout << std::endl;

	}
#endif

	cudaFree(bestIdx);
	cudaFree(bestDist);
	cudaFree(bins);
	cudaFree(nBins);
	cudaFree(assignIdx);
	cudaFree(assignVal);
	cudaFree(assignd);

}

void PerturbationProTree::queryBIGKNN(vector<uint>& _resIdx,
		vector<float>& _resDist, const float* _Q, uint _QN, uint _kVec,
		const std::vector<uint>& _gtBins, uint _offset) {

	_resIdx.resize(_QN * _kVec);
	_resDist.resize(_QN * _kVec);

//	uint k1 = 32;
	uint k1 = 8;
	k1 = 16;

//	prepareDistSequence(d_nClusters2 * k1, d_p);

//	prepareDistSequence(14, d_p);

	uint* assignd;
	float *assignVal;
	uint *assignIdx;

	cudaMalloc(&assignd, 1 * k1 * d_p * _QN * sizeof(uint));

	float *queryL1;
	cudaMalloc(&queryL1, d_nClusters * d_lineParts * _QN * sizeof(uint));

	cudaMalloc(&assignVal,
			_QN * 1 * d_p * k1 * d_nClusters2 * sizeof(float));
	cudaMalloc(&assignIdx,
			_QN * 1 * d_p * k1 * d_nClusters2 * sizeof(uint));

//	uint k2 = 40;
//	k2 = 4096;

//	k2 = 8192;

	//	k2 = _kVec;

//	uint maxBins = 4096;

	uint maxBins = 64 * 8192;
//	uint maxBins = 2048;

//	uint maxBins = 16;

	uint* nBins;
	uint* bins;

	cudaMalloc(&nBins, _QN * 1 * sizeof(uint));
	cudaMalloc(&bins, _QN * maxBins * 1 * sizeof(uint));

	if (!nBins || !bins) {
		std::cout << "Bins: did not get memory" << std::endl;
		exit(1);
	}

	uint maxVec = _kVec;
	float* bestDist;
	uint* bestIdx;
	cudaMalloc(&bestDist, _QN * maxVec * sizeof(float));
	cudaMalloc(&bestIdx, _QN * maxVec * sizeof(uint));

	getKBestAssignment(assignd, d_multiCodeBook, _Q, d_nClusters, _QN, k1);

//	getLineAssignment(queryL1, _Q, _QN);

	getKBestAssignment2(assignVal, assignIdx, d_multiCodeBook2, _Q,
			d_nClusters2, _QN, assignd, d_nClusters, k1);

//		outputVec("assignVal: ", assignVal, 200);
//		outputVecUint("assignIdx: ", assignIdx, 200);

	std::cout << "done assignements " << std::endl;

#if 1
//	getBIGBins(bins, nBins, assignVal, assignIdx, _QN, k1, k2, maxBins);
//	getBIGBins(bins, nBins, assignVal, assignIdx, _QN, k1, _kVec, maxBins);
//	getBIGBins(bins, nBins, assignVal, assignIdx, _QN, k1, 4 * 8192, maxBins);

//	getBIGBinsSorted(bins, nBins, assignVal, assignIdx, _QN, k1, _kVec,
//			maxBins);
//
	getBIGBins2D(bins, nBins, assignVal, assignIdx, _QN, k1, _kVec, maxBins);

//	outputVecUint("final Bins", bins, _QN * maxBins);

//	outputVecUint("nBins: ", nBins, 2);
//	outputVecUint("nBins: ", nBins, 400);

	countZeros("bins: ", bins, _QN * maxBins);

	std::cout << "done bins " << std::endl;

	////////////////////////////////////////////////////////////////////////////////////
	// check Bins
	uint* hBins = new uint[_QN * maxBins];
	uint* hnBins = new uint[_QN];
	cudaMemcpy(hBins, bins, _QN * maxBins * sizeof(uint),
			cudaMemcpyDeviceToHost);
	cudaMemcpy(hnBins, nBins, _QN * sizeof(uint), cudaMemcpyDeviceToHost);

	uint found = 0;
	uint avg = 0;

	uint binMax = 0;
	uint binMin = 100000;
	for (int i = 0; i < _QN; i++) {

		int b = 0;
		for (; b < hnBins[i]; b++) {
			if (hBins[i * maxBins + b] == _gtBins[i + _offset]) {
				found++;
				avg += b;
				if (b < binMin)
					binMin = b;
				if (b > binMax)
					binMax = b;
				break;
			}
		}
	}

	std::cout << "found bins: " << found;
	std::cout << "  at avg location: " << (float(avg) / float(found));

	std::cout << " min " << binMin << " max " << binMax << std::endl;

	delete[] hnBins;
	delete[] hBins;

	// check Bins - End
	///////////////////////////////////////////////////////////////////////////////////

//		if (maxVec <= 1024)
//		getKBestVectors(bestDist, bestIdx, bins, nBins, maxBins, _Q, _QN,
//					maxVec);
//		else
//			getKBestVectorsLarge(bestDist, bestIdx, bins, nBins, maxBins, _Q,
//					_QN, maxVec);

	rerankBIGKBestVectors(bestDist, bestIdx, queryL1, bins, nBins, maxBins, _Q,
			_QN, maxVec);
#endif
	std::cout << "done vectors " << _QN << std::endl;

//	outputVecUint("BestIdx: ", bestIdx, _QN * maxVec);

	countZeros("bestIdx: ", bestIdx, _QN * maxVec);

	cudaMemcpy(&_resIdx[0], bestIdx, _QN * maxVec * sizeof(uint),
			cudaMemcpyDeviceToHost);
	cudaMemcpy(&_resDist[0], bestDist, _QN * maxVec * sizeof(float),
			cudaMemcpyDeviceToHost);

	cudaFree(bestIdx);
	cudaFree(bestDist);
	cudaFree(bins);
	cudaFree(nBins);
	cudaFree(assignIdx);
	cudaFree(assignVal);
	cudaFree(assignd);

}

void PerturbationProTree::queryBIGKNNRerank(vector<uint>& _resIdx,
		vector<float>& _resDist, const float* _Q, uint _QN, uint _kVec,
		const float* _hLines) {

	_resIdx.resize(_QN * _kVec);
	_resDist.resize(_QN * _kVec);

	d_lineParts = 16; // TODO !!!!

	if (!d_codeBookDistL1L2)
		computeCBL1L1Dist(d_lineParts);

	uint k1 = 16;

//	prepareDistSequence(d_nClusters2 * k1, d_p);

//	prepareDistSequence(14, d_p);

	uint* assignd;
	float *assignVal;
	uint *assignIdx;

	cudaMalloc(&assignd, 1 * k1 * d_p * _QN * sizeof(uint));

	float *queryL1;
	cudaMalloc(&queryL1, d_nClusters * d_lineParts * _QN * sizeof(uint));

	cudaMalloc(&assignVal,
			_QN * 1 * d_p * k1 * d_nClusters2 * sizeof(float));
	cudaMalloc(&assignIdx,
			_QN * 1 * d_p * k1 * d_nClusters2 * sizeof(uint));

	uint k2 = 40;
	k2 = 4096;

//	k2 = 8192;

	//	k2 = _kVec;

//	uint maxBins = 4096;

	uint maxBins = 4 * 8192;
//	uint maxBins = 2048;

	uint* nBins;
	uint* bins;

	cudaMalloc(&nBins, _QN * 1 * sizeof(uint));
	cudaMalloc(&bins, _QN * maxBins * 1 * sizeof(uint));

	if (!nBins || !bins) {
		std::cout << "Bins: did not get memory" << std::endl;
		exit(1);
	}

	uint maxVec = _kVec;
	float* bestDist;
	uint* bestIdx;
	cudaMalloc(&bestDist, _QN * maxVec * sizeof(float));
	cudaMalloc(&bestIdx, _QN * maxVec * sizeof(uint));

	getKBestAssignment(assignd, d_multiCodeBook, _Q, d_nClusters, _QN, k1);

//	getLineAssignment(queryL1, _Q, _QN);

	getKBestAssignment2(assignVal, assignIdx, d_multiCodeBook2, _Q,
			d_nClusters2, _QN, assignd, d_nClusters, k1);

//		outputVec("assignVal: ", assignVal, 200);
//		outputVecUint("assignIdx: ", assignIdx, 200);

	std::cout << "done assignements " << std::endl;

#if 1
	getBIGBins(bins, nBins, assignVal, assignIdx, _QN, k1, k2, maxBins);

//	outputVecUint("Bins", bins, 2000);

	std::cout << "done bins " << std::endl;

//		if (maxVec <= 1024)
//		getKBestVectors(bestDist, bestIdx, bins, nBins, maxBins, _Q, _QN,
//					maxVec);
//		else
//			getKBestVectorsLarge(bestDist, bestIdx, bins, nBins, maxBins, _Q,
//					_QN, maxVec);

	rerankBIGKBestVectors(_resIdx, bestDist, bestIdx, queryL1, bins, nBins,
			maxBins, _Q, _QN, maxVec, _hLines);
#endif
	std::cout << "done vectors " << _QN << std::endl;

//	outputVecUint("BestIdx: ", bestIdx, 1000);

	cudaMemcpy(&_resIdx[0], bestIdx, _QN * maxVec * sizeof(uint),
			cudaMemcpyDeviceToHost);
	cudaMemcpy(&_resDist[0], bestDist, _QN * maxVec * sizeof(float),
			cudaMemcpyDeviceToHost);

	cudaFree(bestIdx);
	cudaFree(bestDist);
	cudaFree(bins);
	cudaFree(nBins);
	cudaFree(assignIdx);
	cudaFree(assignVal);
	cudaFree(assignd);

}

void PerturbationProTree::queryBIGKNNRerank2(vector<uint>& _resIdx,
		vector<float>& _resDist, const float* _Q, uint _QN, uint _kVec,
		const float* _hLines) {

	_resIdx.resize(_QN * _kVec);
	_resDist.resize(_QN * _kVec);

//	uint k1 = 32;
	uint k1 = 8;
	k1 = 16;

//	d_lineParts = 16; // TODO !!!!

	if (!d_codeBookDistL1L2)
		computeCBL1L1Dist(d_lineParts);

//	prepareDistSequence(d_nClusters2 * k1, d_p);

//	prepareDistSequence(14, d_p);

	uint* assignd;
	float *assignVal;
	uint *assignIdx;

	cudaMalloc(&assignd, 1 * k1 * d_p * _QN * sizeof(uint));

	float *queryL1;
	cudaMalloc(&queryL1, d_nClusters * d_lineParts * _QN * sizeof(uint));

	cudaMalloc(&assignVal,
			_QN * 1 * d_p * k1 * d_nClusters2 * sizeof(float));
	cudaMalloc(&assignIdx,
			_QN * 1 * d_p * k1 * d_nClusters2 * sizeof(uint));

//	uint k2 = 40;
//	k2 = 4096;

//	k2 = 8192;

	//	k2 = _kVec;

//	uint maxBins = 4096;

	uint maxBins = 64 * 8192;
//	uint maxBins = 2048;

//	uint maxBins = 16;

	uint* nBins;
	uint* bins;

	cudaMalloc(&nBins, _QN * 1 * sizeof(uint));
	cudaMalloc(&bins, _QN * maxBins * 1 * sizeof(uint));

	if (!nBins || !bins) {
		std::cout << "Bins: did not get memory" << std::endl;
		exit(1);
	}

	uint maxVec = _kVec;
	float* bestDist;
	uint* bestIdx;
	cudaMalloc(&bestDist, _QN * maxVec * sizeof(float));
	cudaMalloc(&bestIdx, _QN * maxVec * sizeof(uint));

	getKBestAssignment(assignd, d_multiCodeBook, _Q, d_nClusters, _QN, k1);

	getLineAssignment(queryL1, _Q, _QN);

	getKBestAssignment2(assignVal, assignIdx, d_multiCodeBook2, _Q,
			d_nClusters2, _QN, assignd, d_nClusters, k1);

//		outputVec("assignVal: ", assignVal, 200);
//		outputVecUint("assignIdx: ", assignIdx, 200);

	std::cout << "done assignements " << std::endl;

//
	getBIGBins2D(bins, nBins, assignVal, assignIdx, _QN, k1, _kVec, maxBins);

	rerankBIGKBestVectors2(bestDist, bestIdx, queryL1, bins, nBins, maxBins, _Q,
			_QN, maxVec, _hLines);

	outputVecUint( "bestIdx: ", bestIdx, 4096);
	outputVec( "bestDist: ", bestDist, 4096 );

	std::cout << "done vectors " << _QN << std::endl;

//	outputVecUint("BestIdx: ", bestIdx, _QN * maxVec);

	countZeros("bestIdx: ", bestIdx, _QN * maxVec);

	cudaMemcpy(&_resIdx[0], bestIdx, _QN * maxVec * sizeof(uint),
			cudaMemcpyDeviceToHost);
	cudaMemcpy(&_resDist[0], bestDist, _QN * maxVec * sizeof(float),
			cudaMemcpyDeviceToHost);

	cudaFree(bestIdx);
	cudaFree(bestDist);
	cudaFree(bins);
	cudaFree(nBins);
	cudaFree(assignIdx);
	cudaFree(assignVal);
	cudaFree(assignd);

}

void PerturbationProTree::queryBIGKNNRerankPerfect(vector<uint>& _resIdx,
		vector<float>& _resDist, const float* _Q, uint _QN, uint _kVec,
		const float* _hLines) {

	_resIdx.resize(_QN * _kVec);
	_resDist.resize(_QN * _kVec);

//	uint k1 = 32;
	uint k1 = 8;
	k1 = 16;

	uint* assignd;
	float *assignVal;
	uint *assignIdx;

	cudaMalloc(&assignd, 1 * k1 * d_p * _QN * sizeof(uint));

//	float *queryL1;
//	cudaMalloc(&queryL1, d_nClusters * d_lineParts * _QN * sizeof(uint));

	cudaMalloc(&assignVal,
			_QN * 1 * d_p * k1 * d_nClusters2 * sizeof(float));
	cudaMalloc(&assignIdx,
			_QN * 1 * d_p * k1 * d_nClusters2 * sizeof(uint));

//	uint k2 = 40;
//	k2 = 4096;

//	k2 = 8192;

	//	k2 = _kVec;

//	uint maxBins = 4096;

	uint maxBins = 64 * 8192;
//	uint maxBins = 2048;

//	uint maxBins = 16;

	uint* nBins;
	uint* bins;

	cudaMalloc(&nBins, _QN * 1 * sizeof(uint));
	cudaMalloc(&bins, _QN * maxBins * 1 * sizeof(uint));

	if (!nBins || !bins) {
		std::cout << "Bins: did not get memory" << std::endl;
		exit(1);
	}

	uint maxVec = _kVec;
	float* bestDist;
	uint* bestIdx;
	cudaMalloc(&bestDist, _QN * maxVec * sizeof(float));
	cudaMalloc(&bestIdx, _QN * maxVec * sizeof(uint));

	getKBestAssignment(assignd, d_multiCodeBook, _Q, d_nClusters, _QN, k1);

//	getLineAssignment(queryL1, _Q, _QN);

	getKBestAssignment2(assignVal, assignIdx, d_multiCodeBook2, _Q,
			d_nClusters2, _QN, assignd, d_nClusters, k1);

	std::cout << "done assignements " << std::endl;

	getBIGBins2D(bins, nBins, assignVal, assignIdx, _QN, k1, _kVec, maxBins);

//	rerankBIGKBestVectors2(bestDist, bestIdx, queryL1, bins, nBins, maxBins, _Q,
//			_QN, maxVec, _hLines);

	rerankBIGKBestVectorsPerfect(bestDist, bestIdx, bins, nBins, maxBins, _Q,
			_QN, maxVec, _hLines);

	std::cout << "done vectors " << _QN << std::endl;

//	outputVecUint("BestIdx: ", bestIdx, _QN * maxVec);

	countZeros("bestIdx: ", bestIdx, _QN * maxVec);

	cudaMemcpy(&_resIdx[0], bestIdx, _QN * maxVec * sizeof(uint),
			cudaMemcpyDeviceToHost);
	cudaMemcpy(&_resDist[0], bestDist, _QN * maxVec * sizeof(float),
			cudaMemcpyDeviceToHost);

	cudaFree(bestIdx);
	cudaFree(bestDist);
	cudaFree(bins);
	cudaFree(nBins);
	cudaFree(assignIdx);
	cudaFree(assignVal);
	cudaFree(assignd);

}

#if 0

__global__ void locateIDs(uint _baseNum, const uint* _prefix,
		const uint* _counts, const uint* _dbIdx, uint _N, const uint* _gtIdx,
		uint* _gtPos) {

	extern __shared__ float shm[];

	float* shmIter = shm;

	uint* pref = (uint*) shmIter;
	shmIter += blockDim.x;

	uint* pId = (uint*) shmIter;
	shmIter += blockDim.x;

	// load corresponding queries;
	for (uint qId = blockIdx.y * blockDim.x + threadIdx.x; qId < _N;
			qId += gridDim.y * blockDim.x) {

		uint q = _gtIdx[qId];

		for (uint bId = blockIdx.x * blockDim.x + threadIdx.x; bId < _baseNum;
				bid += gridDim.x * blockDim.x) {
			__syncthreads();

			pref[threadIdx.x] = _prefix[bId];
			pId[threadIdx.x] = bId;

			__syncthreads();

			for (int i = 0; i < blockDim.x; i++) {
				if (pref[i] == q)
				_gtPos[qId] = pId[i];
			}
		}
	}
}

void PerturbationProTree::locateIDs(uint _baseNum, const vector<uint>& _gt,
		vector<uint>& _gtBins) {

	uint N = _gt.size();

	_gtBins.resize(N);

	uint* gtIdx = NULL;
	uint* gtBins = NULL;

	cudaMalloc(&gtIdx, N * sizeof(uint));
	cudaMalloc(&gtBins, N * sizeof(uint));

	if ((gtIdx == NULL) || (gtBins == NULL)) {
		std::cout << "locateIDs: did not get memory!" << std::endl;
		return;
	}

	uint nThreads = 1024;
	dim3 block(nThreads, 1, 1);
	dim3 grid(1024, N / nThreads + 1, 1);

	uint shmSize = (2 * nThreads) * sizeof(float);

	locateIDs<<<grid, block, shmSize>>>(_baseNum, d_prefix, d_dbIdx, N, gtIdx,
			gtBins);

	checkCudaErrors(cudaDeviceSynchronize());
	cudaMemcpy(&(_gtBins[0]), gtBins, N * sizeof(uint), cudaMemcpyDeviceToHost);

	cudaFree(gtBins);
	cudaFree(gtIdx);

}

#endif

}
/* namespace */
