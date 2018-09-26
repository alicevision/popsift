#ifndef NEARESTNEIGHBOR_PROQUANTIZATION_C
#define NEARESTNEIGHBOR_PROQUANTIZATION_C

#include "ProQuantization.hh"

#define OUTPUT
#include "helper.hh"

#include <algorithm>
#include <assert.h>
#include <fstream>

#include "sortingNetworks_common.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>


using namespace std;

namespace pqt {

/** default constructor */

ProQuantization::ProQuantization(uint _dim, uint _p) :
		d_dim(_dim), d_codeBook(NULL), d_p(_p), d_vl(_dim / _p), d_nClusters(0) {

	cout << "dim, p, vl: " << d_dim << " " << d_p << " " << d_vl << endl;
}

ProQuantization::~ProQuantization() {
	if (d_codeBook)
		cudaFree(d_codeBook);
}

void ProQuantization::writeCodebookToFile(const std::string& _name) {

	std::ofstream f(_name.c_str(), std::ofstream::out | std::ofstream::binary);

	f << d_dim << endl;
	f << d_p << endl;
	f << d_nClusters << endl;

	float * cb1Host = new float[d_nClusters * d_dim];

	cudaMemcpy(cb1Host, d_codeBook, d_nClusters * d_dim * sizeof(float),
			cudaMemcpyDeviceToHost);

	f.write((char*) cb1Host, d_nClusters * d_dim * sizeof(float));

	f.close();

	delete[] cb1Host;

	cout << "written " << _name << endl;

}

void ProQuantization::readCodebookFromFile(const std::string& _name) {

	ifstream f(_name.c_str(), std::ofstream::in | std::ofstream::binary);

	f >> d_dim;
	f >> d_p;
	f >> d_nClusters;

	f.ignore(1);

	cout << d_dim << endl;
	cout << d_p << endl;
	cout << d_nClusters << endl;

	d_vl = d_dim / d_p;

	if (d_codeBook)
		cudaFree(d_codeBook);

	float * cb1Host = new float[d_nClusters * d_dim];

	cudaMalloc(&d_codeBook, d_nClusters * d_dim * sizeof(float));

	f.read((char*) cb1Host, d_nClusters * d_dim * sizeof(float));

	cudaMemcpy(d_codeBook, cb1Host, d_nClusters * d_dim * sizeof(float),
			cudaMemcpyHostToDevice);

	checkCudaErrors(cudaDeviceSynchronize());

	f.close();

	delete[] cb1Host;

	cout << "read " << _name << endl;

}

// each block is responsible for one vector, blockDim.x should be _dim
// requires _dim * float shared memory
// looping multiple times to process all B vectors
// _vl is the length of the _p vector segments (should be 2^n)
__global__ void calcDistKernel(float *_res, const float* _A, const float* _B,
		uint _Arows, uint _Brows, uint _dim, uint _p, uint _vl) {

	extern __shared__ float shm[];

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		// load vector
		float b = _B[iter * _dim + threadIdx.x];

		// loop over all vectors of A
		for (int a = 0; a < _Arows; a++) {
			shm[threadIdx.x] = sqr(b - _A[a * _dim + threadIdx.x]);

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
				_res[(iter * _Arows + a) * _p + threadIdx.x] = shm[threadIdx.x
						* _vl];
			}

			__syncthreads();
		}
	}
}

void ProQuantization::calcDist(float *_res, const float* _A, const float* _B,
		uint _Arows, uint _Brows) const {

	dim3 block(d_dim, 1, 1);
	//dim3 grid(idiv(_Brows, MAX_BLOCKS), 1, 1);
	dim3 grid((_Brows < 1024) ? _Brows : 1024, 1, 1);

//	uint shm = d_dim * (_Arows + 1)* sizeof(float);
	uint shm = d_dim * sizeof(float);
	calcDistKernel<<<grid, block, shm>>>(_res, _A, _B, _Arows, _Brows, d_dim,
			d_p, d_vl);

	checkCudaErrors(cudaDeviceSynchronize());
}

void ProQuantization::calcDist(float *_res, const float* _A, const float* _B,
		uint _Arows, uint _Brows, uint _dim, uint _p) const {

	dim3 block(_dim, 1, 1);
	dim3 grid((_Brows < 1024) ? _Brows : 1024, 1, 1);

	uint shm = _dim * sizeof(float);
	calcDistKernel<<<grid, block, shm>>>(_res, _A, _B, _Arows, _Brows, _dim,
			_p, _dim / _p);

	checkCudaErrors(cudaDeviceSynchronize());
}


void ProQuantization::parallelSort(float *_vec, uint _lower, uint _upper) {

	thrust::device_ptr<float> trVec(_vec);

	thrust::sort(trVec + _lower, trVec + _upper);

}


void ProQuantization::testDist(float *_B, uint _Brows) {

	float *A, *Ad;
	float *resd;

	uint Arows = 8;

	A = new float[Arows * d_dim];

	for (int i = 0; i < Arows; i++) {
		for (int p = 0; p < d_p; p++) {
			for (int v = 0; v < d_vl; v++) {
				A[i * d_dim + p * d_vl + v] = p;
			}
		}
	}

	cudaMalloc(&Ad, Arows * d_dim * sizeof(float));
	cudaMalloc(&resd, d_p * _Brows * Arows * sizeof(float));

	cudaMemset(_B, 0, _Brows * d_dim * sizeof(float));

	cout << "starting" << endl;
	cout.flush();

	cudaMemcpy(Ad, A, Arows * d_dim * sizeof(float), cudaMemcpyHostToDevice);

	calcDist(resd, Ad, _B, Arows, _Brows);

	outputVec("Res:", resd, 10 * d_p);

	cudaFree(resd);
	cudaFree(Ad);

	delete[] A;

}

// each block is responsible for one vector, blockDim.x should be _dim
// requires _dim * float shared memory
// looping multiple times to process all B vectors
// _vl is the length of the _p vector segments (should be 2^n)
__global__ void assignClusterKernel(uint *_assign, const float* _A,
		const float* _B, uint _Arows, uint _Brows, uint _dim, uint _p,
		uint _vl) {

	extern __shared__ float shm[];

	float minVal;
	uint minIdx;

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		if (threadIdx.x < _p) {
			minVal = 1000000.;
			minIdx = 0;
		}

		// load vector
		float b = _B[iter * _dim + threadIdx.x];

		// loop over all vectors of A
		for (int a = 0; a < _Arows; a++) {
			shm[threadIdx.x] = sqr(b - _A[a * _dim + threadIdx.x]);

			// compute the sum of differences for the vector segments
			// i.e. all _p segments in parallel
			for (uint stride = _vl >> 1; stride > 0; stride >>= 1) {
				__syncthreads();
//				if ((threadIdx.x % _vl) < stride) {
//					shm[threadIdx.x] += shm[threadIdx.x + stride];
//				}
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
			_assign[iter * _p + threadIdx.x] = minIdx;
			// if (iter == 0) printf("val: %f \n ", minVal );
		}

	} // iter
}

void ProQuantization::getAssignment(uint *_assign, const float* _A,
		const float* _B, uint _Arows, uint _Brows) const {

	dim3 block(d_dim, 1, 1);
	dim3 grid((_Brows < 1024) ? _Brows : 1024, 1, 1);

	uint shm = d_dim * sizeof(float);
	assignClusterKernel<<<grid, block, shm>>>(_assign, _A, _B, _Arows, _Brows,
			d_dim, d_p, d_vl);

	checkCudaErrors(cudaDeviceSynchronize());
}

void ProQuantization::testAssignment(float *_B, uint _Brows) {

	float *A, *Ad;
	uint *assignd;

	uint Arows = 8;

	cudaMalloc(&Ad, Arows * d_dim * sizeof(float));
	cudaMalloc(&assignd, d_p * _Brows * sizeof(uint));

	A = new float[Arows * d_dim];

	for (int i = 0; i < Arows; i++) {
		for (int p = 0; p < d_p; p++) {
			for (int v = 0; v < d_vl; v++) {
				A[i * d_dim + p * d_vl + v] = p + i * 10;
			}
		}
	}

	cudaMemcpy(Ad, A, Arows * d_dim * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemset(_B, 0, _Brows * d_dim * sizeof(float));

	for (int i = 0; i < Arows; i++) {
		for (int p = 0; p < d_p; p++) {
			for (int v = 0; v < d_vl; v++) {
				A[i * d_dim + p * d_vl + v] = p * 10 + i;
			}
		}
	}
	cudaMemcpy(_B, A, Arows * d_dim * sizeof(float), cudaMemcpyHostToDevice);

	cout << "starting Assign" << endl;
	cout.flush();

	getAssignment(assignd, Ad, _B, Arows, _Brows);

	uint* assign = new uint[10 * d_p];

	cudaMemcpy(assign, assignd, 10 * d_p * sizeof(uint),
			cudaMemcpyDeviceToHost);

	cout << "assigned" << endl;
	for (int i = 0; i < 10 * d_p; i++) {
		cout << assign[i] << " ";
	}
	cout << endl;

	cudaFree(assignd);
	cudaFree(Ad);

	delete[] assign;
	delete[] A;

}

template<class T>
__device__ void swap(T& _a, T&_b) {
	T h = _a;
	_a = _b;
	_b = h;
}

// parallel bitonic sort
__device__ void bitonic(volatile float _val[], volatile uint _idx[], uint _N) {

	for (int k = 2; k <= _N; k <<= 1) {

		// bitonic merge
		for (int j = k / 2; j > 0; j /= 2) {
			int ixj = threadIdx.x ^ j;  // XOR
			if ((ixj > threadIdx.x) && (ixj < _N)) {
				if ((threadIdx.x & k) == 0) // ascending - descending
						{
					if (_val[threadIdx.x] > _val[ixj]) {

						swap(_val[threadIdx.x], _val[ixj]);
						swap(_idx[threadIdx.x], _idx[ixj]);
					}
				} else {
					if (_val[threadIdx.x] < _val[ixj]) {

						swap(_val[threadIdx.x], _val[ixj]);
						swap(_idx[threadIdx.x], _idx[ixj]);
					}

				}
			}
			__syncthreads();
		}
	}
}

// each block is responsible for one vector, blockDim.x should be _dim
// requires _dim * float shared memory
// looping multiple times to process all B vectors
// _vl is the length of the _p vector segments (should be 2^n)

// 			 k0			k1			k2		k3		k4
// output    p0,p1,..   p0,p1,..	..
__global__ void assignKBestClusterKernel(uint *_assign, const float* _A,
		const float* _B, uint _Arows, uint _Brows, uint _dim, uint _p, uint _vl,
		uint _k, uint _NP2) {

	extern __shared__ float shm[];

	float* shmIter = shm;
	shmIter += _NP2;
	uint* shmIdx = (uint*) shmIter;
	shmIter += _NP2;

	uint offs = (2 * _NP2 > _dim) ? 2 * _NP2 : _dim;

	shmIter = shm + offs;

	float* val = shmIter;
	shmIter += _p * _Arows;
	uint* idx = (uint*) (shmIter);
//
//	uint* shmIdx = (uint*) (shm + _NP2);

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		__syncthreads();
		// load vector
		float b;
		if (threadIdx.x < _dim)
			b = _B[iter * _dim + threadIdx.x];

		// loop over all vectors of A
		for (int a = 0; a < _Arows; a++) {
			if (threadIdx.x < _dim)
				shm[threadIdx.x] = sqr(b - _A[a * _dim + threadIdx.x]);

			// compute the sum of differences for the vector segments
			// i.e. all _p segments in parallel
			for (uint stride = _vl >> 1; stride > 0; stride >>= 1) {
				__syncthreads();
//				if ((threadIdx.x % _vl) < stride) {
//					shm[threadIdx.x] += shm[threadIdx.x + stride];
//				}
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

//		if (threadIdx.x == 0) {
//			printf("before sorted: ");
//			for (int k = 0; k < _k; k++)
//				printf( "%f/%d ", val[k], idx[k]);
//			printf("\n");
//		}

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

			bitonic(shm, shmIdx, _NP2);

			if (threadIdx.x < _Arows) {
				val[threadIdx.x + i * _Arows] = shm[threadIdx.x];
				idx[threadIdx.x + i * _Arows] = shmIdx[threadIdx.x];
			}

			__syncthreads();

		}

//		if (threadIdx.x == 0) {
//			printf("sorted: ");
//			for (int k = 0; k < _k; k++)
//				printf( "%f/%d ", val[k], idx[k]);
//			printf("\n");
//		}

		// write out decision
		for (int k = 0; k < _k; k++) {
			if (threadIdx.x < _p) {
				_assign[iter * _k * _p + k * _p + threadIdx.x] = idx[threadIdx.x
						* _Arows + k];

			}
		}

	} // iter
}

__global__ void assignKBestClusterKernelSingleP(uint *_assign, const float* _A,
		const float* _B, uint _Arows, uint _Brows, uint _dim, uint _p, uint _vl,
		uint _k, uint _NP2) {

	extern __shared__ float shm[];

//	float* val = shm + blockDim.x;
//	uint* idx = (uint*) (val + _Arows);

	uint* shmIdx = (uint*) (shm + _NP2);

	float val[16];

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		__syncthreads();

		// load vector
		float b;
		if (threadIdx.x < _dim)
			b = _B[iter * _dim + threadIdx.x];

		// loop over all vectors of A
		for (int a = 0; a < _Arows; a++) {

			if (threadIdx.x < _dim)
				shm[threadIdx.x] = sqr(b - _A[a * _dim + threadIdx.x]);

#if 0
			// compute the sum of differences for the vector segments
			// i.e. all _p segments in parallel
			for (uint stride = _vl >> 1; stride > 0; stride >>= 1) {
				__syncthreads();
//				if ((threadIdx.x % _vl) < stride) {
//					shm[threadIdx.x] += shm[threadIdx.x + stride];
//				}
				if ((threadIdx.x) < _p * stride) {
					uint p = threadIdx.x / stride * _vl;
					uint bias = threadIdx.x % stride;
					shm[p + bias] += shm[p + bias + stride];
				}
			}
#endif
			__syncthreads();

			// store the result
			if (threadIdx.x == a) {

//				for (int p = 0; p < _p; p++)
//					val[p] = shm[p * _vl];

				for (int p = 0; p < _p; p++) {
					val[p] = 0;
					for (int i = 0; i < _vl; i++) {
						val[p] += shm[p * _vl + i];
					}
				}

//				if (a < 10) {
//					for (int p = 0; p < _p; p++)
//						printf( "val %d %d:  %f \n", a, p, val[p]);
//				}
// 507   487   968    12   662   961  1019
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

			bitonic(shm, shmIdx, _NP2);

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
				_assign[iter * _k * _p + k * _p + threadIdx.x] =
						shmIdx[threadIdx.x];

			}
			__syncthreads();
		}

	} // iter
}

void ProQuantization::getKBestAssignment(uint *_assign, const float* _A,
		const float* _B, uint _Arows, uint _Brows, uint _k) const {

	if (_Arows <= 1024) {

		uint NP2 = log2(_Arows);

//	cout << "NP2 " << NP2 << endl;
//	cout << "Arows " << _Arows <<"  Brows " << _Brows << endl;

//	assert(d_dim > (2 * NP2));

		uint nThreads = (NP2 > d_dim) ? NP2 : d_dim;

		dim3 block(nThreads, 1, 1);
		dim3 grid((_Brows < 1024) ? _Brows : 1024, 1, 1);

		uint shm = (2 * nThreads + 2 * _Arows * d_p) * sizeof(float);

		if (shm > 32000) {
			shm = (nThreads + 2 * _Arows) * sizeof(float);
			//	cout << "kbest single p : shm " << shm << endl;
			assignKBestClusterKernelSingleP<<<grid, block, shm>>>(_assign, _A,
					_B, _Arows, _Brows, d_dim, d_p, d_vl, _k, NP2);

		} else {

			// cout << "kbest: shm " << shm << endl;
			assignKBestClusterKernel<<<grid, block, shm>>>(_assign, _A, _B,
					_Arows, _Brows, d_dim, d_p, d_vl, _k, NP2);
		}
	} else {

		if (d_p > 1) {
			cout << "not implemented";
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

#if 0
void ProQuantization::testKBestAssignment(float *_B, uint _Brows) {

	float *A, *Ad;
	uint *assignd;

	uint Arows = 8;

	uint k = 3;

	cudaMalloc(&Ad, Arows * d_dim * sizeof(float));
	cudaMalloc(&assignd, k * d_p * _Brows * sizeof(uint));

	A = new float[Arows * d_dim];

	for (int i = 0; i < Arows; i++) {
		for (int p = 0; p < d_p; p++) {
			for (int v = 0; v < d_vl; v++) {
				A[i * d_dim + p * d_vl + v] = p + i * 10;
			}
		}
	}

	cudaMemcpy(Ad, A, Arows * d_dim * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemset(_B, 0, _Brows * d_dim * sizeof(float));

	for (int i = 0; i < Arows; i++) {
		for (int p = 0; p < d_p; p++) {
			for (int v = 0; v < d_vl; v++) {
				A[i * d_dim + p * d_vl + v] = p * 10 + i;
			}
		}
	}
	cudaMemcpy(_B, A, Arows * d_dim * sizeof(float), cudaMemcpyHostToDevice);

	cout << "starting Assign" << endl;
	cout.flush();

	getKBestAssignment(assignd, Ad, _B, Arows, _Brows, k);

	uint* assign = new uint[10 * k * d_p];

	cudaMemcpy(assign, assignd, 10 * k * d_p * sizeof(uint),
			cudaMemcpyDeviceToHost);

	cout << "assigned" << endl;
	for (int i = 0; i < 10 * k * d_p; i++) {
		cout << assign[i] << " ";
	}
	cout << endl;

	cudaFree(assignd);
	cudaFree(Ad);

	delete[] assign;
	delete[] A;

	cout << "done with free" << endl;

}
#endif

void ProQuantization::testKBestAssignment(float* _Qh, float *_Qd, uint _QN) {

	uint numSamples = 100;

	uint* assignd;
	uint* refAssignh;
	cudaMalloc(&assignd, d_nClusters * d_p * sizeof(uint));
	refAssignh = new uint[d_nClusters * d_p];
	float* codebookh = new float[d_nClusters * d_dim];

	cudaMemcpy(codebookh, d_codeBook, d_nClusters * d_dim * sizeof(float),
			cudaMemcpyDeviceToHost);

	for (int j = 0; j < numSamples; j++) {
		getKBestAssignment(assignd, d_codeBook, _Qd + j * d_dim, d_nClusters, 1,
				d_nClusters);
//		outputVecUint("assignd: ", assignd, d_p);

		cudaMemcpy(refAssignh, assignd, d_p * d_nClusters * sizeof(uint),
				cudaMemcpyDeviceToHost);

		for (int p = 0; p < d_p; p++) {

			vector<pair<float, uint> > ddd;
			ddd.clear();
			ddd.resize(d_nClusters);
			for (int c = 0; c < d_nClusters; c++) {
				float dist = 0.;
				for (int a = 0; a < d_vl; a++) {
					float v = _Qh[j * d_dim + p * d_vl + a]
							- codebookh[c * d_dim + p * d_vl + a];
					dist += v * v;
				}
				ddd[c] = pair<float, uint>(dist, c);
			}

			sort(ddd.begin(), ddd.end());

			float passed = 0.;

			for (int c = 0; c < d_nClusters; c++) {
				if (ddd[c].second != refAssignh[c * d_p + p])
					passed += 1.;
			}

			if (passed > 0.20 * d_nClusters) {
				cout << "host:   " << endl;
				for (int c = 0; c < d_nClusters; c++) {
					cout << ddd[c].second << "  ";
				}
				cout << endl;

				cout << "device: " << endl;
				for (int c = 0; c < d_nClusters; c++) {
					cout << refAssignh[c * d_p + p] << "  ";
				}
				cout << endl;
				cout
						<< "testKBestAssignment not passed!!!!!!!!!!!!!!!!!!!!!!!!!"
						<< endl;
				cout << "found " << passed << " differences " << endl;
				exit(1);
			}

		}
	}

	delete[] refAssignh;
	delete[] codebookh;
	cudaFree(assignd);

	cout << "testKBestAssignment passed." << endl;
}

// each block is responsible for some vectors of _B and one segment
// blockIdx.y indicates the segment
// the kernel locally accumulates the segments for all cluster centers
// blockIdx.z indicates which centers this block is responsible for
__global__ void avgClusterKernel(float* _codebook, float * _count,
		uint *_retirementCount, const float * _B, uint _Brows, uint _nClusters,
		uint _nClustersPerBlock, uint _dim, uint _vl, uint _p, uint* _assignd) {

//	__shared__ bool amLast;
	extern __shared__ float shm[];

	bool& amLast(*(bool*) (shm + _nClustersPerBlock * _vl));

	uint clusterOffset = blockIdx.z * _nClustersPerBlock;
	uint count = 0;
//	for (int i = 0; i < _nClustersPerBlock; i++) {
//		shm[i * _vl + threadIdx.x] = 0.;
//	}

	for (int i = threadIdx.x; i < _nClustersPerBlock * _vl; i += blockDim.x)
		shm[i] = 0.;

	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		__syncthreads();
		int bid = _assignd[iter * _p + blockIdx.y] - clusterOffset;

//		if ((blockIdx.y == 2) && (threadIdx.x == 0)) printf("%d,%d  ", blockIdx.y, bid);

		if ((bid < 0) || (bid >= _nClustersPerBlock))
			continue;

		if (bid == threadIdx.x) {
			count++;
		}

		// load vector segment
		float b = _B[iter * _dim + blockIdx.y * _vl + threadIdx.x];

		shm[bid * _vl + threadIdx.x] += b;

	}

	__syncthreads();

//	if ((blockIdx.x == 0) && (threadIdx.x == 4)) {
//			printf("%f %f %f %f %f \n" , shm[0], shm[1], shm[2], shm[3], shm[4]);
//	}

	// store the result;
	for (int i = 0; i < _nClustersPerBlock; i++) {
		atomicAdd(
				_codebook + (clusterOffset + i) * _dim + blockIdx.y * _vl
						+ threadIdx.x, shm[i * _vl + threadIdx.x]);
	}

	// write the counts
	if (threadIdx.x < _nClustersPerBlock) {
		atomicAdd(
				_count + blockIdx.y * _nClusters + clusterOffset + threadIdx.x,
				count);

		//	if (blockIdx.y == 1) printf( "%d ", count);
	}

//	__threadfence();
	__syncthreads();

	if (threadIdx.x == 0) {
		uint ticket = atomicInc(
				_retirementCount + blockIdx.y * gridDim.z + blockIdx.z,
				gridDim.x);
		// If the ticket ID is equal to the number of blocks, we are the last block!
		amLast = (ticket == gridDim.x - 1);

	}
	__syncthreads();

#if 1
	// the last block is responsible for dividing by the number of vectors added to this center
	if (amLast) {
		for (int i = 0; i < _nClustersPerBlock; i++) {
			float count = _count[blockIdx.y * _nClusters + clusterOffset + i];
			count = (count == 0.) ? 1. : count;
			_codebook[(clusterOffset + i) * _dim + blockIdx.y * _vl
					+ threadIdx.x] /= count;

//			if (i== 0)
//			printf( "idx,count: %d %f %f \n", threadIdx.x, count, _codebook[(clusterOffset + i) * _dim + blockIdx.y * _vl
//			                                            					+ threadIdx.x]);
		}

		// reset retirement count for next iteration
		if (threadIdx.x == 0) {
			_retirementCount[blockIdx.y * gridDim.z + blockIdx.z] = 0;
		}
	}
#endif

}

void ProQuantization::getClusterAverage(float *_codebook, float *_count,
		uint* _retirementCount, uint _nClusters, const float *_B, uint _Brows,
		uint *_assign) {

	dim3 block(d_vl, 1, 1);

	uint nblocks = idiv(_Brows, 32);

	uint nClustersPerBlock = (_nClusters > 8) ? 8 : _nClusters;
	nClustersPerBlock =
			(nClustersPerBlock > block.x) ? block.x : nClustersPerBlock;

	dim3 grid((nblocks > 1024) ? 1024 : nblocks, d_p,
			idiv(_nClusters, nClustersPerBlock));

	uint shmSize = (nClustersPerBlock * d_vl + 2) * sizeof(float);

//	uint shmSize = (d_dim * 2 + 2) * sizeof(float);

//	cout << "block: " << block.x << endl;
//	cout << "nClustersPerBlock" << nClustersPerBlock;
//	cout << "grid: " << grid.x << " " << grid.y << " " << grid.z << endl;
//	cout << "shm: " << shmSize << endl;

	//uint shmSize = (d_dim + 2 ) * sizeof(float);

	avgClusterKernel<<<grid, block, shmSize>>>(_codebook, _count,
			_retirementCount, _B, _Brows, _nClusters, nClustersPerBlock, d_dim,
			d_vl, d_p, _assign);

	checkCudaErrors(cudaDeviceSynchronize());
}

void ProQuantization::testAvg(float *_B, uint _Brows) {

	uint *assignd;
	float *codebook;
	float *count;
	uint *retirementCount;

	uint Arows = 4;

	cudaMalloc(&codebook, Arows * d_dim * sizeof(float));

	cudaMemset(codebook, 0, Arows * d_dim * sizeof(float));

	cudaMalloc(&assignd, d_p * _Brows * sizeof(uint));

	uint *assign = new uint[d_p * _Brows];

	cout << "starting Avg" << endl;
	cout.flush();

	cudaMalloc(&count, Arows * d_p * sizeof(float));
	cudaMalloc(&retirementCount, Arows * d_p * sizeof(uint));
	cudaMemset(retirementCount, 0, Arows * d_p * sizeof(uint));
	cudaMemset(count, 0, Arows * d_p * sizeof(float));

	float *b = new float[d_dim * _Brows];

	for (int i = 0; i < d_dim * _Brows; i++)
		b[i] = 1.;

	for (int i = 0; i < d_p * _Brows; i++) {
		assign[i] = (i * 72) % Arows;
	}

	for (int i = 0; i < _Brows; i++) {
		for (int p = 0; p < d_p; p++) {
			for (int v = 0; v < d_vl; v++) {
				b[i * d_dim + p * d_vl + v] = 1.;
			}
			assign[i * d_p + p] = ((p + i) % Arows);
		}
	}

	cudaMemcpy(_B, b, d_dim * _Brows * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(assignd, assign, d_p * _Brows * sizeof(uint),
			cudaMemcpyHostToDevice);

	getClusterAverage(codebook, count, retirementCount, Arows, _B, _Brows,
			assignd);

	cout << "averaged" << endl;
	outputVec("Codebook", codebook, 4 * d_dim);

	cudaFree(retirementCount);
	cudaFree(count);
	cudaFree(assignd);
	cudaFree(codebook);

	delete[] assign;

}

__global__ void splitVectorKernelPro(float* _codeBook, uint _dim,
		uint _nClusters, float _epsilon) {

	uint idx = blockIdx.x * _dim + threadIdx.x;
	float orig = _codeBook[idx];

	_codeBook[idx] = orig * (1. + _epsilon);

	idx += _nClusters * _dim;

	_codeBook[idx] = orig * (1. - _epsilon);

}


void ProQuantization::splitCodeBook(uint &_nClusters, float _epsilon) {

	dim3 block(d_dim, 1, 1);

	dim3 grid(_nClusters, 1, 1);

	splitVectorKernelPro<<<grid, block>>>(d_codeBook, d_dim, _nClusters,
			_epsilon);


	_nClusters *= 2;

}

void ProQuantization::createCodeBook(uint _k, const float* _A, uint _N) {

	uint *assign = new uint[_N * d_p];
	uint *old_assign = new uint[_N * d_p];

	uint *assignd;
	float* countd;
	uint* retirementCountd;
	float* distd;
	float* dist;
	float* oldCodeBook;

	cudaMalloc(&assignd, _N * d_p * sizeof(uint));

	if (d_codeBook != NULL)
		cudaFree(d_codeBook);

	cudaMalloc(&d_codeBook, _k * d_dim * sizeof(float));
	cudaMalloc(&countd, _k * d_p * sizeof(float));
	cudaMalloc(&retirementCountd, _k * d_p * sizeof(uint));
	cudaMalloc(&distd, _k * d_p * _k * sizeof(float));
	cudaMalloc(&oldCodeBook, _k * d_dim * sizeof(float));

	dist = new float[_k * d_p * _k];

	uint nClusters = 1;
	// initialize to get the first cluster average
	cudaMemset(assignd, 0, _N * d_p * sizeof(uint));

	cudaMemset(retirementCountd, 0, _k * d_p * sizeof(uint));
	cudaMemset(countd, 0, _k * d_p * sizeof(int));
	cudaMemset(d_codeBook, 0, d_dim * sizeof(float));
	cudaMemset(oldCodeBook, 0, d_dim * sizeof(float));

	getClusterAverage(d_codeBook, countd, retirementCountd, 1, _A, _N, assignd);

	float epsilon = 0.0001;
	float change = 0.;

	while (nClusters < _k) {

		splitCodeBook(nClusters, epsilon);
		cout << "nClusters" << nClusters << endl;

		uint converged = 0;

		do {

			cudaMemset(countd, 0, _k * d_p * sizeof(int));

			getAssignment(assignd, d_codeBook, _A, nClusters, _N);

			//	outputVecUint("Assign", assignd, 100);

			getClusterAverage(d_codeBook, countd, retirementCountd, nClusters,
					_A, _N, assignd);

//			cout << nClusters << endl;
//			outputVec("count:", countd, nClusters * d_p);
			//outputVec("avg: ", d_codeBook, d_dim);

			cudaMemcpy(assign, assignd, _N * d_p * sizeof(uint),
					cudaMemcpyDeviceToHost);
			converged = 0;
			for (int i = 0; i < _N * d_p; i++) {
				if (assign[i] != old_assign[i]) {
					converged++;
				}
			}
			memcpy(old_assign, assign, _N * d_p * sizeof(uint));

//			cout << "non- converged" << converged << endl;

			calcDist(distd, d_codeBook, oldCodeBook, nClusters, nClusters);
			cudaMemcpy(dist, distd, nClusters * nClusters * d_p * sizeof(float),
					cudaMemcpyDeviceToHost);

			change = 0.;

			for (int i = 0; i < nClusters; i++) {
				for (int p = 0; p < d_p; p++) {
					change += dist[(i * nClusters + i) * d_p + p];
					//	cout << dist[(i * nClusters + i) * d_p + p ] << endl;
				}
			}

//			cout << "change " << change << endl;

			cudaMemcpy(oldCodeBook, d_codeBook,
					d_dim * nClusters * sizeof(float),
					cudaMemcpyDeviceToDevice);

			//	} while ((change > 0.000005) && (converged > 0.00002 * _N));
		} while ((change > 0.005) && (converged > 0.002 * _N));

		//  } while ((change > 0.005) && (converged > 0.02 * _N));

		//outputMat("dist:", distd, _N, nClusters);
//		outputMat("codebook", d_codeBook, nClusters, d_dim);

//		getMaxRad(maxRadd, nClusters, assignd, _N, distd);

//		calcDist(distd, d_codeBook, d_codeBook, nClusters, nClusters, d_dim);

//		outputMat("codebook Dist", distd, nClusters, nClusters);

//		char c;
//		cin >> c;

	}

	d_nClusters = _k;

	cudaFree(oldCodeBook);
	cudaFree(distd);
	cudaFree(countd);
	cudaFree(retirementCountd);
	cudaFree(assignd);

	delete[] dist;
	delete[] old_assign;
	delete[] assign;
}

float ProQuantization::calcStatistics(vector<float>& _histogram, float* _Q,
		uint _QN, float *_dbVec, uint _nDB,
		std::vector<std::pair<uint, uint> >& _distSeq) {
//
//	for (int i = 0; i < _distSeq.size(); i++) {
//			cout << i <<  "\t" << _distSeq[i].second << endl;
//		}
//
//	return 0.;

	uint k1 = d_nClusters;

	float* resd;
	cudaMalloc(&resd, d_p * _nDB * sizeof(float));

	float* distClusterd;
	cudaMalloc(&distClusterd, d_p * d_nClusters * sizeof(float));

	uint* assignd;
	cudaMalloc(&assignd, d_nClusters * d_p * _QN * sizeof(uint));

	uint* assignh;
	uint* refAssignh;

	assignh = new uint[d_nClusters * d_p];
	refAssignh = new uint[d_nClusters * d_p];

	float* resh = new float[d_p * _nDB];
	float* distClusterh = new float[d_p * d_nClusters];

	uint numSamples = 1000;

	uint numNearest = 1;

	float accu[d_p];

	for (int p = 0; p < d_p; p++)
		accu[p] = 0.;

	_histogram.resize(pow(d_nClusters, d_p));

	for (int i = 0; i < _histogram.size(); i++) {
		_histogram[i] = 0.;
	}

	vector<uint> denom;

	denom.resize(d_p);

	denom[0] = 1;
	for (int p = 1; p < d_p; p++) {
		denom[p] = denom[p - 1] * d_nClusters;
	}

	for (int j = 0; j < numSamples; j++) {
		calcDist(resd, _dbVec, _Q + j * d_dim, _nDB, 1);

		cudaMemcpy(resh, resd, d_p * _nDB * sizeof(float),
				cudaMemcpyDeviceToHost);

		vector<pair<float, uint> > ddd;
		ddd.clear();
		ddd.resize(_nDB);

		for (int i = 0; i < _nDB; i++) {
			float val = 0.;
			for (int p = 0; p < d_p; p++)
				val += resh[i * d_p + p];
			ddd[i] = pair<float, uint>(val, i);
		}

		sort(ddd.begin(), ddd.end());

		for (int i = 0; i < 3; i++)
			cout << " brute: " << ddd[i].second << "  " << ddd[i].first << endl;

		if (ddd[1].first > 19.)
			outputVec("PROBLEMMMMMMMMMMMMMMMMMMMMMMMMM", _Q + j * d_dim, d_dim);

		getKBestAssignment(assignd, d_codeBook, _Q + j * d_dim, d_nClusters, 1,
				d_nClusters);
		cudaMemcpy(refAssignh, assignd, d_p * d_nClusters * sizeof(uint),
				cudaMemcpyDeviceToHost);

		// compute distance to cluster centers
		calcDist(distClusterd, d_codeBook, _Q + j * d_dim, d_nClusters, 1);

		cudaMemcpy(distClusterh, distClusterd,
				d_p * d_nClusters * sizeof(float), cudaMemcpyDeviceToHost);

		// sort for each p
		for (int p = 0; p < d_p; p++) {

			vector<float> d;
			d.resize(d_nClusters);

			for (int c = 0; c < d_nClusters; c++) {
				d[c] = distClusterh[c * d_p + p];
			}
			sort(d.begin(), d.end());
			for (int c = 0; c < d_nClusters; c++) {
				distClusterh[c * d_p + p] = d[c];
			}
		}

//		for (int p = 0; p < d_p; p++) {
//			for (int c = 0; c < d_nClusters; c++) {
//				cout << refAssignh[c * d_p + p] << "  ";
//			}
//			cout << endl;
//		}

		cout << endl;

		for (int i = 0; i < numNearest; i++) {
			cout << i << "  " << ddd[i].second << "  " << ddd[i].first << endl;

			for (int p = 0; p < d_p; p++)
				cout << resh[ddd[i].second * d_p + p] << "  ";
			cout << endl;

			getKBestAssignment(assignd, d_codeBook,
					_dbVec + ddd[i].second * d_dim, d_nClusters, 1, k1);
			cudaMemcpy(assignh, assignd, d_p * sizeof(uint),
					cudaMemcpyDeviceToHost);

			cout << "nearest" << endl;

			vector<uint> slot;
			slot.resize(d_p);
			for (int p = 0; p < d_p; p++) {
				cout << assignh[p] << "  ";
				uint b = 0;
				for (; b < d_nClusters; b++) {
					if (assignh[p] == refAssignh[b * d_p + p])
						break;
				}
				accu[p] += b;

				slot[p] = b;
				//_histogram[b] += 1.;
			}

//			uint code = slot[0];
//			for (int p = 1; p < d_p; p++)
//				code = code * d_nClusters + slot[p];
			uint code = slot[d_p - 1];
			for (int p = d_p - 2; p >= 0; p--)
				code = code * d_nClusters + slot[p];
			cout << endl;

			// lookup bin n distanceSequence
			uint bin = 0;
			for (; bin < _distSeq.size(); bin++) {
//				cout << "\t" << _distSeq[bin].second;
				if (_distSeq[bin].second == code)
					break;
			}

#if 1
			// compute the exact location in the sorted enumerated bins
			vector<pair<float, uint> > dist;
			dist.clear();
			for (int i = 0; i <= bin; i++) {
				uint code = _distSeq[i].second;
				float d = 0.;
				for (int p = 0; p < d_p; p++) {
					uint c = code / denom[p] % d_nClusters;
					d += distClusterh[c * d_p + p];
				}
				dist.push_back( pair<float,uint>( d, code));
			}
			sort(dist.begin(), dist.end());

//			uint length = (dist.size() > 10) ? 10 : dist.size();
//			for (int k = 0; k < length; k++) {
//				cout << k << " " << dist[k].first << "  " << dist[k].second << endl;
//			}

			uint oldbin = bin;
			bin = 0;
			for (; bin < dist.size(); bin++) {
				//				cout << "\t" << _distSeq[bin].second;
				if (dist[bin].second == code)
				break;
			}

			if (oldbin > bin) {
				cout << "improvement " << oldbin << "  " << bin << endl;
			}
#endif
//			cout << endl;
//			cout << code << ", bin:  " << bin;
//			for (int p = 0; p < d_p; p++)
//				cout << "\t" << slot[p];
//
//			cout << endl;

			_histogram[bin] += 1;

			cout << endl;

		}

	}

	cout << "accumulated" << endl;

	float total = 0.;

	for (int p = 0; p < d_p; p++) {
		accu[p] /= numSamples;

		cout << accu[p] << " " << endl;

		total += accu[p];
	}

	total /= (float) d_p;

//	for (int i = 0; i < _histogram.size(); i++) {
//		_histogram[i] /= numSamples * d_p;
//	}
	for (int i = 0; i < _histogram.size(); i++) {
		_histogram[i] /= numSamples;
	}

// compute cummulative distribution
	for (int i = 1; i < _histogram.size(); i++) {
		_histogram[i] += _histogram[i - 1];
	}

	delete[] resh;

	delete[] assignh;
	delete[] refAssignh;

	cudaFree(assignd);
	cudaFree(resd);

	return total;
}

void ProQuantization::testDistReal(float* _Mh, float* _Md, uint _N) {

	uint numSamples = 10;

	float* resd;
	cudaMalloc(&resd, numSamples * d_p * _N * sizeof(float));

	float* resh;
	resh = new float[numSamples * d_p * _N];

	calcDist(resd, _Md, _Md, _N, numSamples);

	cudaMemcpy(resh, resd, numSamples * d_p * _N * sizeof(float),
			cudaMemcpyDeviceToHost);

	for (int j = 0; j < numSamples; j++) {

		cout << "host:   ";
		for (int l = 0; l < 100; l++) {
			float dist = 0.;
			for (int i = 0; i < d_dim; i++) {
				float v = _Mh[j * d_dim + i] - _Mh[l * d_dim + i];
				dist += v * v;
			}

			cout << dist << " ";
		}
		cout << endl;

		cout << "device: ";
		for (int l = 0; l < 100; l++) {
			float dist = 0.;
			for (int i = 0; i < d_p; i++) {

				dist += resh[(j * _N + l) * d_p + i];
			}

			cout << dist << " ";
		}
		cout << endl;

	}

	delete[] resh;

	cudaFree(resd);
}

void ProQuantization::testCodeBook() {

	float* codeBookh;
	codeBookh = new float[d_nClusters * d_dim];

	cudaMemcpy(codeBookh, d_codeBook, d_nClusters * d_dim * sizeof(float),
			cudaMemcpyDeviceToHost);

	cout << "length " << endl;
	for (int j = 0; j < d_nClusters; j++) {

		float* v = codeBookh + j * d_dim;

		float dist = 0.;
		for (int i = 0; i < d_dim; i++)
			dist += v[i] * v[i];
		dist = sqrt(dist);

		cout << dist << " ";

	}
	cout << endl;

	delete[] codeBookh;
}

} /* namespace */

#endif /* NEARESTNEIGHBOR_PROQUANTIZATION_C */
