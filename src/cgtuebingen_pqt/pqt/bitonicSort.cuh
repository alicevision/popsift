#ifndef BITONIC_SORT_CUH
#define BITONIC_SORT_CUH

#include <stdio.h>
#include <stdlib.h>

namespace pqt {
template<class T>
__device__ void swap(T& _a, T&_b) {
	T h = _a;
	_a = _b;
	_b = h;
}

// parallel bitonic sort
template<class T>
__device__ void bitonic3(volatile T _val[], volatile uint _idx[], uint _N) {

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

// parallel bitonic sort
template<class T>
__device__ void bitonicLarge(volatile T _val[], volatile uint _idx[], uint _N) {

	for (int k = 2; k <= _N; k <<= 1) {

		// bitonic merge
		for (int j = k / 2; j > 0; j /= 2) {

			for (int tid = threadIdx.x; tid < _N; tid += blockDim.x) {
				int ixj = tid ^ j;  // XOR
				if ((ixj > tid) && (ixj < _N)) {
					if ((tid & k) == 0) // ascending - descending
							{
						if (_val[tid] > _val[ixj]) {

							swap(_val[tid], _val[ixj]);
							swap(_idx[tid], _idx[ixj]);
						}
					} else {
						if (_val[tid] < _val[ixj]) {

							swap(_val[tid], _val[ixj]);
							swap(_idx[tid], _idx[ixj]);
						}

					}
				}
			}
			__syncthreads();
		}
	}
}

// parallel bitonic sort (descending)
template<class T>
__device__ void bitonic3Descending(volatile T _val[], volatile uint _idx[],
		uint _N) {

	for (int k = 2; k <= _N; k <<= 1) {

		// bitonic merge
		for (int j = k / 2; j > 0; j /= 2) {
			int ixj = threadIdx.x ^ j;  // XOR
			if ((ixj > threadIdx.x) && (ixj < _N)) {
				if ((threadIdx.x & k) != 0) // ascending - descending
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

template<typename scalar>
__device__ scalar scan_warp2(volatile scalar * ptr, bool _inclusive,
		const uint idx = threadIdx.x) {
	const uint lane = idx & 31;

	if (lane >= 1)
		ptr[idx] = ptr[idx - 1] + ptr[idx];
	if (lane >= 2)
		ptr[idx] = ptr[idx - 2] + ptr[idx];
	if (lane >= 4)
		ptr[idx] = ptr[idx - 4] + ptr[idx];
	if (lane >= 8)
		ptr[idx] = ptr[idx - 8] + ptr[idx];
	if (lane >= 16)
		ptr[idx] = ptr[idx - 16] + ptr[idx];

	if (_inclusive)
		return ptr[idx];
	else
		return (lane > 0) ? ptr[idx - 1] : 0.;
}

__device__ float scan_block2(volatile uint *ptr, bool _inclusive,
		const uint idx = threadIdx.x) {
	const uint lane = idx & 31;
	const uint warpid = idx >> 5;

	// Step 1: Intra-warp scan in each warp
	float val = scan_warp2(ptr, _inclusive, idx);
	__syncthreads();

	// Step 2: Collect per-warp partial results
	if (lane == 31)
		ptr[warpid] = ptr[idx];
	__syncthreads();

	// Step 3: Use 1st warp to scan per-warp results
	if (warpid == 0)
		scan_warp2(ptr, true, idx);
	__syncthreads();

	// Step 4: Accumulate results from Steps 1 and 3
	if (warpid > 0)
		val = ptr[warpid - 1] + val;
	__syncthreads();

	// Step 5: Write and return the final result
	ptr[idx] = val;
	__syncthreads();

	return val;
}

__device__ void scan_blockLarge(volatile uint *ptr, bool _inclusive, uint _N,
		const uint idx = threadIdx.x) {
//	const uint lane = idx & 31;
	const uint warpid = idx >> 5;

	float val[4];

	uint nWarps = blockDim.x / 32;
	uint nIter = _N / blockDim.x;

	// Step 1: Intra-warp scan in each warp
	for (int iter = 0; iter < nIter; iter++) {
		__syncthreads();
		val[iter] = scan_warp2(ptr + iter * blockDim.x, _inclusive);
		__syncthreads();
	}

	// Step 2: Collect per-warp partial results

	if (idx < _N / 32)
		ptr[idx] = ptr[idx * 32 + 31];
	__syncthreads();

	// Step 3: scan per-warp results
	scan_block2(ptr, true);
	__syncthreads();

//	if (idx == 0) {
//		for (int i = 0 ; i < _N / 32; i++)
//			printf("%d: %d \n", i, ptr[i]);
//	}
//
//	__syncthreads();

	// Step 4: Accumulate results from Steps 1 and 3

	for (int tid = idx, iter = 0; tid < _N; tid += blockDim.x, iter++)
		if (tid > 31)
			val[iter] = ptr[warpid + iter * nWarps - 1] + val[iter];
	__syncthreads();

	// Step 5: Write and return the final result
	for (int tid = idx, iter = 0; tid < _N; tid += blockDim.x, iter++)
		ptr[tid] = val[iter];
	__syncthreads();

}

__global__ void sortTestLarge(uint _N) {

	extern __shared__ float shm[];

	uint* shmIdx = (uint*) (shm + _N);

	for (int tid = threadIdx.x; tid < _N; tid += blockDim.x) {
		shm[tid] = _N - tid;
		shmIdx[tid] = tid;
	}

	__syncthreads();
	bitonicLarge(shm, shmIdx, _N);

	for (int tid = threadIdx.x; tid < _N; tid += blockDim.x) {
		if (shmIdx[tid] != (_N - tid - 1))
			printf("not passed %d %d %d ! \n", _N, tid, shmIdx[tid]);
	}

}

__global__ void scanTestLarge(uint _N) {

	extern __shared__ float shmf[];

	uint* shm = (uint*) shmf;

	for (int tid = threadIdx.x; tid < _N; tid += blockDim.x) {
		shm[tid] = 1;
	}

	__syncthreads();
	scan_blockLarge(shm, false, _N);

	for (int tid = threadIdx.x; tid < _N; tid += blockDim.x) {
		if (shm[tid] != tid)
			printf("scanLarge not passed %d %d %d ! \n", _N, tid, shm[tid]);
	}

}
};
#endif