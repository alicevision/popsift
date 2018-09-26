#pragma once


/*! \file  helper.hh
    \brief a collection of helper classes
 */
//#define OUTPUT

#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

using namespace std;

#define MAX_THREADS 512
#define MAX_BLOCKS 65535
#define WARP_SIZE 32

namespace pqt {

void checkCudaErrors( cudaError_t err );

inline __device__ float sqr(const float &x) {
	return x * x;
}

// returns the ceiling of the log base 2 of an integer, i.e. the mimimum number of bits needed to store x
inline unsigned int log2(unsigned int x) {
	unsigned int y;

	for (y = 0; y < 64; y++)
		if (!((x - 1) >> y))
			break;

	y = 1 << y;

	return y;
}


inline uint idiv(uint _n, uint _d) {
	uint val = _n / _d;
	return (_n % _d) ? val + 1 : val;
}


void outputMat(const std::string& _S, const float* _A,
		uint _rows, uint _cols);

void outputVec(const std::string& _S, const float* _v,
		uint _n);


void outputVecUint(const std::string& _S, const uint* _v,
		uint _n);

void outputVecInt(const std::string& _S, const int* _v,
		uint _n);

inline void setReductionBlocks(dim3& _block, uint _n) {
	unsigned int nThreads = log2(_n);
	nThreads = (nThreads < WARP_SIZE) ? WARP_SIZE : nThreads;
	nThreads = (nThreads > MAX_THREADS) ? MAX_THREADS : nThreads;
	_block = dim3(nThreads, 1, 1);
}

//__device__ void bitonic(volatile float _val[], volatile uint _idx[], uint _N);


void countZeros(const std::string& _S, const uint* _v, uint _n);

} /* namespace */

#ifdef __CUDACC__
#if 1
namespace popsift
{
template<typename T> __device__ inline T shuffle     ( T variable, int src   ) { return __shfl_sync     ( 0xffffffff, variable, src   ); }
template<typename T> __device__ inline T shuffle_up  ( T variable, int delta ) { return __shfl_up_sync  ( 0xffffffff, variable, delta ); }
template<typename T> __device__ inline T shuffle_down( T variable, int delta ) { return __shfl_down_sync( 0xffffffff, variable, delta ); }
template<typename T> __device__ inline T shuffle_xor ( T variable, int delta ) { return __shfl_xor_sync ( 0xffffffff, variable, delta ); }
__device__ inline unsigned int ballot( unsigned int pred ) { return __ballot_sync   ( 0xffffffff, pred ); }
__device__ inline int any            ( unsigned int pred ) { return __any_sync      ( 0xffffffff, pred ); }
__device__ inline int all            ( unsigned int pred ) { return __all_sync      ( 0xffffffff, pred ); }

template<typename T> __device__ inline T shuffle     ( T variable, int src  , int ws ) { return __shfl_sync     ( 0xffffffff, variable, src  , ws ); }
template<typename T> __device__ inline T shuffle_up  ( T variable, int delta, int ws ) { return __shfl_up_sync  ( 0xffffffff, variable, delta, ws ); }
template<typename T> __device__ inline T shuffle_down( T variable, int delta, int ws ) { return __shfl_down_sync( 0xffffffff, variable, delta, ws ); }
template<typename T> __device__ inline T shuffle_xor ( T variable, int delta, int ws ) { return __shfl_xor_sync ( 0xffffffff, variable, delta, ws ); }
#else
template<typename T> __device__ inline T shuffle     ( T variable, int src   ) { return __shfl     ( variable, src   ); }
template<typename T> __device__ inline T shuffle_up  ( T variable, int delta ) { return __shfl_up  ( variable, delta ); }
template<typename T> __device__ inline T shuffle_down( T variable, int delta ) { return __shfl_down( variable, delta ); }
template<typename T> __device__ inline T shuffle_xor ( T variable, int delta ) { return __shfl_xor ( variable, delta ); }
__device__ inline unsigned int ballot( unsigned int pred ) { return __ballot   ( pred ); }
__device__ inline int any            ( unsigned int pred ) { return __any      ( pred ); }
__device__ inline int all            ( unsigned int pred ) { return __all      ( pred ); }

template<typename T> __device__ inline T shuffle     ( T variable, int src  , int ws ) { return __shfl     ( variable, src  , ws ); }
template<typename T> __device__ inline T shuffle_up  ( T variable, int delta, int ws ) { return __shfl_up  ( variable, delta, ws ); }
template<typename T> __device__ inline T shuffle_down( T variable, int delta, int ws ) { return __shfl_down( variable, delta, ws ); }
template<typename T> __device__ inline T shuffle_xor ( T variable, int delta, int ws ) { return __shfl_xor ( variable, delta, ws ); }
#endif

}; // namespace popsift
#endif // __CUDACC__

