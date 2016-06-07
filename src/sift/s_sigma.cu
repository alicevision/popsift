#include "s_sigma.h"

#include "debug_macros.h"
#include <cuda_runtime.h>

namespace popart {

__device__ __constant__ float d_sigma0;
__device__ __constant__ float d_sigma_k;
__device__ __constant__ float d_edge_limit;
__device__ __constant__ float d_threshold;

void init_sigma( float sigma0, int levels, float threshold, float edge_limit )
{
    cudaError_t err;

    const float sigma_k = powf(2.0f, 1.0f / levels );

    err = cudaMemcpyToSymbol( d_sigma0, &sigma0,
                              sizeof(float), 0,
                              cudaMemcpyHostToDevice );
    POP_CUDA_FATAL_TEST( err, "Failed to upload sigma0 to device: " );

    err = cudaMemcpyToSymbol( d_sigma_k, &sigma_k,
                              sizeof(float), 0,
                              cudaMemcpyHostToDevice );
    POP_CUDA_FATAL_TEST( err, "Failed to upload sigma_k to device: " );

    err = cudaMemcpyToSymbol( d_edge_limit, &edge_limit,
                              sizeof(float), 0,
                              cudaMemcpyHostToDevice );
    POP_CUDA_FATAL_TEST( err, "Failed to upload edge_limit to device: " );

    err = cudaMemcpyToSymbol( d_threshold, &threshold,
                              sizeof(float), 0,
                              cudaMemcpyHostToDevice );
    POP_CUDA_FATAL_TEST( err, "Failed to upload threshold to device: " );
}

} // namespace popart
