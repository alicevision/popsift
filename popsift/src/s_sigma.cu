#include "s_pyramid.h"

#include "debug_macros.h"
#include <cuda_runtime.h>

namespace popart {

__device__ __constant__ float d_sigma0;
__device__ __constant__ float d_sigma_k;

void Pyramid::init_sigma( float sigma0, uint32_t levels )
{
    cudaError_t err;

    err = cudaMemcpyToSymbol( d_sigma0, &sigma0,
                              sizeof(float), 0,
                              cudaMemcpyHostToDevice );
    POP_CUDA_FATAL_TEST( err, "Failed to upload sigma0 to device: " );

    const float sigma_k = powf(2.0f, 1.0f / levels );

    err = cudaMemcpyToSymbol( d_sigma_k, &sigma_k,
                              sizeof(float), 0,
                              cudaMemcpyHostToDevice );
    POP_CUDA_FATAL_TEST( err, "Failed to upload sigma_k to device: " );
}

} // namespace popart
