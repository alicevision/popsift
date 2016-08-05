#include "sift_extrema_mgmt.h"
#include "debug_macros.h"

namespace popart
{

MaxInfo h_max;

__device__ __constant__ MaxInfo d_max;

// int h_max_extrema      = 0;
// int h_max_orientations = 0;
// __device__ __constant__ int d_max_extrema;
// __device__ __constant__ int d_max_orientations;

void init_extrema_limits( int max_extrema )
{
    cudaError_t err;

    h_max.extrema      = max_extrema;
    h_max.orientations = max_extrema + max_extrema/4;

    err = cudaMemcpyToSymbol( d_max, &h_max,
                              sizeof(MaxInfo), 0,
                              cudaMemcpyHostToDevice );
    POP_CUDA_FATAL_TEST( err, "Failed to upload h_max_extrema to device: " );
}

} // namespace iopart

