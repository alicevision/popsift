#include "sift_extrema_mgmt.h"
#include "debug_macros.h"

namespace popart
{

int h_max_extrema      = 0;
int h_max_orientations = 0;
__device__ __constant__ int d_max_extrema;
__device__ __constant__ int d_max_orientations;

void ExtremaMgmt::init( int max_extrema )
{
    cudaError_t err;

    h_max_extrema      = max_extrema;
    h_max_orientations = max_extrema + max_extrema/4;

    err = cudaMemcpyToSymbol( d_max_extrema, &h_max_extrema,
                              sizeof(int), 0,
                              cudaMemcpyHostToDevice );
    POP_CUDA_FATAL_TEST( err, "Failed to upload h_max_extrema to device: " );

    err = cudaMemcpyToSymbol( d_max_orientations, &h_max_orientations,
                              sizeof(int), 0,
                              cudaMemcpyHostToDevice );
    POP_CUDA_FATAL_TEST( err, "Failed to upload h_max_orientations to device: " );
}

void ExtremaMgmt::reset( )
{
    _counter = 0;
}

__device__
int ExtremaMgmt::atomicAddCounter( int ct )
{
    int idx = atomicAdd( &_counter, ct );
    return idx;
}


} // namespace iopart

