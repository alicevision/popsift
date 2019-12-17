#include <cuda_runtime.h>
#include "../common/assist.h"
#include "bitosort.h"
#include "test_radix_sort.h"

namespace TestRadix
{
__device__ __managed__ int buffer[64];

__shared__ int sh_val[64];

__host__ void push( int* b )
{
    for( int i=0; i<64; i++ )
        buffer[i] = b[i];
}

__host__ void pull( int* b )
{
    for( int i=0; i<64; i++ )
        b[i] = buffer[i];
}

__global__ void gpuCallSort( )
{
    int x = threadIdx.x;

    sh_val[x]    = buffer[x];
    sh_val[x+32] = buffer[x+32];
    __syncthreads();

    int2 best_index = make_int2( threadIdx.x, threadIdx.x + 32 );

    popsift::BitonicSort::Warp32<int> sorter( sh_val );
    sorter.sort64( best_index );
    // sorter.sort32( threadIdx.x );
    __syncthreads();

    buffer[x]    = sh_val[best_index.x];
    buffer[x+32] = sh_val[best_index.y];
}

__host__ void callSort( )
{
    dim3 block( 32, 1, 1 );

    gpuCallSort<<<1,block>>>( );
    cudaDeviceSynchronize();
}

};

