#include <cuda_runtime.h>
#include <stdio.h>

__global__
void try_shuffle( )
{
    int hist[8];
    int var;

    for( int i=0; i<8; i++ ) {
        hist[i] = i;
    }

    for( int i=0; i<8; i++ ) {
        var = hist[i];
        var     += __shfl_xor( var, 1 );
        var     += __shfl_xor( var, 2 );
        var     += __shfl_xor( var, 4 );
        hist[i]  = var;
    }

    printf( "Thread %d : %5d %5d %5d %5d %5d %5d %5d %5d\n",
            threadIdx.x,
            hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], hist[6], hist[7] );

    __syncthreads();
}

int main()
{
    dim3 block(8);
    dim3 grid(1);
    try_shuffle<<<grid,block>>>( );
    cudaDeviceSynchronize();
}

