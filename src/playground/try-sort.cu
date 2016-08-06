#include <cuda_runtime.h>
#include <stdio.h>

__shared__ int value[32];

__global__
void init( )
{
    for( int i = 0; i<32; i++ )
    {
        value[i] = ( 323 * i ) % 11;
    }
}

__global__
void print( )
{
    for( int i = 0; i<32; i++ )
    {
        printf("%d ", value[i] );
    }
    printf("\n");
}

__device__ int shiftit( int j, int i, int best_index )
{
    int xorval         = 1 << j;
    // int greater_xorval = 1 << i;
    bool swap_dir     = ( threadIdx.x & (2<<i) ); 

    int  other_index  = __shfl_xor( best_index, xorval );
    bool id_below     = ( threadIdx.x < ( threadIdx.x ^ xorval ) );

    // bool other_bigger = ( value[other_index] > value[best_index] );
    // if( ( threadIdx.x / ( greater_xorval << 1 ) ) & 1 ) other_bigger = !other_bigger;
    // if( ( threadIdx.x >> i ) & 2 ) other_bigger = !other_bigger;
    // if( threadIdx.x & (2<<i) ) other_bigger = !other_bigger;
    // other_bigger = swap_dir ? !other_bigger : other_bigger;

    bool other_bigger = swap_dir ^ ( ( value[other_index] > value[best_index] ) );

    if( id_below ^ other_bigger == 0 ) best_index = other_index;
    return best_index;
}

__global__
void sortit( int j, int i )
{
    int best_index = threadIdx.x;

    best_index = shiftit( j, i, best_index );

    int val = value[best_index];
    __syncthreads();
    value[threadIdx.x] = val;
}

int main()
{
    init<<<1,1>>>();
    print<<<1,1>>>();

    for( int i=0; i<5; i++ ) {
        for( int j=i; j>=0; j-- ) {
            sortit<<<1,32>>>( j, i ); print<<<1,1>>>();
        }
    }

    cudaDeviceSynchronize();
}

