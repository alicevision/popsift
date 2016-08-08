#include <cuda_runtime.h>
#include <stdio.h>

namespace BitonicSort
{
template<class T>
class Array
{
    T*  _array;
public:
    __device__ inline
    Array( T* array )
        : _array( array )
    {
        if( threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 )
        {
            if( blockDim.x != 32 ) {
                printf( "%s requires warps of 32\n", __func__ );
            }
        }
    }

    __device__ inline
    int sort( int my_index )
    {
        for( int outer=0; outer<5; outer++ ) {
            for( int inner=outer; inner>=0; inner-- ) {
                my_index = shiftit( my_index, inner, outer+1 );
            }
        }
        return my_index;
    }

private:
    __device__ inline
    int shiftit( int my_index, int shift, int direction )
    {
        const T    my_val      = _array[my_index];
        const T    other_val   = __shfl_xor( my_val, 1 << shift );
        const bool my_less     = ( my_val < other_val );
        const bool reverse     = ( threadIdx.x & ( 1 << direction ) );
        const bool id_less     = ( ( threadIdx.x & ( 1 << shift ) ) == 0 );
        const bool must_swap   = not ( my_less ^ id_less ^ reverse );

        return ( must_swap ? __shfl_xor( my_index, 1 << shift )
                           : my_index );
    }
};
}

__device__ int array_32[32];
__device__ int index_array_32[32];

__global__
void sort_32( )
{
    int my_index = index_array_32[threadIdx.x];

    BitonicSort::Array<int> bs( array_32 );
    my_index = bs.sort( my_index );

    index_array_32[threadIdx.x] = my_index;
}

__global__
void init_32( int i )
{
    for( int i=0; i<32; i++ ) index_array_32[i] = i;

    if( i == 0 )
        for( int i=0; i<32; i++ ) array_32[i] = 327 * i % 97;
    else if( i == 1 )
        for( int i=0; i<32; i++ ) array_32[i] = 32 - i;
    else
        for( int i=0; i<32; i++ ) array_32[i] = i;
}

__global__
void print_32( )
{
    for( int i=0; i<32; i++ ) {
        printf( "%d ", array_32[index_array_32[i]] );
    }
    printf("\n");
}

int main( )
{
    init_32<<<1,1>>>( 0 );
    print_32<<<1,1>>>( );
    sort_32<<<1,32>>>( );
    print_32<<<1,1>>>( );
    cudaDeviceSynchronize( );
    init_32<<<1,1>>>( 1 );
    print_32<<<1,1>>>( );
    sort_32<<<1,32>>>( );
    print_32<<<1,1>>>( );
    cudaDeviceSynchronize( );
    init_32<<<1,1>>>( 2 );
    print_32<<<1,1>>>( );
    sort_32<<<1,32>>>( );
    print_32<<<1,1>>>( );
    cudaDeviceSynchronize( );
    return 0;
}

