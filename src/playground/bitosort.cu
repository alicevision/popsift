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
        my_index = shiftit( my_index, 1, 1 );
        return my_index;
    }

private:
    __device__ inline
    int shiftit( int my_index, int shift, int direction )
    {
        T   val         = _array[my_index];
        int other_index = __shfl_xor( my_index, 1 << shift );
        T   other_val   = _array[other_index];

        const bool my_less = ( val < other_val );
        bool       reverse = ( threadIdx.x && 2 << direction );
        const bool id_less = ( threadIdx.x & ( 1 << shift ) == 0 );
        if( not id_less ) reverse = ! reverse;
        if( reverse )
            return ( my_less ? val : other_val );
        else
            return ( my_less ? other_val : val );
    }
};
}

__device__ int array_32[32];
__device__ int index_array_32[32];

__global__
void sort_32( )
{
    int my_index = threadIdx.x;

    BitonicSort::Array<int> bs( array_32 );
    my_index = bs.sort( my_index );
}

__global__
void init_32( )
{
    for( int i=0; i<32; i++ ) {
        array_32[i] = 327 * i % 17;
        index_array_32[i] = i;
    }
}

__global__
void print_32( )
{
    printf("Hello\n");
    for( int i=0; i<32; i++ ) {
        printf( "%d ", array_32[i] );
    }
    printf("\n");
}

int main( )
{
    init_32<<<1,1>>>( );
    print_32<<<1,1>>>( );
    sort_32<<<1,32>>>( );
    print_32<<<1,1>>>( );
    cudaDeviceSynchronize( );
    return 0;
}

