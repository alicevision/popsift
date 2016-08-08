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
    void swap( int& l, int& r )
    {
        int m = r;
        r = l;
        l = m;
    }

    __device__ inline
    int sort32( int my_index )
    {
        for( int outer=0; outer<5; outer++ ) {
            for( int inner=outer; inner>=0; inner-- ) {
                my_index = shiftit( my_index, inner, outer+1, false );
            }
        }
        return my_index;
    }

    __device__ inline
    void sort64( int2& my_indeces )
    {
        for( int outer=0; outer<5; outer++ ) {
            for( int inner=outer; inner>=0; inner-- ) {
                my_indeces.x = shiftit( my_indeces.x, inner, outer+1, false );
                my_indeces.y = shiftit( my_indeces.y, inner, outer+1, true );
            }
        }

        if( _array[my_indeces.x] < _array[my_indeces.y] ) swap( my_indeces.x, my_indeces.y );

        for( int outer=0; outer<5; outer++ ) {
            for( int inner=outer; inner>=0; inner-- ) {
                my_indeces.x = shiftit( my_indeces.x, inner, outer+1, false );
                my_indeces.y = shiftit( my_indeces.y, inner, outer+1, false );
            }
        }
    }

private:
    __device__ inline
    int shiftit( const int my_index, const int shift, const int direction, const bool increasing )
    {
        const T    my_val      = _array[my_index];
        const T    other_val   = __shfl_xor( my_val, 1 << shift );
        const bool my_less     = ( my_val < other_val );
        const bool reverse     = ( threadIdx.x & ( 1 << direction ) );
        const bool id_less     = ( ( threadIdx.x & ( 1 << shift ) ) == 0 );
        const bool must_swap   = not ( my_less ^ id_less ^ reverse ^ increasing );

        return ( must_swap ? __shfl_xor( my_index, 1 << shift )
                           : my_index );
    }
};
}

__device__ int array_32[32];
__device__ int index_array_32[32];
__device__ int array_64[64];
__device__ int index_array_64[64];

__global__
void sort_32( )
{
    int my_index = index_array_32[threadIdx.x];

    BitonicSort::Array<int> bs( array_32 );
    my_index = bs.sort32( my_index );

    index_array_32[threadIdx.x] = my_index;
}

__global__
void sort_64( )
{
    BitonicSort::Array<int> bs( array_64 );
    int2 my_indices = make_int2( index_array_64[threadIdx.x],
                                 index_array_64[threadIdx.x + 32] );
    bs.sort64( my_indices );

    index_array_64[threadIdx.x +  0] = my_indices.x;
    index_array_64[threadIdx.x + 32] = my_indices.y;
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
void init_64( int i )
{
    for( int i=0; i<64; i++ ) index_array_64[i] = i;

    if( i == 0 )
        for( int i=0; i<64; i++ ) array_64[i] = 647 * i % 97;
    else if( i == 1 )
        for( int i=0; i<64; i++ ) array_64[i] = 64 - i;
    else
        for( int i=0; i<64; i++ ) array_64[i] = i;
}

__global__
void print_32( )
{
    for( int i=0; i<32; i++ ) {
        printf( "%d ", array_32[index_array_32[i]] );
    }
    printf("\n");
}

__global__
void print_64( )
{
    for( int i=0; i<64; i++ ) {
        printf( "%d ", array_64[index_array_64[i]] );
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

    init_64<<<1,1>>>( 0 );
    print_64<<<1,1>>>( );
    sort_64<<<1,32>>>( );
    print_64<<<1,1>>>( );
    cudaDeviceSynchronize( );
    init_64<<<1,1>>>( 1 );
    print_64<<<1,1>>>( );
    sort_64<<<1,32>>>( );
    print_64<<<1,1>>>( );
    cudaDeviceSynchronize( );
    init_64<<<1,1>>>( 2 );
    print_64<<<1,1>>>( );
    sort_64<<<1,32>>>( );
    print_64<<<1,1>>>( );
    cudaDeviceSynchronize( );

    return 0;
}

