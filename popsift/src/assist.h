#pragma once

#include <cuda_runtime.h>
#include <iostream>

using namespace std;

std::ostream& operator<<( std::ostream& ostr, const dim3& p );

namespace popart
{

/* This computation is needed very frequently when a dim3 grid block is
 * initialized. It ensure that the tail is not forgotten.
 */
__device__ __host__
inline int grid_divide( int size, int divider )
{
    return size / divider + ( size % divider != 0 ? 1 : 0 );
}

template <typename T>
__device__
inline T d_abs( T value )
{
    return ( ( value < 0 ) ? -value : value );
}

template <typename T>
__device__
inline int d_sign( T value )
{
    return ( ( value < 0 ) ? -1 : 1 );
}

#if 0
__device__
inline
bool reduce_OR_32x32( bool cnt )
{
    __shared__ int reduce_array[32];

    int cnt_row = __any( cnt );
    if( threadIdx.x == 0 ) {
        reduce_array[threadIdx.y] = cnt_row;
    }
    __syncthreads();
    if( threadIdx.y == 0 ) {
        int cnt_col = __any( reduce_array[threadIdx.x] );
        if( threadIdx.x == 0 ) {
            reduce_array[0] = cnt_col;
        }
    }
    __syncthreads();
    cnt_row = reduce_array[0];
    return ( cnt_row != 0 );
}
#endif

}; // namespace popart

