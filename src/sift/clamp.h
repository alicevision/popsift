#pragma once

__device__ __host__
inline int clamp( int val, int maxval )
{
    return min( max( val, 0 ), maxval - 1 );
}

__device__ __host__
inline int clamp( int val, int minval, int maxval )
{
    return min( max( val, minval ), maxval - 1 );
}

