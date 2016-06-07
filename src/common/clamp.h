#pragma once

template<class T>
__device__ __host__
inline T clamp( T val, uint32_t maxval )
{
    return min( max( val, 0 ), maxval - 1 );
}

template<class T>
__device__ __host__
inline T clamp( T val, uint32_t minval, uint32_t maxval )
{
    return min( max( val, minval ), maxval - 1 );
}

