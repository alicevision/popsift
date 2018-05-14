/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <pthread.h> // for pthread_self

#include "sift_config.h"

namespace popsift
{

std::ostream& operator<<( std::ostream& ostr, const dim3& p );

/*
 * Assistance with compatibility-breaking builtin function changes
 */
#ifdef HAVE_SHFL_DOWN_SYNC
template<typename T> __device__ inline T shuffle     ( T variable, int src   ) { return __shfl_sync     ( 0xffffffff, variable, src   ); }
template<typename T> __device__ inline T shuffle_up  ( T variable, int delta ) { return __shfl_up_sync  ( 0xffffffff, variable, delta ); }
template<typename T> __device__ inline T shuffle_down( T variable, int delta ) { return __shfl_down_sync( 0xffffffff, variable, delta ); }
template<typename T> __device__ inline T shuffle_xor ( T variable, int delta ) { return __shfl_xor_sync ( 0xffffffff, variable, delta ); }
template<typename T> __device__ inline T ballot      ( T pred )                { return __ballot_sync   ( 0xffffffff, pred ); }
template<typename T> __device__ inline T any         ( T pred )                { return __any_sync      ( 0xffffffff, pred ); }
template<typename T> __device__ inline T all         ( T pred )                { return __all_sync      ( 0xffffffff, pred ); }

template<typename T> __device__ inline T shuffle     ( T variable, int src  , int ws ) { return __shfl_sync     ( 0xffffffff, variable, src  , ws ); }
template<typename T> __device__ inline T shuffle_up  ( T variable, int delta, int ws ) { return __shfl_up_sync  ( 0xffffffff, variable, delta, ws ); }
template<typename T> __device__ inline T shuffle_down( T variable, int delta, int ws ) { return __shfl_down_sync( 0xffffffff, variable, delta, ws ); }
template<typename T> __device__ inline T shuffle_xor ( T variable, int delta, int ws ) { return __shfl_xor_sync ( 0xffffffff, variable, delta, ws ); }
#else
template<typename T> __device__ inline T shuffle     ( T variable, int src   ) { return __shfl     ( variable, src   ); }
template<typename T> __device__ inline T shuffle_up  ( T variable, int delta ) { return __shfl_up  ( variable, delta ); }
template<typename T> __device__ inline T shuffle_down( T variable, int delta ) { return __shfl_down( variable, delta ); }
template<typename T> __device__ inline T shuffle_xor ( T variable, int delta ) { return __shfl_xor ( variable, delta ); }
template<typename T> __device__ inline T ballot      ( T pred )                { return __ballot   ( pred ); }
template<typename T> __device__ inline T any         ( T pred )                { return __any      ( pred ); }
template<typename T> __device__ inline T all         ( T pred )                { return __all      ( pred ); }

template<typename T> __device__ inline T shuffle     ( T variable, int src  , int ws ) { return __shfl     ( variable, src  , ws ); }
template<typename T> __device__ inline T shuffle_up  ( T variable, int delta, int ws ) { return __shfl_up  ( variable, delta, ws ); }
template<typename T> __device__ inline T shuffle_down( T variable, int delta, int ws ) { return __shfl_down( variable, delta, ws ); }
template<typename T> __device__ inline T shuffle_xor ( T variable, int delta, int ws ) { return __shfl_xor ( variable, delta, ws ); }
#endif

/* This computation is needed very frequently when a dim3 grid block is
 * initialized. It ensure that the tail is not forgotten.
 */
__device__ __host__
inline int grid_divide( int size, int divider )
{
    return size / divider + ( size % divider != 0 ? 1 : 0 );
}

__device__ static inline
float readTex( cudaTextureObject_t tex, float x, float y, float z )
{
    /* Look at CUDA C programming guide:
     * Doesn't matter if we access Linear or Point textures,
     * we will get the expected cell (or an interpolation very 
     * close by) iff we add 0.5f to X and Y coordinate.
     */
    return tex2DLayered<float>( tex, x+0.5f, y+0.5f, z );
}

__device__ static inline
float readTex( cudaTextureObject_t tex, float x, float y )
{
    return tex2D<float>( tex, x+0.5f, y+0.5f );
}


/*********************************************************************************
 * For a debug output to cerr with thread ID at the line start
 *********************************************************************************/

static inline unsigned int microhash( int val )
{
    val = ( val < 0 ? -val : val );
    unsigned int ret = ( ( ( val & ( 0xf <<  0 ) ) >>  0 )
                       ^ ( ( val & ( 0xf <<  4 ) ) >>  4 )
                       ^ ( ( val & ( 0xf <<  8 ) ) >>  8 )
                       ^ ( ( val & ( 0xf << 12 ) ) >> 12 )
                       ^ ( ( val & ( 0xf << 16 ) ) >> 16 )
                       ^ ( ( val & ( 0xf << 20 ) ) >> 20 )
                       ^ ( ( val & ( 0xf << 24 ) ) >> 24 )
                       ^ ( ( val & ( 0xf << 28 ) ) >> 28 ) );
    return ret;
}
#define DERR std::cerr << std::hex << popsift::microhash(pthread_self()) << std::dec << "    "

}; // namespace popsift
