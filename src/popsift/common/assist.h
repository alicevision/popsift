/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <popsift/sift_config.h>

#include <iostream>
#include <thread>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif



namespace popsift
{

/*
 * Assistance with compatibility-breaking builtin function changes
 */
#if 0
template<typename T> inline T shuffle     ( T variable, int src   ) { return __shfl     ( variable, src   ); }
template<typename T> inline T shuffle_up  ( T variable, int delta ) { return __shfl_up  ( variable, delta ); }
template<typename T> inline T shuffle_down( T variable, int delta ) { return __shfl_down( variable, delta ); }
template<typename T> inline T shuffle_xor ( T variable, int delta ) { return __shfl_xor ( variable, delta ); }
inline unsigned int ballot( unsigned int pred ) { return __ballot   ( pred ); }
inline int any            ( unsigned int pred ) { return __any      ( pred ); }
inline int all            ( unsigned int pred ) { return __all      ( pred ); }

template<typename T> inline T shuffle     ( T variable, int src  , int ws ) { return __shfl     ( variable, src  , ws ); }
template<typename T> inline T shuffle_up  ( T variable, int delta, int ws ) { return __shfl_up  ( variable, delta, ws ); }
template<typename T> inline T shuffle_down( T variable, int delta, int ws ) { return __shfl_down( variable, delta, ws ); }
template<typename T> inline T shuffle_xor ( T variable, int delta, int ws ) { return __shfl_xor ( variable, delta, ws ); }
#endif

/* This computation is needed very frequently when a dim3 grid block is
 * initialized. It ensure that the tail is not forgotten.
 */
inline int grid_divide( int size, int divider )
{
    return size / divider + ( size % divider != 0 ? 1 : 0 );
}

#if 0
    /*
     * Must be eliminated.
     * But don't forget that the +0.5 is probably also required for our own
     * interpolation functions.
     */
static inline
float readTex( cudaTextureObject_t tex, float x, float y, float z )
{
    /* Look at CUDA C programming guide:
     * Doesn't matter if we access Linear or Point textures,
     * we will get the expected cell (or an interpolation very 
     * close by) iff we add 0.5f to X and Y coordinate.
     */
    return tex2DLayered<float>( tex, x+0.5f, y+0.5f, z );
}

static inline
float readTex( cudaTextureObject_t tex, float x, float y )
{
    return tex2D<float>( tex, x+0.5f, y+0.5f );
}
#endif

inline std::thread::id getCurrentThreadId()
{
    return std::this_thread::get_id();
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

static inline unsigned int microhash( const std::thread::id& id )
{
    std::hash<std::thread::id> hasher;
    return microhash( hasher(id) );
}

#define DERR std::cerr << std::hex << popsift::microhash(getCurrentThreadId()) << std::dec << "    "


static size_t getPageSize()
{
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwPageSize;
#else
    return sysconf(_SC_PAGESIZE);
#endif
}

static void* memalign(size_t alignment, size_t size)
{
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    void* ret;
    int err = posix_memalign( &ret, alignment, size );
    if( err != 0 ) {
        errno = err;
        ret = nullptr;
    }
    return ret;
#endif
}

static void memalign_free( void* ptr )
{
#ifdef _WIN32
    _aligned_free( ptr );
#else
    free( ptr );
#endif
}

} // namespace popsift
