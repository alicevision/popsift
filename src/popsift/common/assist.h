/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <popsift/sift_config.h>
#include <popsift/sift_textures.h>

#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif



namespace popsift
{

std::ostream& operator<<( std::ostream& ostr, const dim3& p );

/*
 * Assistance with compatibility-breaking builtin function changes
 */
#if POPSIFT_IS_DEFINED(POPSIFT_HAVE_SHFL_DOWN_SYNC)
template<typename T> __device__ inline T shuffle     ( T variable, int src   ) { return __shfl_sync     ( 0xffffffff, variable, src   ); }
template<typename T> __device__ inline T shuffle_up  ( T variable, int delta ) { return __shfl_up_sync  ( 0xffffffff, variable, delta ); }
template<typename T> __device__ inline T shuffle_down( T variable, int delta ) { return __shfl_down_sync( 0xffffffff, variable, delta ); }
template<typename T> __device__ inline T shuffle_xor ( T variable, int delta ) { return __shfl_xor_sync ( 0xffffffff, variable, delta ); }
__device__ inline unsigned int ballot( unsigned int pred ) { return __ballot_sync   ( 0xffffffff, pred ); }
__device__ inline int any            ( unsigned int pred ) { return __any_sync      ( 0xffffffff, pred ); }
__device__ inline int all            ( unsigned int pred ) { return __all_sync      ( 0xffffffff, pred ); }

template<typename T> __device__ inline T shuffle     ( T variable, int src  , int ws ) { return __shfl_sync     ( 0xffffffff, variable, src  , ws ); }
template<typename T> __device__ inline T shuffle_up  ( T variable, int delta, int ws ) { return __shfl_up_sync  ( 0xffffffff, variable, delta, ws ); }
template<typename T> __device__ inline T shuffle_down( T variable, int delta, int ws ) { return __shfl_down_sync( 0xffffffff, variable, delta, ws ); }
template<typename T> __device__ inline T shuffle_xor ( T variable, int delta, int ws ) { return __shfl_xor_sync ( 0xffffffff, variable, delta, ws ); }
#else
/* The mask-free builtins. This is the pre-CUDA-9 spelling, and it is also the
 * spelling HIP provides, where the width parameter may be up to the 64-lane
 * wavefront size.
 */
template<typename T> __device__ inline T shuffle     ( T variable, int src   ) { return __shfl     ( variable, src   ); }
template<typename T> __device__ inline T shuffle_up  ( T variable, int delta ) { return __shfl_up  ( variable, delta ); }
template<typename T> __device__ inline T shuffle_down( T variable, int delta ) { return __shfl_down( variable, delta ); }
template<typename T> __device__ inline T shuffle_xor ( T variable, int delta ) { return __shfl_xor ( variable, delta ); }
__device__ inline unsigned int ballot( unsigned int pred ) { return __ballot   ( pred ); }
__device__ inline int any            ( unsigned int pred ) { return __any      ( pred ); }
__device__ inline int all            ( unsigned int pred ) { return __all      ( pred ); }

template<typename T> __device__ inline T shuffle     ( T variable, int src  , int ws ) { return __shfl     ( variable, src  , ws ); }
template<typename T> __device__ inline T shuffle_up  ( T variable, int delta, int ws ) { return __shfl_up  ( variable, delta, ws ); }
template<typename T> __device__ inline T shuffle_down( T variable, int delta, int ws ) { return __shfl_down( variable, delta, ws ); }
template<typename T> __device__ inline T shuffle_xor ( T variable, int delta, int ws ) { return __shfl_xor ( variable, delta, ws ); }
#endif

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
/* Wave64 sub-warp helpers.
 *
 * These kernels were written for a 32-lane NVIDIA warp. On a 64-lane CDNA
 * wavefront PopSift packs two logical 32-thread rows into one wavefront, so the
 * whole-wavefront builtins (__ballot/__any over all 64 lanes) mix the two rows
 * and miscount. The helpers below restrict a collective to the caller's own
 * 32-lane group (its lane's half of the wavefront), reproducing 32-lane warp
 * semantics. The group index is the thread's logical row (threadIdx.y for the
 * extrema kernel); a single-row 32-thread block is group 0. CUDA is unaffected:
 * this block compiles only on HIP.
 */
__device__ inline unsigned int ballot_group( unsigned int pred, int group )
{
    const unsigned long long b = __ballot( pred );
    return (unsigned int)( b >> ( group * 32 ) );
}
__device__ inline int any_group( unsigned int pred, int group )
{
    const unsigned long long b = __ballot( pred );
    return ( (unsigned int)( b >> ( group * 32 ) ) ) != 0u;
}
#else
__device__ inline unsigned int ballot_group( unsigned int pred, int ) { return popsift::ballot( pred ); }
__device__ inline int any_group( unsigned int pred, int ) { return popsift::any( pred ); }
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
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
    /* Observed on gfx90a, ROCm 7.2.1: creating a hardware-linear-filtered
     * texture over an element-read float array is rejected (see sift_octave.cu;
     * gfx1100 accepts it, so this is per device). The linear textures are
     * therefore created with point filtering and the bilinear filter is done
     * here in software, which keeps one build correct on either device.
     * CUDA's unnormalized linear filter on tex2DLayered(c) samples at index c-0.5,
     * i.e. i0=floor(c-0.5) with weight frac=(c-0.5)-i0; readTex passes c=x+0.5,
     * so i0=floor(x), frac=x-floor(x). Point textures (used with integer x,y) hit
     * frac==0 and reduce to the exact texel, so this path is correct for them too.
     * SIFT blurs are per-layer; z is always an integer level, so interpolate in x,y
     * at the fixed layer z. Point-filtered fetch of texel ix is tex2DLayered(ix+0.5).
     */
    const float fx = floorf( x );
    const float fy = floorf( y );
    const float ax = x - fx;
    const float ay = y - fy;
    const float t00 = tex2DLayered<float>( tex, fx + 0.5f, fy + 0.5f, z );
    const float t10 = tex2DLayered<float>( tex, fx + 1.5f, fy + 0.5f, z );
    const float t01 = tex2DLayered<float>( tex, fx + 0.5f, fy + 1.5f, z );
    const float t11 = tex2DLayered<float>( tex, fx + 1.5f, fy + 1.5f, z );
    const float top = fmaf( ax, t10 - t00, t00 );
    const float bot = fmaf( ax, t11 - t01, t01 );
    return fmaf( ay, bot - top, top );
#else
    return tex2DLayered<float>( tex, x+0.5f, y+0.5f, z );
#endif
}

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
/* Fetch one texel of the pyramid array through its surface, emulating a
 * point-filtered texture fetch: the integer coordinates are clamped into
 * [0,width-1]x[0,height-1] the way cudaAddressModeClamp clamps the textures this
 * read replaces (an out-of-range surface read returns 0, which would corrupt the
 * image borders), and the byte offset of the surface x coordinate is hidden. The
 * z coordinate is the blur level, i.e. the layer index of the CUDA build.
 */
__device__ static inline
float texFetchClamped( const LayeredReadTex& s, int ix, int iy, int layer )
{
    ix = std::clamp( ix, 0, s.width  - 1 );
    iy = std::clamp( iy, 0, s.height - 1 );
    float v;
    surf3Dread( &v, s.surf, ix * 4, iy, layer );
    return v;
}

__device__ static inline
float readTex( const LayeredReadTex& s, float x, float y, float z )
{
    /* Same -0.5 texel-center bilinear convention as the texture readTex above,
     * but sampling the 3D surface (see sift_textures.h). SIFT blurs are
     * per-level, so z is always an integer level: interpolate in x,y at the
     * fixed z slice, matching the CUDA layered point/linear texture fetch.
     */
    const int   layer = (int)z;
    const float fx = floorf( x );
    const float fy = floorf( y );
    const float ax = x - fx;
    const float ay = y - fy;
    const int   ix = (int)fx;
    const int   iy = (int)fy;
    const float t00 = texFetchClamped( s, ix,   iy,   layer );
    const float t10 = texFetchClamped( s, ix+1, iy,   layer );
    const float t01 = texFetchClamped( s, ix,   iy+1, layer );
    const float t11 = texFetchClamped( s, ix+1, iy+1, layer );
    const float top = fmaf( ax, t10 - t00, t00 );
    const float bot = fmaf( ax, t11 - t01, t01 );
    return fmaf( ay, bot - top, top );
}
#endif

__device__ static inline
float readTex( cudaTextureObject_t tex, float x, float y )
{
    return tex2D<float>( tex, x+0.5f, y+0.5f );
}

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


__host__
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
