/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/clamp.h"
#include "common/debug_macros.h"
#include "s_solve.h"
#include "sift_constants.h"
#include "sift_pyramid.h"

#include <cuda_runtime.h>
#include <texture_fetch_functions.h>

#include <cstdio>

namespace popsift{

template<int HEIGHT>
__device__ static inline
uint32_t extrema_count( unsigned int indicator, int* extrema_counter )
{
    uint32_t mask = popsift::ballot( indicator ); // bitfield of warps with results

    int ct = __popc( mask );          // horizontal reduce

    int write_index;
    if( threadIdx.x == 0 ) {
        // atomicAdd returns the old value, we consider this the based
        // index for this thread's write operation
        write_index = atomicAdd( extrema_counter, ct );
    }
    // broadcast from thread 0 to all threads in warp
    write_index = popsift::shuffle( write_index, 0 );

    // this thread's offset: count only bits below the bit of the own
    // thread index; this provides the 0 result and every result up to ct
    write_index += __popc( mask & ((1 << threadIdx.x) - 1) );

    return write_index;
}

__device__
static
inline void extremum_cmp( float val, float f, uint32_t& gt, uint32_t& lt, uint32_t mask )
{
    gt |= ( ( val > f ) ? mask : 0 );
    lt |= ( ( val < f ) ? mask : 0 );
}

#define TX(dx,dy,dz) readTex( obj, x+dx, y+dy, z+dz )

__device__
static
inline bool is_extremum( cudaTextureObject_t obj,
                         int x, int y, int z )
{
    uint32_t gt = 0;
    uint32_t lt = 0;

    const float val0 = TX( 0, 1, 1 );
    const float val2 = TX( 2, 1, 1 );
    const float val  = TX( 1, 1, 1 );

    // bit indeces for neighbours:
    //     7 0 1    0x80 0x01 0x02
    //     6   2 -> 0x40      0x04
    //     5 4 3    0x20 0x10 0x08
    // upper layer << 24 ; own layer << 16 ; lower layer << 8
    // 1st group: left and right neigbhour
    extremum_cmp( val, val0, gt, lt, 0x00400000 ); // ( 0x01<<6 ) << 16
    extremum_cmp( val, val2, gt, lt, 0x00040000 ); // ( 0x01<<2 ) << 16

    if( ( gt != 0x00440000 ) && ( lt != 0x00440000 ) ) return false;

    // 2nd group: requires a total of 8 128-byte reads
    extremum_cmp( val, TX(0,0,1), gt, lt, 0x00800000 ); // ( 0x01<<7 ) << 16
    extremum_cmp( val, TX(0,2,1), gt, lt, 0x00200000 ); // ( 0x01<<5 ) << 16
    extremum_cmp( val, TX(0,0,0), gt, lt, 0x80000000 ); // ( 0x01<<6 ) << 24
    extremum_cmp( val, TX(0,2,0), gt, lt, 0x40000000 ); // ( 0x01<<6 ) << 24
    extremum_cmp( val, TX(0,1,0), gt, lt, 0x20000000 ); // ( 0x01<<6 ) << 24
    extremum_cmp( val, TX(0,0,2), gt, lt, 0x00008000 ); // ( 0x01<<6 ) <<  8
    extremum_cmp( val, TX(0,1,2), gt, lt, 0x00004000 ); // ( 0x01<<6 ) <<  8
    extremum_cmp( val, TX(0,2,2), gt, lt, 0x00002000 ); // ( 0x01<<6 ) <<  8

    if( ( gt != 0xe0e4e000 ) && ( lt != 0xe0e4e000 ) ) return false;

    // 3rd group: remaining 2 cache misses in own layer
    extremum_cmp( val, TX(1,0,1), gt, lt, 0x00010000 ); // ( 0x01<<0 ) << 16
    extremum_cmp( val, TX(2,0,1), gt, lt, 0x00020000 ); // ( 0x01<<1 ) << 16
    extremum_cmp( val, TX(1,2,1), gt, lt, 0x00100000 ); // ( 0x01<<4 ) << 16
    extremum_cmp( val, TX(2,2,1), gt, lt, 0x00080000 ); // ( 0x01<<3 ) << 16

    if( ( gt != 0xe0ffe000 ) && ( lt != 0xe0ffe000 ) ) return false;

    // 4th group: 3 cache misses higher layer
    extremum_cmp( val, TX(1,0,0), gt, lt, 0x01000000 ); // ( 0x01<<0 ) << 24
    extremum_cmp( val, TX(2,0,0), gt, lt, 0x02000000 ); // ( 0x01<<1 ) << 24
    extremum_cmp( val, TX(1,1,0), gt, lt, 0x00000004 ); // ( 0x01<<2 )
    extremum_cmp( val, TX(2,1,0), gt, lt, 0x04000000 ); // ( 0x01<<2 ) << 24
    extremum_cmp( val, TX(1,2,0), gt, lt, 0x10000000 ); // ( 0x01<<4 ) << 24
    extremum_cmp( val, TX(2,2,0), gt, lt, 0x08000000 ); // ( 0x01<<3 ) << 24

    if( ( gt != 0xffffe004 ) && ( lt != 0xffffe004 ) ) return false;

    // 5th group: 3 cache misss lower layer
    extremum_cmp( val, TX(1,0,2), gt, lt, 0x00000100 ); // ( 0x01<<0 ) <<  8
    extremum_cmp( val, TX(2,0,2), gt, lt, 0x00000200 ); // ( 0x01<<1 ) <<  8
    extremum_cmp( val, TX(1,1,2), gt, lt, 0x00000001 ); // ( 0x01<<0 )
    extremum_cmp( val, TX(2,1,2), gt, lt, 0x00000400 ); // ( 0x01<<2 ) <<  8
    extremum_cmp( val, TX(1,2,2), gt, lt, 0x00001000 ); // ( 0x01<<4 ) <<  8
    extremum_cmp( val, TX(2,2,2), gt, lt, 0x00000800 ); // ( 0x01<<3 ) <<  8

    if( ( gt != 0xffffff05 ) && ( lt != 0xffffff05 ) ) return false;

    return true;
}

template<int sift_mode>
class ModeFunctions
{
public:
    inline __device__
    bool first_contrast_ok( float val ) const;

    /* refine
     * returns -1 : break loop and fail
     *          0 : continue looping
     *          1 : break loop and succeed
     */
    inline __device__
    int refine( float3& d, int3& n, int width, int height, int maxlevel, bool last_it );

    /*
     * returns true  : values after refine make sense
     *         false : they do not
     */
    inline __device__
    bool verify( float xn, float yn, float sn, int width, int height, int maxlevel ) const;
};

template<>
class ModeFunctions<Config::OpenCV>
{
public:
    inline __device__
    bool first_contrast_ok( float val ) const
    {
        return ( fabsf( val ) >= floorf( d_consts.threshold ) );
    }

    inline __device__
    int refine( float3& d, int3& n, int width, int height, int maxlevel, bool last_it ) const
    {
        // OpenCV mode is a special case because d remains unmodified.
        // Either we return 1, and n has not been modified.
        // Or we quit the loop by exceeding the limit, and reject the point anyway.

        const float3 t = make_float3( fabsf(d.x), fabsf(d.y), fabsf(d.z) );

        if( t.x < 0.5f && t.y < 0.5f && t.z < 0.5f ) {
            // return false, quit the loop, success
            return 1;
        }

        // This test in OpenCV is totally useless in CUDA because the thread
        // would simply idle before failing 7 instructions below anyway.
        // if( t.x > (float)(INT_MAX/3) || t.y > (float)(INT_MAX/3) || t.z > (float)(INT_MAX/3) ) {
            // return false, quit the loop, fail
            // return -1;
        // }

        n.x += roundf( d.x );  // choose rintf or roundf
        n.y += roundf( d.y );  // rintf is quicker, roundf is more exact
        n.z += roundf( d.z );

        const int retval = ( n.x < 5 || n.x >= width-5 ||
                             n.y < 5 || n.y >= height-5 ||
                             n.z < 1 || n.z > maxlevel-2 ) ? -1 : 0;
            // if outside of all DoG images (minus border),
            // quit the loop, fail

        return retval;
    }

    inline __device__
    bool verify( float xn, float yn, float sn, int width, int height, int maxlevel ) const
    {
        return true;
    }
};

template<>
class ModeFunctions<Config::VLFeat>
{
public:
    inline __device__
    bool first_contrast_ok( const float val ) const
    {
        return ( fabsf( val ) >= 0.8f * 2.0f * d_consts.threshold );
    }

    inline __device__
    int refine( float3& d, int3& n, int width, int height, int maxlevel, bool last_it ) const
    {
        if( last_it ) return 0;

        float2 t;

        t.x = ((d.x >= 0.6f && n.x < width-2) ?  1.0f : 0.0f )
            + ((d.x <= -0.6f && n.x > 1)? -1.0f : 0.0f );

        t.y = ((d.y >= 0.6f && n.y < height-2)  ?  1.0f : 0.0f )
            + ((d.y <= -0.6f && n.y > 1) ? -1.0f : 0.0f );

        // t.z = ((d.z >= 0.6f && n.z < maxlevel-1)  ?  1 : 0 )
            // + ((d.z <= -0.6f && n.z > 1) ? -1 : 0 );

        if( t.x == 0 && t.y == 0 ) {
            // no more changes
            return 1;
        }

        n.x += t.x;
        n.y += t.y;
        // n.z += t.z; - VLFeat is not changing levels !!!

        return 0;
    }

    inline __device__
    bool verify( float xn, float yn, float sn, int width, int height, int maxlevel ) const
    {
        // reject if outside of image bounds or far outside DoG bounds
        return ( ( xn < 0.0f ||
                   xn > width - 1.0f ||
                   yn < 0.0f ||
                   yn > height - 1.0f ||
                   sn < 0.0f ||
                   sn > maxlevel ) ? false
                                   : true );
    }
};

template<>
class ModeFunctions<Config::PopSift>
{
public:
    inline __device__
    bool first_contrast_ok( const float val ) const
    {
        return ( fabsf( val ) >= 1.6f * d_consts.threshold );
    }

    inline __device__
    int refine( float3& d, int3& n, int width, int height, int maxlevel, bool last_it ) const
    {
        if( last_it ) return 0;

        int3 t;

        t.x = ((d.x >=  0.6f && n.x < width-2) ?  1 : 0 )
            + ((d.x <= -0.6f && n.x > 1)       ? -1 : 0 );

        t.y = ((d.y >=  0.6f && n.y < height-2)  ?  1 : 0 )
            + ((d.y <= -0.6f && n.y > 1)         ? -1 : 0 );

        t.z = ((d.z >=  0.6f && n.z < maxlevel-1)  ?  1 : 0 )
            + ((d.z <= -0.6f && n.z > 1)           ? -1 : 0 );

        if( t.x == 0 && t.y == 0 && t.z == 0 ) {
            // no more changes
            return 1;
        }

        n.x += t.x;
        n.y += t.y;
        n.z += t.z;

        return 0;
    }

    inline __device__
    bool verify( float xn, float yn, float sn, int width, int height, int maxlevel ) const
    {
        // reject if outside of image bounds or far outside DoG bounds
        return ( ( xn < 0.0f ||
                   xn > width - 1.0f ||
                   yn < 0.0f ||
                   yn > height - 1.0f ||
                   sn < -0.0f ||
                   sn > maxlevel ) ? false
                                   : true );
    }
};

template<int sift_mode>
__device__ inline bool find_extrema_in_dog_sub(cudaTextureObject_t dog,
                                               int debug_octave,
                                               int width,
                                               int height,
                                               uint32_t maxlevel,
                                               float w_grid_divider,
                                               float h_grid_divider,
                                               int grid_width,
                                               InitialExtremum& ec)
{
    ec.xpos    = 0.0f;
    ec.ypos    = 0.0f;
    ec.lpos    = 0;
    ec.sigma   = 0.0f;

    /*
     * First consideration: extrema cannot be found on any outermost edge,
     * one pixel on the left, right, upper, lower edge will never qualify.
     * Also, the upper and lower DoG layer will never qualify. So there is
     * no reason for selecting any of those pixel for the center of a 3x3x3
     * region.
     * Instead, I use groups of 32xHEIGHT threads that read from a 34x34x3 area,
     * but implicitly, they fetch * 64xHEIGHT+2x3 floats (bad luck).
     * To find maxima, compare first on the left edge of the 3x3x3 cube, ie.
     * a 1x3x3 area. If the rightmost 2 threads of a warp (x==30 and 3==31)
     * are not extreme w.r.t. to the left slice, 8 fetch operations.
     */
    const int block_x = blockIdx.x * 32;
    const int block_y = blockIdx.y * blockDim.y;
    const int block_z = blockIdx.z;
    const int y       = block_y + threadIdx.y + 1;
    const int x       = block_x + threadIdx.x + 1;
    const int level   = block_z + 1;

    if( sift_mode == Config::OpenCV ) {
        if( x < 5 || y < 5 || x >= width-5 || y >= height-5 ) {
            return false;
        }
    }

    const float val = readTex( dog, x, y, level );

    ModeFunctions<sift_mode> f;
    if( ! f.first_contrast_ok( val ) ) return false;

    if( ! is_extremum( dog, x-1, y-1, level-1 ) ) {
        // if( debug_octave==0 && level==2 && x==14 && y==73 ) printf("But I fail\n");
        return false;
    }

    float3 D; // Dx Dy Ds
    float3 DD; // Dxx Dyy Dss
    float3 DX; // Dxy Dxs Dys
    float3 d; // dx dy ds

    float v = val;

    int3 n = make_int3( x, y, level ); // nj ni ns

    int32_t iter = 0;

#define MAX_ITERATIONS 5

    do {
        iter++;

        // const int z = level - 1;
        /* compute gradient */
        const float x2y1z1 = readTex( dog, n.x+1, n.y  , n.z   );
        const float x0y1z1 = readTex( dog, n.x-1, n.y  , n.z   );
        const float x1y2z1 = readTex( dog, n.x  , n.y+1, n.z   );
        const float x1y0z1 = readTex( dog, n.x  , n.y-1, n.z   );
        const float x1y1z2 = readTex( dog, n.x  , n.y  , n.z+1 );
        const float x1y1z0 = readTex( dog, n.x  , n.y  , n.z-1 );
        // D.x = 0.5f * ( x2y1z1 - x0y1z1 );
        // D.y = 0.5f * ( x1y2z1 - x1y0z1 );
        // D.z = 0.5f * ( x1y1z2 - x1y1z0 );
        D.x = scalbnf( x2y1z1 - x0y1z1, -1 );
        D.y = scalbnf( x1y2z1 - x1y0z1, -1 );
        D.z = scalbnf( x1y1z2 - x1y1z0, -1 );

        /* compute Hessian */
        const float x1y1z1 = readTex( dog, n.x  , n.y  , n.z   );
        // DD.x = x2y1z1 + x0y1z1 - 2.0f * x1y1z1;
        // DD.y = x1y2z1 + x1y0z1 - 2.0f * x1y1z1;
        // DD.z = x1y1z2 + x1y1z0 - 2.0f * x1y1z1;
        DD.x = x2y1z1 + x0y1z1 - scalbnf( x1y1z1, 1 );
        DD.y = x1y2z1 + x1y0z1 - scalbnf( x1y1z1, 1 );
        DD.z = x1y1z2 + x1y1z0 - scalbnf( x1y1z1, 1 );

        const float x0y0z1 = readTex( dog, n.x-1, n.y-1, n.z   );
        const float x0y1z0 = readTex( dog, n.x-1, n.y  , n.z-1 );
        const float x0y1z2 = readTex( dog, n.x-1, n.y  , n.z+1 );
        const float x0y2z1 = readTex( dog, n.x-1, n.y+1, n.z   );
        const float x1y0z0 = readTex( dog, n.x  , n.y-1, n.z-1 );
        const float x1y0z2 = readTex( dog, n.x  , n.y-1, n.z+1 );
        const float x1y2z0 = readTex( dog, n.x  , n.y+1, n.z-1 );
        const float x1y2z2 = readTex( dog, n.x  , n.y+1, n.z+1 );
        const float x2y0z1 = readTex( dog, n.x+1, n.y-1, n.z   );
        const float x2y1z0 = readTex( dog, n.x+1, n.y  , n.z-1 );
        const float x2y1z2 = readTex( dog, n.x+1, n.y  , n.z+1 );
        const float x2y2z1 = readTex( dog, n.x+1, n.y+1, n.z   );
        // DX.x = 0.25f * ( x2y2z1 + x0y0z1 - x0y2z1 - x2y0z1 );
        // DX.y = 0.25f * ( x2y1z2 + x0y1z0 - x0y1z2 - x2y1z0 );
        // DX.z = 0.25f * ( x1y2z2 + x1y0z0 - x1y2z0 - x1y0z2 );
        DX.x = scalbnf( x2y2z1 + x0y0z1 - x0y2z1 - x2y0z1, -2 );
        DX.y = scalbnf( x2y1z2 + x0y1z0 - x0y1z2 - x2y1z0, -2 );
        DX.z = scalbnf( x1y2z2 + x1y0z0 - x1y2z0 - x1y0z2, -2 );

        float3 b;
        float A[3][3];

        /* Solve linear system. */
        A[0][0] = DD.x;
        A[1][1] = DD.y;
        A[2][2] = DD.z;
        A[1][0] = A[0][1] = DX.x;
        A[2][0] = A[0][2] = DX.y;
        A[2][1] = A[1][2] = DX.z;

        b.x = -D.x;
        b.y = -D.y;
        b.z = -D.z;

        if(!solve(A, b)) {
            d.x = 0;
            d.y = 0;
            d.z = 0;
            break ;
        }

        d = b;

        /* If the translation of the keypoint is big, move the keypoint
         * and re-iterate the computation. Otherwise we are all set.
         */
        const int retval = f.refine( d, n, width, height, maxlevel, iter==MAX_ITERATIONS );

        if( retval == -1 ) {
            return false;
        } else if( retval == 1 ) {
            break;
        }
    }
    while( iter < MAX_ITERATIONS ); /* go to next iter */

    if (iter >= MAX_ITERATIONS) {
        if( sift_mode == Config::OpenCV ) {
            /* ensure convergence of interpolation */
            return false;
        }
    }

    if( sift_mode == Config::PopSift || sift_mode == Config::VLFeat ) {
        if( d.x >= 1.5f || d.y >= 1.5f || d.z >= 1.5f ) {
            // excessive pixel movement in at least dimension, reject
            return false;
        }
    }

    const float xn      = n.x + d.x;
    const float yn      = n.y + d.y;
    const float sn      = n.z + d.z;

    if( ! f.verify( xn, yn, sn, width, height, maxlevel ) ) {
        return false;
    }

    // float contr   = v + 0.5f * (D.x * d.x + D.y * d.y + D.z * d.z);
    const float contr   = v + scalbnf( D.x * d.x + D.y * d.y + D.z * d.z , -1 );
    const float tr      = DD.x + DD.y;
    const float det     = DD.x * DD.y - DX.x * DX.x;
    const float edgeval = tr * tr / det;

    // redundant check, verify() is stricter
    // if( sift_mode == Config::PopSift && iter >= MAX_ITERATIONS && ( sn<0 || sn>maxlevel) ) { return false; }

    /* negative determinant => curvatures have different signs -> reject it */
    if (det <= 0.0f) {
        return false;
    }

    /* accept-reject extremum */
    // if( fabsf(contr) < (d_consts.threshold*2.0f) )
    if( d_consts.threshold > 0.0f )
    {
        if( fabsf(contr) < scalbnf( d_consts.threshold, 1 ) )
        {
            return false;
        }
    }

    if( d_consts.edge_limit > 0.0f )
    {
        /* reject condition: tr(H)^2/det(H) < (r+1)^2/r */
        if( edgeval >= (d_consts.edge_limit+1.0f)*(d_consts.edge_limit+1.0f)/d_consts.edge_limit )
        {
            return false;
        }
    }

    ec.xpos      = xn;
    ec.ypos      = yn;
    ec.lpos      = (int)roundf(sn);
    ec.sigma     = d_consts.sigma0 * pow(d_consts.sigma_k, sn); // * 2;
    ec.cell      = floorf( yn / h_grid_divider ) * grid_width + floorf( xn / w_grid_divider );
        // const float sigma_k = powf(2.0f, 1.0f / levels );

    return true;
}


template<int HEIGHT, int sift_mode>
__global__
void find_extrema_in_dog( cudaTextureObject_t dog,
                          int                 octave,
                          int                 width,
                          int                 height,
                          const uint32_t      maxlevel,
                          int*                d_number_of_blocks,
                          int                 number_of_blocks,
                          const float         w_grid_divider,
                          const float         h_grid_divider,
                          const int           grid_width )
{
    InitialExtremum ec;
    ec.ignore = false;

    bool indicator = find_extrema_in_dog_sub<sift_mode>( dog,
                                                         octave,
                                                         width,
                                                         height,
                                                         maxlevel,
                                                         w_grid_divider,
                                                         h_grid_divider,
                                                         grid_width,
                                                         ec );

    uint32_t write_index = extrema_count<HEIGHT>( indicator, &dct.ext_ct[octave] );

    InitialExtremum* d_extrema = dobuf.i_ext_dat[octave];
    int*             d_ext_off = dobuf.i_ext_off[octave];

    if( indicator && write_index < d_consts.max_extrema ) {
        ec.write_index = write_index;
        // store the initial extremum in an array
        d_extrema[write_index] = ec;

        // index for indirect access to d_extrema, to enable
        // access after filtering some initial extrema
        d_ext_off[write_index] = write_index;
    }

    // without syncthreads, (0,0) threads may precede some calls to extrema_count()
    // in non-(0,0) threads and increase barrier count too early
    __syncthreads();

    if( threadIdx.x == 0 && threadIdx.y == 0 ) {
        int ct = atomicAdd( d_number_of_blocks, 1 );
        if( ct >= number_of_blocks-1 ) {
            int num_ext = atomicMin( &dct.ext_ct[octave], d_consts.max_extrema );
            // printf( "Block %d,%d,%d num ext %d\n", blockIdx.x, blockIdx.y, blockIdx.z, dct.ext_ct[octave] );
        }
    }
}

__host__
void Pyramid::find_extrema( const Config& conf )
{
    static const int HEIGHT = 4;

    for( int octave=0; octave<_num_octaves; octave++ ) {
        Octave&      oct_obj = _octaves[octave];

        int*  extrema_num_blocks = getNumberOfBlocks( octave );

        int cols = oct_obj.getWidth();
        int rows = oct_obj.getHeight();

        dim3 block( 32, HEIGHT );
        dim3 grid;
        grid.x  = grid_divide( cols, block.x );
        grid.y  = grid_divide( rows, block.y );
        grid.z  = _levels - 3;

        cudaStream_t oct_str = oct_obj.getStream();

        int*  num_blocks      = extrema_num_blocks;

#ifdef USE_DOG_TEX_LINEAR
#define getDogTexture getDogTextureLinear
#else
#define getDogTexture getDogTexturePoint
#endif
        switch( conf.getSiftMode() )
        {
        case Config::VLFeat :
                find_extrema_in_dog<HEIGHT,Config::VLFeat>
                    <<<grid,block,0,oct_str>>>
                    ( oct_obj.getDogTexture( ),
                      octave,
                      cols,
                      rows,
                      _levels-1,
                      num_blocks,
                      grid.x * grid.y,
                      oct_obj.getWGridDivider(),
                      oct_obj.getHGridDivider(),
                      conf.getFilterGridSize() );
                POP_SYNC_CHK;
                break;
        case Config::OpenCV :
                find_extrema_in_dog<HEIGHT,Config::OpenCV>
                    <<<grid,block,0,oct_str>>>
                    ( oct_obj.getDogTexture( ),
                      octave,
                      cols,
                      rows,
                      _levels-1,
                      num_blocks,
                      grid.x * grid.y,
                      oct_obj.getWGridDivider(),
                      oct_obj.getHGridDivider(),
                      conf.getFilterGridSize() );
                POP_SYNC_CHK;
                break;
        default :
                find_extrema_in_dog<HEIGHT,Config::PopSift>
                    <<<grid,block,0,oct_str>>>
                    ( oct_obj.getDogTexture( ),
                      octave,
                      cols,
                      rows,
                      _levels-1,
                      num_blocks,
                      grid.x * grid.y,
                      oct_obj.getWGridDivider(),
                      oct_obj.getHGridDivider(),
                      conf.getFilterGridSize() );
                POP_SYNC_CHK;
                break;
        }
#undef getDogTexture

        cuda::event_record( oct_obj.getEventExtremaDone(), oct_str, __FILE__, __LINE__ );
    }
}

} // namespace popsift

