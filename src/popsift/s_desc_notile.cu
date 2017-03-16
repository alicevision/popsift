/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "sift_pyramid.h"
#include "s_desc_notile.h"

namespace popsift
{
__device__ static inline
void ext_desc_get_grad( const float                  x,
                        const float                  y,
                        const int                    level,
                        cudaTextureObject_t          texLinear,
                        const float                  cos_t,
                        const float                  sin_t,
                        const float                  SBP,
                        const int                    offx,
                        const int                    offy,
                        float&                       mod,
                        float&                       th )
{
    const float mvx = -2.5f + offx/8.0f + 1.0f/16.0f;
    const float mvy = -2.5f + offy/8.0f + 1.0f/16.0f;
    const float ptx  = ( cos_t * mvx - sin_t * mvy ) * SBP;
    const float pty  = ( cos_t * mvy + sin_t * mvx ) * SBP;
    get_gradiant( mod, th, x + ptx, y + pty, cos_t, sin_t, texLinear, level );
    th += ( th <  0.0f  ? M_PI2 : 0.0f ); //  if (th <  0.0f ) th += M_PI2;
    th -= ( th >= M_PI2 ? M_PI2 : 0.0f ); //  if (th >= M_PI2) th -= M_PI2;
}

__device__ static inline
void ext_desc_inc_tile( float* dpt, const int ix, const int iy, const int xd, const int yd, const float th, const float mod, const float ww )
{
    if( ix < 0 || iy < 0 || ix > 3 || iy > 3 ) return;

    const float wx = d_consts.desc_tile[xd];
    const float wy = d_consts.desc_tile[yd];

    const float  wgt = ww * wx * wy * mod;

    const float tth  = th * M_4RPI;
    const int   fo   = (int)floorf(tth);
    const float do0  = tth - fo;
    const float wgt1 = 1.0f - do0;
    const float wgt2 = do0;

    const int fo0  =   fo       & (8-1); // % 8;
    const int fo1  = ( fo + 1 ) & (8-1); // % 8;
    dpt[fo0] += ( wgt * wgt1 );
    dpt[fo1] += ( wgt * wgt2 );
}

__device__ inline
void ext_desc_notile_sub( const float                  x,
                          const float                  y,
                          const int                    level,
                          const float                  cos_t,
                          const float                  sin_t,
                          const float                  SBP,
                          const Extremum* __restrict__ ext,
                          float* __restrict__          features,
                          cudaTextureObject_t          texLinear )
{
    const int xd    = threadIdx.x & (8-1); // % 8 - xd 0..7
    const int block = threadIdx.x / 8;     //     - block 0..3

    float dpt[2][2][8];

    {
        int iy = 0;
        memset( dpt[0][0], 0, 8*sizeof(float) );
        memset( dpt[0][1], 0, 8*sizeof(float) );

        for( int yd = 0; yd < 8; yd ++ )
        {
            if( block == 0 )
            {
                const int ix = block;
                const int offx = ix*8+xd;
                const int offy = iy*8+yd;
                float mod, th;
                ext_desc_get_grad( x, y, level, texLinear, cos_t, sin_t, SBP, offx, offy, mod, th );
                float ww = d_consts.desc_gauss[offy][offx];

                ext_desc_inc_tile( dpt[0][0], ix,   iy,   xd,   yd,   th, mod, ww );
            }

            {
                const int ix = block + 1;
                const int offx = ix*8+xd;
                const int offy = iy*8+yd;
                float mod, th;
                ext_desc_get_grad( x, y, level, texLinear, cos_t, sin_t, SBP, offx, offy, mod, th );
                float ww = d_consts.desc_gauss[offy][offx];

                ext_desc_inc_tile( dpt[0][0], ix-1, iy,   xd+8, yd,   th, mod, ww );
                ext_desc_inc_tile( dpt[0][1], ix,   iy,   xd,   yd,   th, mod, ww );
            }
        }
    }
    /* until here, thread (block,xd) has written into [0][block..block+1][0..7] */

    for( int iy=1; iy<5; iy++ )
    {

        memset( dpt[iy&1?1:0][0], 0, 8*sizeof(float) );
        memset( dpt[iy&1?1:0][1], 0, 8*sizeof(float) );
        for( int yd = 0; yd<8; yd++ )
        {
            if( block == 0 )
            {
                const int ix = block;
                const int offx = ix*8+xd;
                const int offy = iy*8+yd;
                float mod, th;
                ext_desc_get_grad( x, y, level, texLinear, cos_t, sin_t, SBP, offx, offy, mod, th );
                float ww = d_consts.desc_gauss[offy][offx];

                ext_desc_inc_tile( dpt[iy&1?0:1][0], ix,   iy-1, xd,   yd+8, th, mod, ww );
                ext_desc_inc_tile( dpt[iy&1?1:0][0], ix,   iy,   xd,   yd,   th, mod, ww );
            }

            {
                const int ix = block + 1;
                const int offx = ix*8+xd;
                const int offy = iy*8+yd;
                float mod, th;
                ext_desc_get_grad( x, y, level, texLinear, cos_t, sin_t, SBP, offx, offy, mod, th );
                float ww = d_consts.desc_gauss[offy][offx];

                ext_desc_inc_tile( dpt[iy&1?0:1][0], ix-1, iy-1, xd+8, yd+8, th, mod, ww );
                ext_desc_inc_tile( dpt[iy&1?0:1][1], ix,   iy-1, xd,   yd+8, th, mod, ww );
                ext_desc_inc_tile( dpt[iy&1?1:0][0], ix-1, iy,   xd+8, yd,   th, mod, ww );
                ext_desc_inc_tile( dpt[iy&1?1:0][1], ix,   iy,   xd,   yd,   th, mod, ww );
            }
        }

        /* Until here, thread (block,xd) has written into [0..1][block..block+1][0..7]
         * This means that we waste comparisons in the following, since the range
         *     d[0..1][0..block-1][0..8] and
         *     d[0..1][block+2..3][0..8]
         * are actually empty.
         */

        for( int b=0; b<4; b++ )
        {
            for( int i=0; i<8; i++ ) {
                float d = (b==block) ? dpt[iy&1?0:1][0][i]
                                     : (b==block+1) ? dpt[iy&1?0:1][1][i]
                                                    : 0;
                d += __shfl_xor( d,  1 );
                d += __shfl_xor( d,  2 );
                d += __shfl_xor( d,  4 );
                d += __shfl_xor( d,  8 );
                d += __shfl_xor( d, 16 );
                if( threadIdx.x == block ) {
                    features[(iy-1)*32 + b*8 + i] = d;
                }
            }
        }

        __syncthreads();

        // features[(iy-1)*32+threadIdx.x] = sdpt[threadIdx.z][(iy-1)*32+threadIdx.x];
    }
}

/*
 * We assume that this is started with
 * block = 16,4,4 or with 32,4,4, depending on macros
 * grid  = nunmber of orientations
 */
__global__
void ext_desc_notile( const int octave,
                      cudaTextureObject_t texLinear )
{
    const int num      = dct.ori_ct[octave];
    const int offset   = blockIdx.x * blockDim.z + threadIdx.z;
    const int o_offset = dct.ori_ps[octave] + offset;
    if( offset >= num ) return;

    Descriptor* desc     = &dbuf.desc           [o_offset];
    const int   ext_idx  =  dobuf.feat_to_ext_map[o_offset];
    Extremum*   ext      =  dobuf.extrema + ext_idx;

    if( ext->sigma == 0 ) return;
    const float SBP   = fabsf(DESC_MAGNIFY * ext->sigma);

    const int   ext_base =  ext->idx_ori;
    const int   ori_num  =  o_offset - ext_base;
    const float ang      =  ext->orientation[ori_num];

    float cos_t;
    float sin_t;
    __sincosf( ang, &sin_t, &cos_t );

    ext_desc_notile_sub( ext->xpos, ext->ypos, ext->lpos,
                         cos_t, sin_t, SBP,
                         ext,
                         desc->features,
                         texLinear );
}

}; // namespace popsift
