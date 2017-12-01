/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <stdio.h>
#include <iso646.h>

#include "sift_constants.h"
#include "s_gradiant.h"
#include "s_desc_notile.h"
#include "common/assist.h"
#include "common/vec_macros.h"

using namespace popsift;

static const float stepbase =  - 2.5f + 1.0f / 16.0f;

__device__ static inline
void ext_desc_notile_sub( const float x, const float y, const int level,
                         const float cos_t, const float sin_t, const float SBP,
                         const Extremum*     ext,
                         float* __restrict__ features,
                         cudaTextureObject_t texLinear )
{
    float dpt[8] = { 0 };

    const int in_x  = threadIdx.x & 0x7;   // threadIdx.x % 8;
    const int out_y = threadIdx.y;

    for( int xoff = 0; xoff<2; xoff++ )
    {
        const int xd   = ( xoff << 3  )+ in_x;
        const int newx = ( xoff << 3 ) + threadIdx.x;

        for( int yoff = 0; yoff<2; yoff++ )
        {
            for( int in_y = 0; in_y<8; in_y++ )
            {
                const int   yd = ( yoff  << 3 ) + in_y;
                const int newy = ( out_y << 3 ) + yd; // out_y*8 + yd;

                const float wgt = d_consts.desc_tile[xd] * d_consts.desc_tile[yd];

                const float stepx = stepbase + scalbnf( newx, -3 ); //  newx/8.0f;
                const float stepy = stepbase + scalbnf( newy, -3 ); //  newy/8.0f;
                const float ptx   = cos_t * stepx + -sin_t * stepy;
                const float pty   = cos_t * stepy +  sin_t * stepx;
                float       mod;
                float       th;
                get_gradiant( mod, th, x + ptx * SBP, y + pty * SBP, cos_t, sin_t, texLinear, level );
                th += ( th <  0.0f  ? M_PI2 : 0.0f );
                th -= ( th >= M_PI2 ? M_PI2 : 0.0f );

                const float tth  = th * M_4RPI;
                const int   fo   = (int)floorf(th * M_4RPI);
                const float do0  = tth - fo;
                const int   fo0  = fo & 7; // % 8
                const int   fo1  = ( fo0 + 1 ) & 7;

                const float  ww   = d_consts.desc_gauss[newy][newx] * mod;
                const float2 owgt = make_float2( ( 1.0f - do0 ) * ww, do0 * ww );

                dpt[fo0] += ( wgt * owgt.x );
                dpt[fo1] += ( wgt * owgt.y );
            }
        }
    }

    for( int i=0; i<8; i++ )
    {
        dpt[i] += __shfl_down( dpt[i], 4, 8 ); // add n+4
        dpt[i] += __shfl_down( dpt[i], 2, 8 ); // add n+2
        dpt[i] += __shfl_down( dpt[i], 1, 8 ); // add n+1
        dpt[i]  = __shfl     ( dpt[i], 0, 8 ); // move 0 to all
    }

    __syncthreads();

    features[out_y * 32 + threadIdx.x] = dpt[in_x];
}

__global__
void ext_desc_notile( const int           octave,
                     cudaTextureObject_t texLinear )
{
    const int   num      = dct.ori_ct[octave];

    const int   offset   = blockIdx.x * BLOCK_Z_NOTILE + threadIdx.z;

    const int   o_offset =  dct.ori_ps[octave] + offset;
    if( offset >= num ) return;

    Descriptor* desc     = &dbuf.desc            [o_offset];
    const int   ext_idx  =  dobuf.feat_to_ext_map[o_offset];
    Extremum*   ext      =  dobuf.extrema + ext_idx;

    if( ext->sigma == 0 ) return;
    const float SBP      = fabsf( DESC_MAGNIFY * ext->sigma );

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

