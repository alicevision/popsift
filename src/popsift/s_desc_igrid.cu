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
#include "s_desc_igrid.h"
#include "common/assist.h"
#include "common/vec_macros.h"

using namespace popsift;

__device__ static inline
void ext_desc_igrid_sub( const float x, const float y, const int level,
                         const float cos_t, const float sin_t, const float SBP,
                         const Extremum*     ext,
                         float* __restrict__ features,
                         cudaTextureObject_t texLinear )
{
    const int ix   = threadIdx.y & 3;
    const int iy   = threadIdx.y / 4;
    const int tile = ( ( ( iy << 2 ) + ix ) << 3 ); // base of the 8 floats written by this group of 16 threads


    float dpt[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    int xd = threadIdx.x;
    for( int yd=0; yd<16; yd++ )
    {
        const float stepx = ix - 2.5f + 1.0f / 16.0f + xd/8.0f;
        const float stepy = iy - 2.5f + 1.0f / 16.0f + yd/8.0f;
        const float ptx   = cos_t * stepx + -sin_t * stepy;
        const float pty   = cos_t * stepy +  sin_t * stepx;

        float mod;
        float th;
        get_gradiant( mod, th, x + ptx * SBP, y + pty * SBP, cos_t, sin_t, texLinear, level );
        th += ( th <  0.0f  ? M_PI2 : 0.0f ); //  if (th <  0.0f ) th += M_PI2;
        th -= ( th >= M_PI2 ? M_PI2 : 0.0f ); //  if (th >= M_PI2) th -= M_PI2;

        const float ww = d_consts.desc_gauss[iy*8+yd][ix*8+xd];
        const float wx = d_consts.desc_tile[xd];
        const float wy = d_consts.desc_tile[yd];

        const float  wgt = ww * wx * wy * mod;

        const float tth  = __fmul_ru( th, M_4RPI ); // th * M_4RPI;
        const int   fo   = (int)floorf(tth);
        const float do0  = tth - fo;
        const float wgt2 = do0;
        const int   fo1  = (fo+1) & 7; // % 8
        dpt[fo1] = dpt[fo1] + wgt * wgt2; 

        const float wgt1 = 1.0f - do0;
        const int   fo0  =  fo    & 7; // % 8
        dpt[fo0] = dpt[fo0] + wgt * wgt1; 
    }

    /* reduction here */
    for (int i = 0; i < 8; i++) {
        dpt[i] += popsift::shuffle_xor( dpt[i], 1, 16 );
        dpt[i] += popsift::shuffle_xor( dpt[i], 2, 16 );
        dpt[i] += popsift::shuffle_xor( dpt[i], 4, 16 );
        dpt[i] += popsift::shuffle_xor( dpt[i], 8, 16 );
    }

    if( threadIdx.x < 8 ) {
        features[tile+threadIdx.x] = dpt[threadIdx.x];
    }
}

__global__
void ext_desc_igrid( const int           octave,
                     cudaTextureObject_t texLinear )
{
    const int   num      = dct.ori_ct[octave];

    const int   offset   = blockIdx.x * blockDim.z + threadIdx.z;
    const int   o_offset =  dct.ori_ps[octave] + offset;
    if( offset >= num ) return;

    Descriptor* desc     = &dbuf.desc           [o_offset];
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

    ext_desc_igrid_sub( ext->xpos, ext->ypos, ext->lpos,
                        cos_t, sin_t, SBP,
                        ext,
                        desc->features,
                        texLinear );
}

