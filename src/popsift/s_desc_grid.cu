/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/vec_macros.h"
#include "s_desc_grid.h"
#include "s_gradiant.h"
#include "sift_constants.h"

#include <cstdio>

using namespace popsift;

__device__ static inline
void ext_desc_grid_sub( const int           ix,
                        const int           iy,
                        const float         ang,
                        const Extremum*     ext,
                        float* __restrict__ features,
                        cudaTextureObject_t layer_tex )
{
    const int tile = ( ( ( iy << 2 ) + ix ) << 3 ); // base of the 8 floats written by this group of 16 threads

    const float x     = ext->xpos;
    const float y     = ext->ypos;
    const float sig   = ext->sigma;
    const int   level = ext->lpos; // old_level;
    const float SBP   = fabsf(DESC_MAGNIFY * sig);

    if( SBP == 0 ) {
        return;
    }

    float cos_t;
    float sin_t;
    __sincosf( ang, &sin_t, &cos_t );

    const float csbp  = cos_t * SBP;
    const float ssbp  = sin_t * SBP;

    const float2 offset = make_float2( ix - 1.5f, iy - 1.5f );

    // The following 2 lines were the primary bottleneck of this kernel
    // const float ptx = csbp * offsetptx - ssbp * offsetpty + x;
    // const float pty = csbp * offsetpty + ssbp * offsetptx + y;
    // const float ptx = ::fmaf( csbp, offsetptx, ::fmaf( -ssbp, offsetpty, x ) );
    // const float pty = ::fmaf( csbp, offsetpty, ::fmaf(  ssbp, offsetptx, y ) );
    const float2 pt = make_float2( ::fmaf( csbp, offset.x, ::fmaf( -ssbp, offset.y, x ) ),
                                   ::fmaf( csbp, offset.y, ::fmaf(  ssbp, offset.x, y ) ) );

    float dpt[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    // const float2 rgt_up = make_float2(  cos_t - sin_t,  cos_t + sin_t );
    // const float2 lft_up = make_float2( -cos_t - sin_t,  cos_t - sin_t );
    // const float2 rgt_dn = make_float2(  cos_t + sin_t, -cos_t + sin_t );
    const float2 lft_dn = make_float2( -cos_t + sin_t, -cos_t - sin_t );
    // const float2 rgt_stp = ( rgt_dn - lft_dn ) / 16.0f;
    // const float2 up__stp = ( lft_up - lft_dn ) / 16.0f;
    const float2 rgt_stp = make_float2(  cos_t, sin_t ) / 8.0f;
    const float2 up__stp = make_float2( -sin_t, cos_t ) / 8.0f;

    int xd = threadIdx.x;
    for( int yd=0; yd<16; yd++ )
    {
        float2 pixo = lft_dn + (xd+0.5f) * rgt_stp + (yd+0.5f) * up__stp;
        float2 pix  = pixo * SBP;
        pix = round( pt + pix ) - pt;
        pixo = pix / SBP;

        float mod;
        float th;
        get_gradiant( mod, th, (pt+pix).x, (pt+pix).y, layer_tex, level );

        const float2 norm_pix = make_float2( ::fmaf( cos_t, pixo.x,  sin_t * pixo.y ),
                                             ::fmaf( cos_t, pixo.y, -sin_t * pixo.x ) );

        const float2 dn  = norm_pix + offset;
        const float  ww  = expf( -scalbnf(dn.x*dn.x + dn.y*dn.y, -3)); // expf(-0.125f * (dnx*dnx + dny*dny));
        const float2 w   = make_float2( 1.0f - fabsf(norm_pix.x),
                                        1.0f - fabsf(norm_pix.y) );

        if( w.x < 0.0f || w.y < 0.0f ) continue;

        const float  wgt = ww * w.x * w.y * mod;

        th -= ang;
        th += ( th <  0.0f  ? M_PI2 : 0.0f ); //  if (th <  0.0f ) th += M_PI2;
        th -= ( th >= M_PI2 ? M_PI2 : 0.0f ); //  if (th >= M_PI2) th -= M_PI2;

        const float tth  = __fmul_ru( th, M_4RPI ); // th * M_4RPI;
        const int   fo0  = (int)floorf(tth);
        const float do0  = tth - fo0;             
        const float wgt1 = 1.0f - do0;
        const float wgt2 = do0;

        int fo  = fo0 % DESC_BINS;
        dpt[fo]   = __fmaf_ru( wgt1, wgt, dpt[fo] );   // dpt[fo]   += (wgt1*wgt);
        dpt[fo+1] = __fmaf_ru( wgt2, wgt, dpt[fo+1] ); // dpt[fo+1] += (wgt2*wgt);
    }
    __syncthreads();

    dpt[0] += dpt[8];

    /* reduction here */
    for (int i = 0; i < 8; i++) {
        // dpt[i] += popsift::shuffle_down( dpt[i], 16 );
        dpt[i] += popsift::shuffle_down( dpt[i], 8, 16 );
        dpt[i] += popsift::shuffle_down( dpt[i], 4, 16 );
        dpt[i] += popsift::shuffle_down( dpt[i], 2, 16 );
        dpt[i] += popsift::shuffle_down( dpt[i], 1, 16 );
        dpt[i]  = popsift::shuffle     ( dpt[i], 0, 16 );
    }


    if( threadIdx.x < 8 ) {
        features[tile+threadIdx.x] = dpt[threadIdx.x];
    }
}

__global__ void ext_desc_grid(int octave, cudaTextureObject_t layer_tex)
{
    const int   o_offset =  dct.ori_ps[octave] + blockIdx.x;
    const int   ix       = threadIdx.y;
    const int   iy       = threadIdx.z;

    Descriptor* desc     = &dbuf.desc           [o_offset];
    const int   ext_idx  =  dobuf.feat_to_ext_map[o_offset];
    Extremum*   ext      =  dobuf.extrema + ext_idx;

    const int   ext_base =  ext->idx_ori;
    const int   ori_num  =  o_offset - ext_base;
    const float ang      =  ext->orientation[ori_num];

    ext_desc_grid_sub( ix,
                       iy,
                       ang,
                       ext,
                       desc->features,
                       layer_tex );
}

