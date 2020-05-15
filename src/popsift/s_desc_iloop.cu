/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/vec_macros.h"
#include "s_desc_iloop.h"
#include "s_gradiant.h"
#include "sift_constants.h"

#include <cstdio>

using namespace popsift;

__device__ static inline
void ext_desc_iloop_sub( const float         ang,
                        const Extremum*     ext,
                        float* __restrict__ features,
                        cudaTextureObject_t layer_tex,
                        const int           width,
                        const int           height )
{
    const int ix   = ( threadIdx.z &  0x3 );
    const int iy   = ( threadIdx.z >> 2 );
    const int tile = ( threadIdx.z << 3 );

    const float x    = ext->xpos;
    const float y    = ext->ypos;
    const int   level = ext->lpos; // old_level;
    const float sig  = ext->sigma;
    const float SBP  = fabsf(DESC_MAGNIFY * sig);

    if( SBP == 0 ) {
        return;
    }

    // const float cos_t = cosf(ang);
    // const float sin_t = sinf(ang);
    float cos_t;
    float sin_t;
    __sincosf( ang, &sin_t, &cos_t );

    const float csbp  = cos_t * SBP;
    const float ssbp  = sin_t * SBP;
    // const float crsbp = cos_t / SBP;
    // const float srsbp = sin_t / SBP;

    const float2 offsetpt = make_float2( ix - 1.5f,
                                         iy - 1.5f );

    // The following 2 lines were the primary bottleneck of this kernel
    // const float ptx = csbp * offsetptx - ssbp * offsetpty + x;
    // const float pty = csbp * offsetpty + ssbp * offsetptx + y;
    const float ptx = ::fmaf( csbp, offsetpt.x, -ssbp * offsetpt.y );
    const float pty = ::fmaf( csbp, offsetpt.y,  ssbp * offsetpt.x );

    const float bsz = fabsf(cos_t) + fabsf(sin_t);

    float dpt[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    const int j = threadIdx.x;
    for( int i = 0; i<32; i++ )
    {
        const float dx = ( - bsz + j * bsz/16.0f );
        const float dy = ( - bsz + i * bsz/16.0f );

        const float2 n = make_float2( ::fmaf( cos_t, dx,  sin_t * dy ),
                                      ::fmaf( cos_t, dy, -sin_t * dx ) );
        const float2 nn = abs(n);
        if (nn.x < 1.0f && nn.y < 1.0f) {
            const float jj = x + ptx + dx * SBP;
            const float ii = y + pty + dy * SBP;

            float mod;
            float th;
#if 0
            get_gradiant( mod, th, jj, ii, layer_tex, level );
#else
            get_gradiant( mod, th, jj, ii, cos_t, sin_t, layer_tex, level );
#endif

            const float2 dn = n + offsetpt;
            const float  ww = __expf( -scalbnf(dn.x*dn.x + dn.y*dn.y, -3));
            // const float ww  = __expf(-0.125f * (dnx*dnx + dny*dny)); // speedup !
            const float2 w  = make_float2( 1.0f - nn.x,
                                           1.0f - nn.y );
            const float wgt = ww * w.x * w.y * mod;

#if 0
            th -= ang;
#endif
            th += ( th <  0.0f  ? M_PI2 : 0.0f ); //  if (th <  0.0f ) th += M_PI2;
            th -= ( th >= M_PI2 ? M_PI2 : 0.0f ); //  if (th >= M_PI2) th -= M_PI2;

            const float tth  = __fmul_ru( th, M_4RPI ); // th * M_4RPI;
            const int   fo0  = (int)floorf(tth);
            const float do0  = tth - fo0;             
            const float wgt1 = 1.0f - do0;
            const float wgt2 = do0;

            int fo  = fo0 % DESC_BINS;

                // maf: multiply-add
                // _ru - round to positive infinity equiv to froundf since always >=0
            dpt[fo]   = __fmaf_ru( wgt1, wgt, dpt[fo] );   // dpt[fo]   += (wgt1*wgt);
            dpt[fo+1] = __fmaf_ru( wgt2, wgt, dpt[fo+1] ); // dpt[fo+1] += (wgt2*wgt);
        }
        __syncthreads();
    }

    dpt[0] += dpt[8];

    /* reduction here */
    for (int i = 0; i < 8; i++) {
        dpt[i] += popsift::shuffle_down( dpt[i], 16 );
        dpt[i] += popsift::shuffle_down( dpt[i], 8 );
        dpt[i] += popsift::shuffle_down( dpt[i], 4 );
        dpt[i] += popsift::shuffle_down( dpt[i], 2 );
        dpt[i] += popsift::shuffle_down( dpt[i], 1 );
        dpt[i]  = popsift::shuffle     ( dpt[i], 0 );
    }

    if( threadIdx.x < 8 ) {
        features[tile+threadIdx.x] = dpt[threadIdx.x];
    }
}

__global__ void ext_desc_iloop(int octave, cudaTextureObject_t layer_tex, int w, int h)
{
    const int   o_offset =  dct.ori_ps[octave] + blockIdx.x;
    Descriptor* desc     = &dbuf.desc           [o_offset];
    const int   ext_idx  =  dobuf.feat_to_ext_map[o_offset];
    Extremum*   ext      =  dobuf.extrema + ext_idx;

    const int   ext_base =  ext->idx_ori;
    const int   ori_num  =  o_offset - ext_base;
    const float ang      =  ext->orientation[ori_num];

    ext_desc_iloop_sub( ang,
                        ext,
                        desc->features,
                        layer_tex,
                        w,
                        h );
}

