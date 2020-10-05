/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/vec_macros.h"
#include "s_desc_vlfeat.h"
#include "s_gradiant.h"
#include "sift_constants.h"

#include <cstdio>

using namespace popsift;

__device__ static inline
void ext_desc_vlfeat_sub( const float         ang,
                        const Extremum*     ext,
                        float* __restrict__ features,
                        cudaTextureObject_t layer_tex,
                        const int           width,
                        const int           height )
{
// #ifndef BLOCK_3_DIMS
//     const int tile_x      = threadIdx.y;
//     const int tile_y      = threadIdx.z;
//     const int tile_offset = ( ( ( tile_y << 2 ) + tile_x ) << 3 ); // base of the 8 floats written by this group of 16 threads
// #else
//     const int tile_x      = ( threadIdx.z &  0x3 );
//     const int tile_y      = ( threadIdx.z >> 2 );
//     const int tile_offset = ( threadIdx.z << 3 );
// #endif

    const float x     = ext->xpos;
    const float y     = ext->ypos;
    const int   level = ext->lpos; // old_level;
    const float sig   = ext->sigma;
    const float SBP   = fabsf(DESC_MAGNIFY * sig);
    // printf( "Keypoint at (%f,%f) angle %f sigma %f\n", x, y, ang*180.0f/M_PI, sig );

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
    const float crsbp = cos_t / SBP;
    const float srsbp = sin_t / SBP;

    // const float2 offsetpt = make_float2( tile_x - 1.5f, tile_y - 1.5f );

    // The following 2 lines were the primary bottleneck of this kernel
    // // const float ptx = csbp * offsetptx - ssbp * offsetpty + x;
    // // const float pty = csbp * offsetpty + ssbp * offsetptx + y;
    // const float ptx = ::fmaf( csbp, offsetpt.x, ::fmaf( -ssbp, offsetpt.y, x ));
    // const float pty = ::fmaf( csbp, offsetpt.y, ::fmaf(  ssbp, offsetpt.x, y ));

    // We have 4x4=16 bins.
    // There centers have the offsets -1.5, -0.5, 0.5, 1.5 from the
    // keypoint. The points that support them stretch from -2 to 2
    const float2 maxdist = make_float2( -2.0f, -2.0f );

    // We rotate the corner of the maximum range by the keypoint orientation.
    //
    const float ptx = fabsf( ::fmaf( csbp, maxdist.x, ::fmaf( -ssbp, maxdist.y, x )) );
    const float pty = fabsf( ::fmaf( csbp, maxdist.y, ::fmaf(  ssbp, maxdist.x, y ) ) );

    const float bsz = fabsf(csbp) + fabsf(ssbp);
    const int   xmin = max(1,          (int)floorf(x - ptx - bsz));
    const int   ymin = max(1,          (int)floorf(y - pty - bsz));
    const int   xmax = min(width - 2,  (int)floorf(x + ptx + bsz));
    const int   ymax = min(height - 2, (int)floorf(y + pty + bsz));

    // const int wx = xmax - xmin + 1;
    // const int hy = ymax - ymin + 1;
    // const int loops = wx * hy;

    float dpt[128];

    for( int i=0; i<128; i++ ) dpt[i] = 0.0f;

    for( int pix_y = ymin; pix_y <= ymax; pix_y++ )
    {
      for( int pix_x = xmin; pix_x <= xmax; pix_x++ )
      {
        // d : distance from keypoint
        const float2 d = make_float2( pix_x - x, pix_y - y );

        // n : normalized distance from keypoint
        const float2 n = make_float2( ::fmaf( crsbp, d.x,  srsbp * d.y ),
                                      ::fmaf( crsbp, d.y, -srsbp * d.x ) ); 

        // nn : abs value of normalized distance from keypoint
        const float2 nn = abs(n);

        if (nn.x < 2.0f && nn.y < 2.0f)
        {
            float mod;
            float th;
// printf("reading pixel (%d,%d)\n", pix_x, pix_y);
            get_gradiant( mod, th, pix_x, pix_y, layer_tex, level );

            mod /= 2; // Our mod is double that of vlfeat. Huh.

            // const float2 dn = n + offsetpt;

            // weight: (dn.x^2 * dn.y^2) / (2 * windowSize^2), windowSize==2
            // const float ww  = __expf(-0.125f * (dnx*dnx + dny*dny)); // speedup !
            // note that this is determined by a parameter in case of VLFeat, but default is the same
            const float  ww = __expf( -scalbnf(n.x*n.x + n.y*n.y, -3));
// WEIGHT: ww is identical to win in VLFeat
// printf("DESC: mod: %.2f (gradient strength) ori: %.2f angle: %.2f diff: %.5f sample weight: %.2f\n", mod, ang*180.0f/M_PI, th*180.0f/M_PI, (th-ang)*180.0f/M_PI, ww);


            th -= ang;
            // th += ( th <  0.0f  ? M_PI2 : 0.0f ); //  if (th <  0.0f ) th += M_PI2;
            // th -= ( th >= M_PI2 ? M_PI2 : 0.0f ); //  if (th >= M_PI2) th -= M_PI2;
            while( th > M_PI2 ) th -= M_PI2;
            while( th < 0.0f  ) th += M_PI2;
            const float nt = 8.0f * th / M_PI2; // was 9.0f instead of 8.0f ???
// printf("DESC: th: %.2f nt: %.2f\n", th*180.0f/M_PI, nt );

            // neighbouring tile on the lower side: -2, -1, 0 or 1
            // (must use floorf because casting rounds towards zero
            const int3 t0 = make_int3( (int)floorf(n.x - 0.5f),  // binx
                                       (int)floorf(n.y - 0.5f),  // biny
                                       (int)nt );                // bint
// printf("DESC: the base is (X,Y,rot)=(%d,%d,%d)\n", t0.x, t0.y, t0.z );
            float3 wgt[2];
            wgt[1] = make_float3( - ( n.x - ( t0.x + 0.5f ) ),  // - rbinx
                                  - ( n.y - ( t0.y + 0.5f ) ),  // - rbiny
                                  - ( nt  - t0.z ) );           // - rbint
            wgt[0] = make_float3( 1.0f + wgt[1].x,
                                  1.0f + wgt[1].y,
                                  1.0f + wgt[1].z );
            for( int tx : { 0, 1 } )
                for( int ty : { 0, 1 } )
                    for( int tt : { 0, 1 } )
                    {
                        if( ( t0.y + ty < -2 ) ||
                            ( t0.x + tx < -2 ) ||
                            ( t0.y + ty > 1 ) ||
                            ( t0.x + tx > 1 ) )
                        {
// printf("DESC: drop bin (X,Y,rot)=(%d,%d,%d)\n", t0.x + tx, t0.y + ty, (t0.z + tt ) % 8 );
                        }
                        else
                        {
                            float val = ww
                                      * mod
                                      * fabsf( wgt[tx].x )
                                      * fabsf( wgt[ty].y )
                                      * fabsf( wgt[tt].z );

// printf("DESC: put %.2f into bin (X,Y,rot)=(%d,%d,%d)\n", val, t0.x + tx, t0.y + ty, (t0.z + tt)%8 );
                            dpt[ 80
                                 + ( t0.y + ty ) * 32
                                 + ( t0.x + tx ) * 8
                                 + ( t0.z + tt ) % 8 ] += val;
// printf( "DESC:       at %d = 80 + X %d*%d + Y %d*%d + Z %d*%d\n",
//          80 + ( t0.y + ty ) * 32 + ( t0.x + tx ) * 8 + ( t0.z + tt ) % 8,
//          8, t0.x + tx,
//          32, t0.y + ty,
//          1, ( t0.z + tt ) % 8 );
                        }
                    }
        }
      }
    }

    for (int i = 0; i < 128; i++)
    {
        features[i] = dpt[i];
    }
}

__global__ void ext_desc_vlfeat(int octave, cudaTextureObject_t layer_tex, int w, int h)
{           
    const int   o_offset =  dct.ori_ps[octave] + blockIdx.x;
    Descriptor* desc     = &dbuf.desc           [o_offset];
    const int   ext_idx  =  dobuf.feat_to_ext_map[o_offset];
    Extremum*   ext      =  dobuf.extrema + ext_idx;

    const int   ext_base =  ext->idx_ori;
    const int   ori_num  =  o_offset - ext_base;
    const float ang      =  ext->orientation[ori_num];

    ext_desc_vlfeat_sub( ang,
                       ext,
                       desc->features,
                       layer_tex,
                       w,
                       h );
}

__global__ void ext_desc_vlfeat_print(int octave, cudaTextureObject_t layer_tex, int w, int h)
{
    int max_num = dct.ori_ct[octave];
    for( int num=0; num<max_num; num++ )
    {
        int         off      =  dct.ori_ps[octave] + num;
        // Descriptor* desc     = &dbuf.desc[off];
        const int   ext_idx  =  dobuf.feat_to_ext_map[off];
        Extremum*   ext      =  dobuf.extrema + ext_idx;
        const float x    = ext->xpos;
        const float y    = ext->ypos;
        const int   level = ext->lpos; // old_level;
        printf( "Extracting for (%f,%f) at level %d\n", x, y, level );
    }
}

