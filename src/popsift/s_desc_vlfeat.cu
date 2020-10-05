/*
 * Copyright 2016-2017, Simula Research Laboratory
 *           2018-2020, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/debug_macros.h"
#include "common/vec_macros.h"
#include "s_desc_vlfeat.h"
#include "s_gradiant.h"
#include "sift_constants.h"
#include "sift_pyramid.h"

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

    if( SBP == 0 ) {
        return;
    }

    float cos_t;
    float sin_t;
    __sincosf( ang, &sin_t, &cos_t );

    const float csbp  = cos_t * SBP;
    const float ssbp  = sin_t * SBP;
    const float crsbp = cos_t / SBP;
    const float srsbp = sin_t / SBP;

    // We have 4x4*16 bins.
    // There centers have the offsets -1.5, -0.5, 0.5, 1.5 from the
    // keypoint. The points that support them stretch from -2 to 2
    const float2 maxdist = make_float2( -2.0f, -2.0f );

    // We rotate the corner of the maximum range by the keypoint orientation.
    const float ptx = fabsf( ::fmaf( csbp, maxdist.x, ::fmaf( -ssbp, maxdist.y, x )) );
    const float pty = fabsf( ::fmaf( csbp, maxdist.y, ::fmaf(  ssbp, maxdist.x, y ) ) );

    const float bsz = fabsf(csbp) + fabsf(ssbp);
    const int   xmin = max(1,          (int)floorf(x - ptx - bsz));
    const int   ymin = max(1,          (int)floorf(y - pty - bsz));
    const int   xmax = min(width - 2,  (int)floorf(x + ptx + bsz));
    const int   ymax = min(height - 2, (int)floorf(y + pty + bsz));

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

            get_gradiant( mod, th, pix_x, pix_y, layer_tex, level );

            mod /= 2; // Our mod is double that of vlfeat. Huh.

            const float  ww = __expf( -scalbnf(n.x*n.x + n.y*n.y, -3));

            th -= ang;
            while( th > M_PI2 ) th -= M_PI2;
            while( th < 0.0f  ) th += M_PI2;

            const float nt = 8.0f * th / M_PI2;

            // neighbouring tile on the lower side: -2, -1, 0 or 1
            // (must use floorf because casting rounds towards zero
            const int3 t0 = make_int3( (int)floorf(n.x - 0.5f),
                                       (int)floorf(n.y - 0.5f),
                                       (int)nt );
            float3 wgt[2];

            wgt[1] = make_float3( fabsf( n.x - ( t0.x + 0.5f ) ),
                                  fabsf( n.y - ( t0.y + 0.5f ) ),
                                  fabsf( nt  - t0.z ) );
            wgt[0] = make_float3( fabsf( 1.0f - wgt[1].x ),
                                  fabsf( 1.0f - wgt[1].y ),
                                  fabsf( 1.0f - wgt[1].z ) );

            for( int tx : { 0, 1 } )
            {
                for( int ty : { 0, 1 } )
                {
                    for( int tt : { 0, 1 } )
                    {
                        if( ( t0.y + ty >= -2 ) ||
                            ( t0.x + tx >= -2 ) ||
                            ( t0.y + ty < 2 ) ||
                            ( t0.x + tx < 2 ) )
                        {
                            float val = ww
                                      * mod
                                      * wgt[tx].x
                                      * wgt[ty].y
                                      * wgt[tt].z;

                            dpt[ 80
                                 + ( t0.y + ty ) * 32
                                 + ( t0.x + tx ) * 8
                                 + ( t0.z + tt ) % 8 ] += val;
                        }
                    }
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

namespace popsift
{

bool start_ext_desc_vlfeat( const int octave, Octave& oct_obj )
{
    dim3 block;
    dim3 grid;
    grid.x = hct.ori_ct[octave];
    grid.y = 1;
    grid.z = 1;

    if( grid.x == 0 ) return false;

    block.x = 1;
    block.y = 1;
    block.z = 1;

    ext_desc_vlfeat
        <<<grid,block,0,oct_obj.getStream()>>>
        ( octave,
          oct_obj.getDataTexPoint( ),
          oct_obj.getWidth(),
          oct_obj.getHeight() );

    POP_SYNC_CHK;

    return true;
}

}; // namespace popsift

