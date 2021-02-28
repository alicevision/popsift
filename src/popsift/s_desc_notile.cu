/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/vec_macros.h"
#include "s_desc_notile.h"
#include "s_gradiant.h"
#include "sift_constants.h"

#include <cstdio>
#include <iostream>

//   1    -> 19.6 on 980 Ti
//   2    -> 19.5 on 980 Ti
//   3    -> 20.3 on 980 Ti
//   4    -> 19.6 on 980 Ti
//   8    -> 19.7 on 980 Ti

#define BLOCK_Z_NOTILE 1

using namespace popsift;

__device__
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

    for( int i=0; i<8; ++i)
    {
        dpt[i] += popsift::shuffle_down( dpt[i], 4, 8 ); // add n+4
        dpt[i] += popsift::shuffle_down( dpt[i], 2, 8 ); // add n+2
        dpt[i] += popsift::shuffle_down( dpt[i], 1, 8 ); // add n+1
        dpt[i]  = popsift::shuffle     ( dpt[i], 0, 8 ); // move 0 to all
    }

    __syncthreads();

    features[out_y * 32 + threadIdx.x] = dpt[in_x];
}

__global__
// __launch_bounds__(384) // 56/threads
// __launch_bounds__(192) // 56/threads
// no -- __launch_bounds__(128) // 63/thread
// no -- no launch bound // 64/thread/thread
void ext_desc_notile( ExtremaBuffers*     buf,
                      const int           ori_count,
                      const int           ori_base_index,
                      cudaTextureObject_t texLinear )
{
    const int   offset   = blockIdx.x * BLOCK_Z_NOTILE + threadIdx.z;

    const int   o_offset =  ori_base_index + offset;
    if( offset >= ori_count ) return;

    Descriptor* desc     = &buf->desc[o_offset];
    const int   ext_idx  =  buf->feat_to_ext_map[o_offset];
    Extremum*   ext      =  buf->extrema + ext_idx;

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

namespace popsift
{

bool start_ext_desc_notile( const ExtremaCounters* ct, ExtremaBuffers* buf, int octave, Octave& oct_obj )
{
    dim3 block;
    dim3 grid;

    block.x = 32;
    block.y = 4;
    block.z = BLOCK_Z_NOTILE;

    grid.x = grid_divide( ct->ori_ct[octave], block.z );
    grid.y = 1;
    grid.z = 1;

    if( grid.x == 0 ) return false;

    ext_desc_notile
        <<<grid,block,0,oct_obj.getStream()>>>
        ( buf,
          ct->ori_ct[octave],
          ct->getOrientationBase( octave ),
          oct_obj.getDataTexLinear( ).tex );
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError( );
    if( err != cudaSuccess ) {
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                  << "    cudaGetLastError failed: " << cudaGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }

    POP_SYNC_CHK;

    return true;
}

}; // namespace popsift

