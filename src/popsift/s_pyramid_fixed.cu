/*
 * Copyright 2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "sift_pyramid.h"
#include "sift_constants.h"
#include "gauss_filter.h"
#include "common/debug_macros.h"
#include "assist.h"
#include "common/clamp.h"

#include <iostream>
#include <stdio.h>

namespace popsift {

namespace gauss {

namespace fixedSpan {

template<int SPAN, int WIDTH, bool OCT_0>
__global__
void octave_fixed3( cudaTextureObject_t src_data,
                    Plane2D_float dst_data )
{
    const int w     = dst_data.getWidth();
    const int h     = dst_data.getHeight();
    const int level = threadIdx.z + 1;
    const int plane_rows = threadIdx.z * h;
    const int SHIFT = SPAN - 1;

    const float* filter = OCT_0 ? &d_gauss.abs_filter_o0[level*GAUSS_ALIGN]
                                : &d_gauss.abs_filter_oN[level*GAUSS_ALIGN];

    Plane2D_float destination( w, h,
                               dst_data.ptr( plane_rows ),
                               dst_data.getPitch() );

    const int idx = blockIdx.x * WIDTH      + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    float val  = tex2D<float>( src_data, idx-SHIFT+0.5f, idy+0.5f );
    float fval = val * filter[0];
    #pragma unroll
    for( int i=1; i<SPAN; i++ ) {
        val   = tex2D<float>( src_data, idx-SHIFT+0.5f, idy-i+0.5f )
              + tex2D<float>( src_data, idx-SHIFT+0.5f, idy+i+0.5f );
        fval += val * filter[i];
    }

    float out = fval * filter[0];
    #pragma unroll
    for( int i=1; i<SPAN; i++ ) {
        val  = __shfl_up( fval, i ) + __shfl_down( fval, i );
        out += val * filter[i];
    }
    val = __shfl_down( out, SHIFT );

    const bool i_write = ( threadIdx.x < WIDTH & idx < w && idy < h );

    if( i_write ) {
        destination.ptr(idy)[idx] = val;
    }
}

} // namespace fixedSpan

} // namespace gauss

template<int SPAN, bool OCT_0>
__host__
inline void Pyramid::make_octave_sub( Octave& oct_obj, cudaStream_t stream )
{
    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    const int x_size = 32;
    const int w_conf = x_size - 2 * (SPAN-1);
    const int h_conf = 1024 / ( x_size * (_levels-1) );
    dim3 block( x_size, h_conf, _levels-1 );
    dim3 grid;
    grid.x = grid_divide( width, w_conf );
    grid.y = grid_divide( height, block.y );

    assert( block.x * block.y * block.z < 1024 );

    gauss::fixedSpan::octave_fixed3
        <SPAN,w_conf,OCT_0>
        <<<grid,block,0,stream>>>
        ( oct_obj._data_tex[0], oct_obj.getData(1) );
}

void Pyramid::make_octave( const Config& conf, Octave& oct_obj, cudaStream_t stream, bool isOctaveZero )
{
    if( conf.getGaussMode() == Config::Fixed4 ) {
        if( isOctaveZero )
            make_octave_sub<4,true> ( oct_obj, stream );
        else
            make_octave_sub<4,false>( oct_obj, stream );
    } else if( conf.getGaussMode() == Config::Fixed8 ) {
        if( isOctaveZero )
            make_octave_sub<8,true> ( oct_obj, stream );
        else
            make_octave_sub<8,false>( oct_obj, stream );
    } else {
        POP_FATAL("Unsupported Gauss filter mode for making all octaves at once");
    }
}

} // namespace popsift

