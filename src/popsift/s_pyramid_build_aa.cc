/*
 * Copyright 2016-2017, Simula Research Laboratory
 *           2018-2024, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "gauss_filter.h"
#include "sift_pyramid.h"
#include "sift_constants.h"

namespace popsift {
namespace absoluteSource {

__global__ static void horiz(cudaTextureObject_t src_point_texture, cudaSurfaceObject_t dst_data, int dst_level)
{
    const int    src_level = dst_level - 1;
    const int    span      =  d_gauss.inc.span[dst_level];
    const float* filter    = &d_gauss.inc.filter[dst_level*GAUSS_ALIGN];
    const int    block_x   = blockIdx.x * blockDim.x;
    const int    block_y   = blockIdx.y * blockDim.y;
    const int    xpos      = block_x + threadIdx.x;
    const int    ypos      = block_y + threadIdx.y;

    int   idx;
    float g;
    float val;
    float out = 0.0f;

    for( int offset = span; offset>0; offset-- ) {
        g  = filter[offset];

        idx = xpos - offset;
        val = readTex( src_point_texture, idx, ypos, src_level );
        out += ( val * g );

        idx = xpos + offset;
        val = readTex( src_point_texture, idx, ypos, src_level );
        out += ( val * g );
    }

    g  = filter[0];
    val = readTex( src_point_texture, xpos, ypos, src_level );
    out += ( val * g );

    surf2DLayeredwrite( out, dst_data, xpos*4, ypos, dst_level, cudaBoundaryModeZero );
}

__global__ static void vert(cudaTextureObject_t src_point_texture, cudaSurfaceObject_t dst_data, int dst_level)
{
    const int    span    =  d_gauss.inc.span[dst_level];
    const float* filter  = &d_gauss.inc.filter[dst_level*GAUSS_ALIGN];
    const int    block_x = blockIdx.x * blockDim.x;
    const int    block_y = blockIdx.y * blockDim.y;
    const int    xpos    = block_x + threadIdx.x;
    const int    ypos    = block_y + threadIdx.y;

    int   idy;
    float g;
    float val;
    float out = 0.0f;

    for( int offset = span; offset>0; offset-- ) {
        g  = filter[offset];

        idy = ypos - offset;
        val = readTex( src_point_texture, xpos, idy, dst_level );
        out += ( val * g );

        idy = ypos + offset;
        val = readTex( src_point_texture, xpos, idy, dst_level );
        out += ( val * g );
    }

    g  = filter[0];
    val = readTex( src_point_texture, xpos, ypos, dst_level );
    out += ( val * g );

    surf2DLayeredwrite( out, dst_data, xpos*4, ypos, dst_level, cudaBoundaryModeZero );
}

} // namespace absoluteSource

__host__
void Pyramid::horiz_from_prev_level_basic( int octave, int level, cudaStream_t stream )
{
    Octave&      oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    // similar speed: dim3 block( 32,  4 ); dim3 block( 32,  3 ); dim3 block( 32,  2 );
    dim3 block( 32,  8 ); // most stable good perf on GTX 980 TI
    dim3 grid;
    grid.x  = grid_divide( width,  32 );
    grid.y  = grid_divide( height, block.y );

    absoluteSource::horiz
        <<<grid,block,0,stream>>>
        ( oct_obj.getDataTexPoint( ),
          oct_obj.getIntermediateSurface( ),
          level );
    POP_SYNC_CHK;
}

__host__
void Pyramid::vert_from_interm_basic( int octave, int level, cudaStream_t stream )
{
    Octave& oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    dim3 block( 64, 2 );
    dim3 grid;
    grid.x = (unsigned int)grid_divide( width,  block.x );
    grid.y = (unsigned int)grid_divide( height, block.y );

    absoluteSource::vert
        <<<grid,block,0,stream>>>
        ( oct_obj.getIntermDataTexPoint( ),
          oct_obj.getDataSurface( ),
          level );
    POP_SYNC_CHK;
}

} // namespace popsift

