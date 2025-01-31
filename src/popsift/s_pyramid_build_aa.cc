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

static void horiz( Grid& g,
                   cudaTextureObject_t src_data,
                   cudaSurfaceObject_t dst_data,
                   int dst_level)
{
    const int    src_level = dst_level - 1;
    const int    span      =  d_gauss.inc.span[dst_level];
    const float* filter    = &d_gauss.inc.filter[dst_level*GAUSS_ALIGN];
    const int    block_x   = g.blockIdx.x * g.blockDim.x;
    const int    block_y   = g.blockIdx.y * g.blockDim.y;
    const int    xpos      = block_x + g.threadIdx.x;
    const int    ypos      = block_y + g.threadIdx.y;

    int   idx;
    float g;
    float val;
    float out = 0.0f;

    for( int offset = span; offset>0; offset-- ) {
        g  = filter[offset];

        idx = xpos - offset;
        val = src_data[src_level].ptr(ypos)[idx];
        out += ( val * g );

        idx = xpos + offset;
        val = src_data[src_level].ptr(ypos)[idx];
        out += ( val * g );
    }

    g  = filter[0];
    val = src_data[src_level].ptr(ypos)[xpos];
    out += ( val * g );

    dst_data[dst_level].ptr(ypos)[xpos] = out;
}

static void vert( Grid& g,
                  cudaTextureObject_t src_point_texture,
                  cudaSurfaceObject_t dst_data,
                  int dst_level)
{
    const int    span    =  d_gauss.inc.span[dst_level];
    const float* filter  = &d_gauss.inc.filter[dst_level*GAUSS_ALIGN];
    const int    block_x = g.blockIdx.x * g.blockDim.x;
    const int    block_y = g.blockIdx.y * g.blockDim.y;
    const int    xpos    = block_x + g.threadIdx.x;
    const int    ypos    = block_y + g.threadIdx.y;

    int   idy;
    float g;
    float val;
    float out = 0.0f;

    for( int offset = span; offset>0; offset-- ) {
        g  = filter[offset];

        idy = ypos - offset;
        val = src_data[dst_level].ptr(idy)[xpos];
        out += ( val * g );

        idy = ypos + offset;
        val = src_data[dst_level].ptr(idy)[xpos];
        out += ( val * g );
    }

    g  = filter[0];
    val = src_data[dst_level].ptr(ypos)[xpos];
    out += ( val * g );

    dst_data[dst_level].ptr(ypos)[xpos] = out;
}

} // namespace absoluteSource

__host__
void Pyramid::horiz_from_prev_level_basic( int octave, int level, cudaStream_t stream )
{
    Octave&      oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    Grid g;
    g.setBlock( 32, 8, 1 );
    g.setGrid( grid_divide( width,  32 ),
               grid_divide( height, 8 ),
               1 );

    g.reset();
    do {
        absoluteSource::horiz
            ( g,
              oct_obj.getDataTexPoint( ),
              oct_obj.getIntermediateSurface( ),
              level );
    } while( g.next() );
}

__host__
void Pyramid::vert_from_interm_basic( int octave, int level, cudaStream_t stream )
{
    Octave& oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    Grid g;
    g.setBlock( 64, 2, 1 ).
    g.setGrid( grid_divide( width,  64 ),
               grid_divide( height, 2 ),
               1 );

    g.reset();
    do {
        absoluteSource::vert
            ( g,
              oct_obj.getIntermDataTexPoint( ),
              oct_obj.getDataSurface( ),
              level );
    } while( g.next() );
}

} // namespace popsift

