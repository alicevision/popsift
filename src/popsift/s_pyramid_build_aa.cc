/*
 * Copyright 2016-2017, Simula Research Laboratory
 *           2018-2024, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/grid.h"
#include "gauss_filter.h"
#include "sift_pyramid.h"
#include "sift_constants.h"

namespace popsift {
namespace absoluteSource {

static void horiz( Grid& g,
                   Plane2D_float& src,   // point data
                   Plane2D_float& dst,
                   int dst_level )
{
    const int    src_level = dst_level - 1;
    const int    span      =  h_gauss.inc.span[dst_level];
    const float* filter    = &h_gauss.inc.filter[dst_level*GAUSS_ALIGN];
    const int    block_x   = g.blockIdx.x * g.blockDim.x;
    const int    block_y   = g.blockIdx.y * g.blockDim.y;
    const int    xpos      = block_x + g.threadIdx.x;
    const int    ypos      = block_y + g.threadIdx.y;

    float weight;
    float val;
    float out = 0.0f;

    for( int offset = span; offset>0; offset-- ) {
        weight  = filter[offset];

        val = src.get( src_level, ypos, xpos-offset );
        out += ( val * weight );

        val = src.get( src_level, ypos, xpos+offset );
        out += ( val * weight );
    }

    weight  = filter[0];
    val = src.get( src_level, ypos, xpos );
    out += ( val * weight );

    dst.set( dst_level, ypos, xpos, out );
}

static void vert( Grid& g,
                  Plane2D_float& src,   // point data
                  Plane2D_float& dst,
                  int dst_level)
{
    const int    span    =  h_gauss.inc.span[dst_level];
    const float* filter  = &h_gauss.inc.filter[dst_level*GAUSS_ALIGN];
    const int    block_x = g.blockIdx.x * g.blockDim.x;
    const int    block_y = g.blockIdx.y * g.blockDim.y;
    const int    xpos    = block_x + g.threadIdx.x;
    const int    ypos    = block_y + g.threadIdx.y;

    int   idy;
    float weight;
    float val;
    float out = 0.0f;

    for( int offset = span; offset>0; offset-- ) {
        weight  = filter[offset];

        val = src.get( dst_level, ypos - offset, xpos );
        out += ( val * weight );

        val = src.get( dst_level, ypos + offset, xpos );
        out += ( val * weight );
    }

    weight  = filter[0];
    val = src.get( dst_level, ypos, xpos );
    out += ( val * weight );

    dst.set( dst_level, ypos, xpos, out );
}

} // namespace absoluteSource

void Pyramid::horiz_from_prev_level_basic( int octave, int level )
{
    Octave&      oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    Grid g;
    g.setBlockDim( 32, 8, 1 );
    g.setGridDim( grid_divide( width,  32 ),
                  grid_divide( height, 8 ),
                  1 );

    g.reset();
    do {
        absoluteSource::horiz
            ( g,
              oct_obj.getData( ),
              oct_obj.getIntm( ),
              level );
    } while( g.next() );
}

void Pyramid::vert_from_interm_basic( int octave, int level )
{
    Octave& oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    Grid g;
    g.setBlockDim( 64, 2, 1 );
    g.setGridDim( grid_divide( width,  64 ),
                  grid_divide( height, 2 ),
                  1 );

    g.reset();
    do {
        absoluteSource::vert
            ( g,
              oct_obj.getIntm( ),
              oct_obj.getData( ),
              level );
    } while( g.next() );
}

} // namespace popsift

