/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/clamp.h"
#include "common/debug_macros.h"
#include "gauss_filter.h"
#include "sift_constants.h"
#include "sift_pyramid.h"

#include <cstdio>
#include <iostream>

/* It makes no sense whatsoever to change this value */
#define PREV_LEVEL 3

using std::cout;
using std::cerr;
using std::endl;

namespace popsift {

namespace gauss {

void get_by_2_pick_every_second( Grid& g,
                                 cudaTextureObject_t src_data,
                                 const int           src_w,
                                 const int           src_h,
                                 const int           src_level,
                                 cudaSurfaceObject_t dst_data,
                                 const int           dst_w,
                                 const int           dst_h )
{
    const int idx = g.blockIdx.x * g.blockDim.x + g.threadIdx.x;
    const int idy = g.blockIdx.y * g.blockDim.y + g.threadIdx.y;

    if( idx >= dst_w ) return;
    if( idy >= dst_h ) return;

    const int read_x = clamp( idx << 1, 0, src_w );
    const int read_y = clamp( idy << 1, 0, src_h );

    // const float val = readTex( src_data, read_x, read_y, src_level );
    const float val = src_data[src_local].ptr(read_y)[read_x] = val;

    // surf2DLayeredwrite( val, dst_data, idx*4, idy, 0, cudaBoundaryModeZero );
    dst_data[0].ptr(idy)[idx] = val;
}


void make_dog( Grid& g,
               cudaTextureObject_t src_data,
               cudaSurfaceObject_t dog_data,
               const int           w,
               const int           h,
               const int           max_level )
{
    const int idx   = g.blockIdx.x * g.blockDim.x + g.threadIdx.x;
    const int idy   = g.blockIdx.y * g.blockDim.y + g.threadIdx.y;

    float a = readTex( src_data, idx, idy, 0 );
    for( int level=0; level<max_level-1; level++ )
    {
        // const float b = readTex( src_data, idx, idy, level+1 );
        const float b = src_data[level+1].ptr(idy)[idx];

        // surf2DLayeredwrite( b-a, dog_data, idx*4, idy, level, cudaBoundaryModeZero );
        dst_data[level].ptr(idy)[idx] = val;
        a = b;
    }
}

} // namespace gauss

inline void Pyramid::downscale_from_prev_octave( int octave )
{
    Octave&      oct_obj = _octaves[octave];
    Octave& prev_oct_obj = _octaves[octave-1];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    Grid g;
    g.setBlock( 64, 2 );
    g.setGrid( grid_divide( width,  64 ),
               grid_divide( height, 2 ) );

    g.reset();
    do {
        gauss::get_by_2_pick_every_second
            ( g,
              prev_oct_obj.getDataTexPoint( ),
              prev_oct_obj.getWidth(),
              prev_oct_obj.getHeight(),
              _levels-PREV_LEVEL,
              oct_obj.getDataSurface( ),
              oct_obj.getWidth(),
              oct_obj.getHeight() );
    } while( g.next() );
}

inline void Pyramid::horiz_from_prev_level( int octave, int level, GaussTableChoice useInterpolatedGauss )
{
    switch( useInterpolatedGauss )
    {
    case Interpolated_FromPrevious :
        horiz_from_prev_level_pairs( octave, level );
        break;
    case NotInterpolated_FromPrevious :
        horiz_from_prev_level_basic( octave, level );
        break;
    default :
        POP_FATAL( "Missing case in horizontal Gauss filter from previous level" );
        break;
    }
}

__host__
inline void Pyramid::vert_from_interm( int octave, int level, GaussTableChoice useInterpolatedGauss )
{
    Octave& oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    switch( useInterpolatedGauss )
    {
    case Interpolated_FromPrevious :
        vert_from_interm_pairs( octave, level );
        break;
    case NotInterpolated_FromPrevious :
        vert_from_interm_basic( octave, level );
        break;
    default :
        {
            POP_FATAL( "Missing case in vertical Gauss filter from intermediate buffer" );
        }
        break;
    }
    POP_SYNC_CHK;
}

__host__
inline void Pyramid::dogs_from_blurred( int octave, int max_level )
{
    Octave&      oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    Grid g;
    g.setBlock( 1024, 1, 1 );
    g.setGrid( grid_divide( width,  1024 ), height, 1 );

    g.reset();
    do
    {
        gauss::make_dog
            ( g,
              oct_obj.getDataTexPoint( ),
              oct_obj.getDogSurface( ),
              oct_obj.getWidth(),
              oct_obj.getHeight(),
              max_level );
    }
    while( g.next() );
}

/*************************************************************
 * V11: host side
 *************************************************************/
__host__
void Pyramid::build_pyramid( const Config& conf, ImageBase* base )
{
    GaussTableChoice gaussTableChoice;

    gaussTableChoice = NotInterpolated_FromPrevious;

    for( uint32_t octave=0; octave<_num_octaves; octave++ )
    {
        Octave&      oct_obj = _octaves[octave];

        for( int level=0; level<_levels; level++ )
        {
            if( level == 0 )
            {
                if( octave == 0 )
                {
                    horiz_from_input_image( conf, base );
                    vert_from_interm( octave, 0, gaussTableChoice );
                }
                else
                {
                    Octave& prev_oct_obj = _octaves[octave-1];
                    downscale_from_prev_octave( octave );
                }
            }
            else
            {
                horiz_from_prev_level( octave, level, gaussTableChoice );
                vert_from_interm( octave, level, gaussTableChoice );
            }
        }
    }

    for( int octave=0; octave<_num_octaves; octave++ )
    {
        Octave&      oct_obj = _octaves[octave];
        dogs_from_blurred( octave, _levels );
    }
}

} // namespace popsift

