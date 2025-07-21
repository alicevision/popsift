/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/clamp.h"
#include "common/grid.h"
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

inline void get_by_2_pick_every_second( Grid&          g,
                                        const int      src_level,
                                        PlaneD<float>& src,
                                        PlaneD<float>& dst )
{
    const int src_w = src.getDimX();
    const int src_h = src.getDimY();
    const int dst_w = dst.getDimX();
    const int dst_h = dst.getDimY();

    g.reset();

    do {
        const int idx = g.blockIdx.x * g.blockDim.x + g.threadIdx.x;
        const int idy = g.blockIdx.y * g.blockDim.y + g.threadIdx.y;

        if( idx >= dst_w ) return;
        if( idy >= dst_h ) return;

        const int read_x = std::clamp( idx << 1, 0, src_w-1 );
        const int read_y = std::clamp( idy << 1, 0, src_h-1 );

        const float val = src.get( src_level, read_y, read_x );

        dst.set( 0, idy, idx, val );
    } while( g.next() );
}


void make_dog( Grid&          g,
               PlaneD<float>& src,
               PlaneD<float>& dog,
               const int      w,
               const int      h,
               const int      max_level )
{
    g.reset();

    do {
        const int idx   = g.blockIdx.x * g.blockDim.x + g.threadIdx.x;
        const int idy   = g.blockIdx.y * g.blockDim.y + g.threadIdx.y;

        // float a = readTex( src_data, idx, idy, 0 );
        float a = src.get( 0, idy, idx );
        for( int level=0; level<max_level-1; level++ )
        {
            const float b = src.get( level+1, idy, idx );

            dog.set( level, idy, idx, b-a );
            a = b;
        }
    } while( g.next() );
}

} // namespace gauss

void Pyramid::downscale_from_prev_octave( int octave )
{
    Octave&      oct_obj = _octaves[octave];
    Octave& prev_oct_obj = _octaves[octave-1];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    Grid g;
    g.setBlockDim( 64, 2 );
    g.setGridDim( grid_divide( width,  64 ),
                  grid_divide( height, 2 ) );

   gauss::get_by_2_pick_every_second( g,
                                      _levels-PREV_LEVEL,
                                      prev_oct_obj.getData( ),
                                      oct_obj.getData( ) );
}

void Pyramid::dogs_from_blurred( int octave, int max_level )
{
    Octave&      oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    Grid g;
    g.setBlockDim( 1024, 1, 1 );
    g.setGridDim( grid_divide( width,  1024 ), height, 1 );

    gauss::make_dog( g,
                     oct_obj.getData( ),
                     oct_obj.getDog( ),
                     oct_obj.getWidth(),
                     oct_obj.getHeight(),
                     max_level );
}

/*************************************************************
 * V11: host side
 *************************************************************/
void Pyramid::build_pyramid( const Config& conf, ImageBase* base )
{
    POP_INFO2( conf.silent(), "enter " << __PRETTY_FUNCTION__ );
    POP_INFO2( conf.silent(), "is image NULL? " << (base->isNull() ? "yes" : "no") );

    for( uint32_t octave=0; octave<_num_octaves; octave++ )
    {
        Octave&      oct_obj = _octaves[octave];

        for( int level=0; level<_levels; level++ )
        {
            if( level == 0 )
            {
                if( octave == 0 )
                {
                    POP_INFO2( conf.silent(), "call horiz_from_input_image" );
                    horiz_from_input_image( conf, base );

                    POP_INFO2( conf.silent(), "call vert_from_interm" );
                    vert_from_interm( octave, 0 );
                }
                else
                {
                    Octave& prev_oct_obj = _octaves[octave-1];
                    POP_INFO2( conf.silent(), "call downscale_from_prev_octave" );
                    downscale_from_prev_octave( octave );
                }
            }
            else
            {
                POP_INFO2( conf.silent(), "call horiz_from_prev_level" );
                horiz_from_prev_level( octave, level );

                POP_INFO2( conf.silent(), "call vert_from_interm" );
                vert_from_interm( octave, level );
            }
        }
    }

    for( int octave=0; octave<_num_octaves; octave++ )
    {
        Octave&      oct_obj = _octaves[octave];
        POP_INFO2( conf.silent(), "call dogs_from_blurred" );
        dogs_from_blurred( octave, _levels );
    }
}

} // namespace popsift

