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

static inline
void get_by_2_pick_every_second( const int      src_level,
                                 PlaneD<float>& src,
                                 PlaneD<float>& dst )
{
    const int dst_level = 0; // always writing to the first plane in the destination octave

    const int src_w = src.getDimX();
    const int src_h = src.getDimY();
    const int dst_w = dst.getDimX();
    const int dst_h = dst.getDimY();

    for( int idy=0; idy<dst_h; idy++ )
    {
        for( int idx=0; idx<dst_w; idx++ )
        {
            const int read_x = std::clamp( idx << 1, 0, src_w-1 );
            const int read_y = std::clamp( idy << 1, 0, src_h-1 );

            const float val = src.get( src_level, read_y, read_x );

            dst.set( dst_level, idy, idx, val );
        }
    }
}

}; // namespace gauss

void Pyramid::downscale_from_prev_octave( int octave )
{
    Octave&      oct_obj = _octaves[octave];
    Octave& prev_oct_obj = _octaves[octave-1];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

   gauss::get_by_2_pick_every_second( _levels-PREV_LEVEL,
                                      prev_oct_obj.getData( ),
                                      oct_obj.getData( ) );
}

namespace gauss {

static
void make_dog( PlaneD<float>& src,
               PlaneD<float>& dog,
               const int      w,
               const int      h,
               const int      max_level )
{
    for( int idy = 0; idy < h; idy++ )
    {
        for( int idx = 0; idx < w; idx++ )
        {
            float a = src.get( 0, idy, idx );
            for( int level = 0; level < max_level-1; level++ )
            {
                const float b = src.get( level+1, idy, idx );
                dog.set( level, idy, idx, b-a );
                a = b;
            }
        }
    }
}

} // namespace gauss

void Pyramid::dogs_from_blurred( int octave, int max_level )
{
    Octave&      oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    gauss::make_dog(oct_obj.getData( ),
                     oct_obj.getDog( ),
                     oct_obj.getWidth(),
                     oct_obj.getHeight(),
                     max_level );
}

/*************************************************************
 * V11: host side
 *************************************************************/
void Pyramid::build_pyramid( const Config& conf, std::shared_ptr<ImageBase> base )
{
    POP_INFO2( conf.silent(), "enter " << __PRETTY_FUNCTION__ );
    POP_INFO2( conf.silent(), "is image NULL? " << ( base->isNull() ? "yes" : "no") );

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

