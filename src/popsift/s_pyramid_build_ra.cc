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
#include "common/plane_2d.h"
#include "gauss_filter.h"
#include "sift_pyramid.h"
#include "sift_constants.h"

#include "common/write_plane_2d.h" // debug

#include <cmath>

namespace popsift {
namespace normalizedSource {

static void horiz( const Config& conf,
                   PlaneD<float>& src,
                   PlaneD<float>& dst,
                   float          shift )
{
    POP_INFO2( conf.silent(), "enter " << __PRETTY_FUNCTION__ );

    if( src.isNull() )
    {
        std::cerr << __FILE__ << ":" << __LINE__ << ": programming error," << std::endl
                  << "source plane is NULL in " << __FUNCTION__ << std::endl;
        assert( 0 );
    }
    if( dst.isNull() )
    {
        std::cerr << __FILE__ << ":" << __LINE__ << ": programming error," << std::endl
                  << "dest plane is NULL in " << __FUNCTION__ << std::endl;
        assert( 0 );
    }

    for( int y=0; y<dst.getDimY(); y++ )
    {
        for( int x=0; x<dst.getDimX(); x++ )
        {
            // Create octave-0 - level-0 from the input image.
            const int write_x = x + 1;
            const int write_y = y + 1;
            const int write_z = 0;

            const int dst_w = dst.getDimX();
            const int dst_h = dst.getDimY();

            const int    span    =  h_gauss.dd.span[0];
            const float* filter  = &h_gauss.dd.filter[0];
            const float  read_x  = ( x + shift ) / dst_w;
            const float  read_y  = ( y + shift ) / dst_h;

            float out = 0.0f;

            #pragma unroll
            for( int offset = span; offset>0; offset-- ) {
                const float& weight  = filter[offset];
                const float  offrel = float(offset) / dst_w;
                const float  v1 = src.getM( PlaneMode::NormalLinear, read_y, read_x - offrel );
                const float  v2 = src.getM( PlaneMode::NormalLinear, read_y, read_x + offrel );
                out += ( ( v1 + v2 ) * weight );
            }
            const float& weight  = filter[0];
            const float v3 = src.getM( PlaneMode::NormalLinear, read_y, read_x );
            out += ( v3 * weight );

            // POP_INFO2( conf.silent(), "span="<< span << " " << write_x << "," << write_y << "," << write_z << " is " << out * 255.f << " v3=" << v3 );
            dst.set( write_z, write_y, write_x, out * 255.0f);
        }
    }
}

} // name'/home/griff/GIT/popsift-versions/popsift-cpp-port/build/dir-octave/pyramid-o-0-l-0.pgm' space normalizedSource

void Pyramid::horiz_from_input_image( const Config& conf, ImageBase* base )
{
    POP_INFO2( conf.silent(), "enter " << __PRETTY_FUNCTION__ );

    Octave&   oct_obj = _octaves[0];

    if( base->isNull() )
    {
        std::cerr << __FILE__ << ":" << __LINE__ << ": programming error," << std::endl
                  << "source plane is NULL in " << __FUNCTION__ << std::endl;
        assert( 0 );
    }
    if( oct_obj.getIntm().isNull() )
    {
        std::cerr << __FILE__ << ":" << __LINE__ << ": programming error," << std::endl
                  << "dest plane is NULL in " << __FUNCTION__ << std::endl;
        assert( 0 );
    }

    float shift  = 0.5f * std::pow( 2.0f, conf.getUpscaleFactor() );

    normalizedSource::horiz
        ( conf,
          base->getFloatPlane(),
          oct_obj.getIntm(),
          shift );
}

} // namespace popsift

