/*
 * Copyright 2016-2017, Simula Research Laboratory
 *           2018-2024, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/plane_2d.h"
#include "gauss_filter.h"
#include "sift_pyramid.h"
#include "sift_constants.h"

namespace popsift {
namespace normalizedSource {

static void horiz( Grid g,
                   PlaneD<float>& src,
                   PlaneD<float>& dst,
                   float          shift )
{
    while( g.next() )
    {
        // Create octave-0 - level-0 from the input image.
        const int write_x = g.blockIdx.x * g.blockDim.x + g.threadIdx.x;
        const int write_y = g.blockIdx.y;
        const int write_z = 0;

        if( write_x >= dst_w ) return;

        const int    span    =  d_gauss.dd.span[0];
        const float* filter  = &d_gauss.dd.filter[0];
        const float  read_x  = ( g.blockIdx.x * g.blockDim.x + g.threadIdx.x + shift ) / dst_w;
        const float  read_y  = ( g.blockIdx.y + shift ) / dst_h;

        float out = 0.0f;

        #pragma unroll
        for( int offset = span; offset>0; offset-- ) {
            const float& g  = filter[offset];
            const float  offrel = float(offset) / dst_w;
            const float  v1 = src.get( PlaneD<float>::NormalLinear, read_y, read_x - offrel );
            const float  v2 = src.get( PlaneD<float>::NormalLinear, read_y, read_x + offrel );
            out += ( ( v1 + v2 ) * g );
        }
        const float& g  = filter[0];
        const float v3 = src.get( PlaneD<float>::NormalLinear, read_y, read_x );
        out += ( v3 * g );

        dst[write_z,write_y,write_x] = out * 255.0f;
    }
}

} // namespace normalizedSource

void Pyramid::horiz_from_input_image( const Config& conf, ImageBase* base )
{
    Octave&   oct_obj = _octaves[0];

    const int width   = oct_obj.getWidth();
    const int height  = oct_obj.getHeight();

    Grid g;
    get.setGrid( grid_divide( width, 128 ),
                 height );
    get.setBlock( 128, 1, 1 );

    float shift  = 0.5f * powf( 2.0f, conf.getUpscaleFactor() );

    normalizedSource::horiz
        ( G,
          base,
          oct_obj.getIntermediateSurface(),
          width,
          height,
          shift );

    POP_SYNC_CHK;
}

} // namespace popsift

