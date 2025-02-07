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

#include <cmath>

namespace popsift {
namespace normalizedSource {

static void horiz( Grid& g,
                   PlaneD<float>& src,
                   PlaneD<float>& dst,
                   float          shift )
{
    do
    {
        // Create octave-0 - level-0 from the input image.
        const int write_x = g.blockIdx.x * g.blockDim.x + g.threadIdx.x;
        const int write_y = g.blockIdx.y;
        const int write_z = 0;

        const int dst_w = dst.getDimX();
        const int dst_h = dst.getDimY();

        if( write_x >= dst_w ) continue;

        const int    span    =  h_gauss.dd.span[0];
        const float* filter  = &h_gauss.dd.filter[0];
        const float  read_x  = ( g.blockIdx.x * g.blockDim.x + g.threadIdx.x + shift ) / dst_w;
        const float  read_y  = ( g.blockIdx.y + shift ) / dst_h;

        float out = 0.0f;

        #pragma unroll
        for( int offset = span; offset>0; offset-- ) {
            const float& weight  = filter[offset];
            const float  offrel = float(offset) / dst_w;
            const float  v1 = src.get( PlaneMode::NormalLinear{}, read_y, read_x - offrel );
            const float  v2 = src.get( PlaneMode::NormalLinear{}, read_y, read_x + offrel );
            out += ( ( v1 + v2 ) * weight );
        }
        const float& weight  = filter[0];
        const float v3 = src.get( PlaneMode::NormalLinear{}, read_y, read_x );
        out += ( v3 * weight );

        dst.set( write_z, write_y, write_x, out * 255.0f);
    }
    while( g.next() );
}

} // namespace normalizedSource

void Pyramid::horiz_from_input_image( const Config& conf, ImageBase* base )
{
    Octave&   oct_obj = _octaves[0];

    const int width   = oct_obj.getWidth();
    const int height  = oct_obj.getHeight();

    Grid g;
    g.setGridDim( grid_divide( width, 128 ), height );
    g.setBlockDim( 128, 1, 1 );

    float shift  = 0.5f * std::pow( 2.0f, conf.getUpscaleFactor() );

    normalizedSource::horiz
        ( g,
          base->getFloatPlane(),
          oct_obj.getIntm(),
          shift );
}

} // namespace popsift

