/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "s_pyramid_build_ra.h"
#include "sift_constants.h"
#include "gauss_filter.h"
#include "common/assist.h"

namespace popsift {
namespace gauss {
namespace relativeSource {

__global__
void horiz( cudaTextureObject_t src_linear_tex,
            cudaSurfaceObject_t dst_data,
            int                 dst_w,
            int                 dst_h,
            int                 octave,
            float               shift )
{
    // The first line creates level-0 octave-0 for the input image only.
    // Since we are computing the direct-downscaling gauss filter tables
    // and the first entry in that table is identical to the "normal"
    // table, we do not need a special case.
    // horiz( src_linear_tex, dst_data, shift, d_gauss.inc.span[0], &d_gauss.inc.filter[0*GAUSS_ALIGN] );

    const int    span   =  d_gauss.dd.span[octave];
    const float* filter = &d_gauss.dd.filter[octave*GAUSS_ALIGN];
    const float  read_y = ( blockIdx.y + shift ) / dst_h;

    const int off_x = blockIdx.x * blockDim.x + threadIdx.x;

    if( off_x >= dst_w ) return;

    float out = 0.0f;

    #pragma unroll
    for( int offset = span; offset>0; offset-- ) {
        const float& g  = filter[offset];
        const float read_x_l = ( off_x - offset );
        const float  v1 = tex2D<float>( src_linear_tex, ( read_x_l + shift ) / dst_w, read_y );
        out += ( v1 * g );

        const float read_x_r = ( off_x + offset );
        const float  v2 = tex2D<float>( src_linear_tex, ( read_x_r + shift ) / dst_w, read_y );
        out += ( v2 * g );
    }
    const float& g  = filter[0];
    const float read_x = off_x;
    const float v3 = tex2D<float>( src_linear_tex, ( read_x + shift ) / dst_w, read_y );
    out += ( v3 * g );

    surf2Dwrite( out * 255.0f, dst_data, off_x*4, blockIdx.y, cudaBoundaryModeZero );
}

} // namespace relativeSource
} // namespace gauss
} // namespace popsift

