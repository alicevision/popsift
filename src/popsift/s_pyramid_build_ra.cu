/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "gauss_filter.h"
#include "s_pyramid_build_ra.h"
#include "sift_constants.h"

namespace popsift {
namespace gauss {
namespace normalizedSource {

__global__
void horiz( cudaTextureObject_t src_linear_tex,
            cudaSurfaceObject_t dst_data,
            int                 dst_w,
            int                 dst_h,
            int                 octave,
            float               shift )
{
    // Create level-0 for any octave from the input image.
    // Since we are computing the direct-downscaling gauss filter tables
    // and the first entry in that table is identical to the "normal"
    // table, we do not need a special case.

    const int    write_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int    write_y = blockIdx.y;

    if( write_x >= dst_w ) return;

    const int    span    =  d_gauss.dd.span[octave];
    const float* filter  = &d_gauss.dd.filter[octave*GAUSS_ALIGN];
    const float  read_x  = ( blockIdx.x * blockDim.x + threadIdx.x + shift ) / dst_w;
    const float  read_y  = ( blockIdx.y + shift ) / dst_h;

    float out = 0.0f;

    #pragma unroll
    for( int offset = span; offset>0; offset-- ) {
        const float& g  = filter[offset];
        const float  offrel = float(offset) / dst_w;
        const float  v1 = tex2D<float>( src_linear_tex, read_x - offrel, read_y );
        const float  v2 = tex2D<float>( src_linear_tex, read_x + offrel, read_y );
        out += ( ( v1 + v2 ) * g );
    }
    const float& g  = filter[0];
    const float v3 = tex2D<float>( src_linear_tex, read_x, read_y );
    out += ( v3 * g );

    surf2DLayeredwrite( out * 255.0f, dst_data, write_x*4, write_y, 0, cudaBoundaryModeZero );
}

__global__
void horiz_level( cudaTextureObject_t src_linear_tex,
                  cudaSurfaceObject_t dst_data,
                  int                 dst_w,
                  int                 dst_h,
                  int                 /* octave */,
                  int                 level,
                  float               shift )
{
    const int    write_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int    write_y = blockIdx.y;

    if( write_x >= dst_w ) return;

    const float  read_x  = ( blockIdx.x * blockDim.x + threadIdx.x + shift ) / dst_w;
    const float  read_y  = ( blockIdx.y + shift ) / dst_h;

    const int    span      =  d_gauss.abs_o0.span[level];
    const float* filter    = &d_gauss.abs_o0.filter[level*GAUSS_ALIGN];

    float out = 0.0f;

    for( int offset = span; offset>0; offset-- ) {
        const float& g  = filter[offset];
        const float  offrel = float(offset) / dst_w;
        const float  v1 = tex2D<float>( src_linear_tex, read_x - offrel, read_y );
        const float  v2 = tex2D<float>( src_linear_tex, read_x + offrel, read_y );
        out += ( ( v1 + v2 ) * g );
    }
    const float& g  = filter[0];
    const float  v3 = tex2D<float>( src_linear_tex, read_x, read_y );
    out += ( v3 * g );

    surf2DLayeredwrite( out * 255.0f, dst_data, write_x*4, write_y, level, cudaBoundaryModeZero );
}

__global__
void horiz_all( cudaTextureObject_t src_linear_tex,
                cudaSurfaceObject_t dst_data,
                int                 dst_w,
                int                 dst_h,
                float               shift,
                const int           max_level ) // dst_level )
{
    const int    write_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int    write_y = blockIdx.y;

    if( write_x >= dst_w ) return;

    const float  read_x  = ( blockIdx.x * blockDim.x + threadIdx.x + shift ) / dst_w;
    const float  read_y  = ( blockIdx.y + shift ) / dst_h;

    for( int dst_level=0; dst_level < max_level; dst_level++ )
    {
        const int    span      =  d_gauss.abs_o0.span[dst_level];
        const float* filter    = &d_gauss.abs_o0.filter[dst_level*GAUSS_ALIGN];

        float out = 0.0f;

        for( int offset = span; offset>0; offset-- ) {
            const float& g  = filter[offset];
            const float  offrel = float(offset) / dst_w;
            const float  v1 = tex2D<float>( src_linear_tex, read_x - offrel, read_y );
            const float  v2 = tex2D<float>( src_linear_tex, read_x + offrel, read_y );
            out += ( ( v1 + v2 ) * g );
        }
        const float& g  = filter[0];
        const float  v3 = tex2D<float>( src_linear_tex, read_x, read_y );
        out += ( v3 * g );

        surf2DLayeredwrite( out * 255.0f, dst_data, write_x*4, write_y, dst_level, cudaBoundaryModeZero );
    }
}

} // namespace normalizedSource
} // namespace gauss
} // namespace popsift

