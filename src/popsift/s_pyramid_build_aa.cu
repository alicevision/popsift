/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "s_pyramid_build_aa.h"
#include "sift_constants.h"
#include "gauss_filter.h"
#include "common/assist.h"

namespace popsift {
namespace gauss {
namespace absoluteSource {

__global__
void horiz( cudaTextureObject_t src_point_texture,
            cudaSurfaceObject_t dst_data,
            const int           dst_level )
{
    const int    src_level = dst_level - 1;
    const int    span      =  d_gauss.inc.span[dst_level];
    const float* filter    = &d_gauss.inc.filter[dst_level*GAUSS_ALIGN];

    const int off_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int off_y = blockIdx.y * blockDim.y + threadIdx.y;

    float out = 0.0f;

    float A = readTex( src_point_texture, off_x - span, off_y, src_level );
    float B = readTex( src_point_texture, off_x + span, off_y, src_level );
    float C = readTex( src_point_texture, off_x       , off_y, src_level );
    float g  = filter[0];
    out += C * g;
    g    = filter[span];
    out += ( A + B ) * g;

    int shiftval = 0;
    for( int offset=span-1; offset>0; offset-- ) {
        shiftval += 1;
        const float D1 = __shfl_down( A, shiftval );
        const float D2 = __shfl_up  ( C, span - shiftval );
        const float D  = threadIdx.x < (32 - shiftval) ? D1 : D2;
        const float E1 = __shfl_up  ( B, shiftval );
        const float E2 = __shfl_down( C, span - shiftval );
        const float E  = threadIdx.x > shiftval        ? E1 : E2;
        g = filter[offset];
        out += ( D + E ) * g;
    }

    surf2Dwrite( out, dst_data, off_x*4, off_y, cudaBoundaryModeZero );
}

__global__
void vert( cudaTextureObject_t src_point_texture,
           cudaSurfaceObject_t dst_data,
           const int           dst_level )
{
    const int    span   =  d_gauss.inc.span[dst_level];
    const float* filter = &d_gauss.inc.filter[dst_level*GAUSS_ALIGN];
    int block_x = blockIdx.x * blockDim.x;
    int block_y = blockIdx.y * blockDim.y;
    int idx     = threadIdx.x;
    int idy;

    float g;
    float val;
    float out = 0;

    for( int offset = span; offset>0; offset-- ) {
        g  = filter[offset];

        idy = threadIdx.y - offset;
        val = tex2D<float>( src_point_texture, block_x + idx + 0.5f, block_y + idy + 0.5f );
        out += ( val * g );

        idy = threadIdx.y + offset;
        val = tex2D<float>( src_point_texture, block_x + idx + 0.5f, block_y + idy + 0.5f );
        out += ( val * g );
    }

    g  = filter[0];
    idy = threadIdx.y;
    val = tex2D<float>( src_point_texture, block_x + idx + 0.5f, block_y + idy + 0.5f );
    out += ( val * g );

    idx = block_x+threadIdx.x;
    idy = block_y+threadIdx.y;

    surf2DLayeredwrite( out, dst_data, idx*4, idy, dst_level, cudaBoundaryModeZero );
}

} // namespace absoluteSource
} // namespace gauss
} // namespace popsift

