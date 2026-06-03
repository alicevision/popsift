/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "gauss_filter.h"
#include "s_pyramid_build_aa.h"
#include "sift_constants.h"

namespace popsift {
namespace gauss {
namespace absoluteSource {

__global__ void horiz(LayeredReadTex src_point_texture, cudaSurfaceObject_t dst_data, int dst_level)
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

    // Horizontal Gauss tap exchange via warp shuffles. The block is (32,blockDim.y):
    // each threadIdx.y row is an independent 32-lane group convolving one image
    // row. On a 64-lane wavefront two rows share a wavefront, so the shuffles must
    // be confined to a width-32 sub-group (threadIdx.x is already the in-row lane
    // id since blockDim.x==32) or a lane would pull a neighbour from the wrong
    // image row and corrupt the pyramid. CUDA: width 32 is the whole warp.
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
    int shiftval = 0;
    for( int offset=span-1; offset>0; offset-- ) {
        shiftval += 1;
        const float D1 = popsift::shuffle_down( A, shiftval, 32 );
        const float D2 = popsift::shuffle_up  ( C, span - shiftval, 32 );
        const float D  = threadIdx.x < (32 - shiftval) ? D1 : D2;
        const float E1 = popsift::shuffle_up  ( B, shiftval, 32 );
        const float E2 = popsift::shuffle_down( C, span - shiftval, 32 );
        const float E  = threadIdx.x > shiftval        ? E1 : E2;
        g = filter[offset];
        out += ( D + E ) * g;
    }
#else
    int shiftval = 0;
    for( int offset=span-1; offset>0; offset-- ) {
        shiftval += 1;
        const float D1 = popsift::shuffle_down( A, shiftval );
        const float D2 = popsift::shuffle_up  ( C, span - shiftval );
        const float D  = threadIdx.x < (32 - shiftval) ? D1 : D2;
        const float E1 = popsift::shuffle_up  ( B, shiftval );
        const float E2 = popsift::shuffle_down( C, span - shiftval );
        const float E  = threadIdx.x > shiftval        ? E1 : E2;
        g = filter[offset];
        out += ( D + E ) * g;
    }
#endif

    surf2DLayeredwrite( out, dst_data, off_x*4, off_y, dst_level, cudaBoundaryModeZero );
}

__global__ void vert(LayeredReadTex src_point_texture, cudaSurfaceObject_t dst_data, int dst_level)
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
        val = readTex( src_point_texture, block_x + idx, block_y + idy, dst_level );
        out += ( val * g );

        idy = threadIdx.y + offset;
        val = readTex( src_point_texture, block_x + idx, block_y + idy, dst_level );
        out += ( val * g );
    }

    g  = filter[0];
    idy = threadIdx.y;
    val = readTex( src_point_texture, block_x + idx, block_y + idy, dst_level );
    out += ( val * g );

    idx = block_x+threadIdx.x;
    idy = block_y+threadIdx.y;

    surf2DLayeredwrite( out, dst_data, idx*4, idy, dst_level, cudaBoundaryModeZero );
}

__global__ void vert_abs0(LayeredReadTex src_point_texture, cudaSurfaceObject_t dst_data, int dst_level)
{
    const int    span   =  d_gauss.abs_o0.span[dst_level];
    const float* filter = &d_gauss.abs_o0.filter[dst_level*GAUSS_ALIGN];
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
        val = readTex( src_point_texture, block_x + idx, block_y + idy, dst_level );
        out += ( val * g );

        idy = threadIdx.y + offset;
        val = readTex( src_point_texture, block_x + idx, block_y + idy, dst_level );
        out += ( val * g );
    }

    g  = filter[0];
    idy = threadIdx.y;
    val = readTex( src_point_texture, block_x + idx, block_y + idy, dst_level );
    out += ( val * g );

    idx = block_x+threadIdx.x;
    idy = block_y+threadIdx.y;

    surf2DLayeredwrite( out, dst_data, idx*4, idy, dst_level, cudaBoundaryModeZero );
}

__global__ void vert_all_abs0(LayeredReadTex src_point_texture,
                              cudaSurfaceObject_t dst_data,
                              int start_level,
                              int max_level)
{
    const int block_x = blockIdx.x * blockDim.x;
    const int block_y = blockIdx.y * blockDim.y;

    for( int dst_level=start_level; dst_level<max_level; dst_level++ )
    {
        const int    span   =  d_gauss.abs_o0.span[dst_level];
        const float* filter = &d_gauss.abs_o0.filter[dst_level*GAUSS_ALIGN];

        int idx = threadIdx.x;
        int idy;

        float g;
        float val;
        float out = 0;

        for( int offset = span; offset>0; offset-- ) {
            g  = filter[offset];

            idy = threadIdx.y - offset;
            val = readTex( src_point_texture, block_x + idx, block_y + idy, dst_level );
            out += ( val * g );

            idy = threadIdx.y + offset;
            val = readTex( src_point_texture, block_x + idx, block_y + idy, dst_level );
            out += ( val * g );
        }

        g  = filter[0];
        idy = threadIdx.y;
        val = readTex( src_point_texture, block_x + idx, block_y + idy, dst_level );
        out += ( val * g );

        idx = block_x+threadIdx.x;
        idy = block_y+threadIdx.y;

        surf2DLayeredwrite( out, dst_data, idx*4, idy, dst_level, cudaBoundaryModeZero );
    }
}

} // namespace absoluteSource
} // namespace gauss
} // namespace popsift

