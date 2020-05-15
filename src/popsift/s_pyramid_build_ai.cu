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
namespace absoluteSourceInterpolated {

__global__ void horiz(cudaTextureObject_t src_linear_tex, cudaSurfaceObject_t dst_data, int dst_level)
{
    const int    src_level = dst_level - 1;
    const int    span      =  d_gauss.inc.i_span[dst_level];
    const float* filter    = &d_gauss.inc.i_filter[dst_level*GAUSS_ALIGN];

    const int off_x = blockIdx.x * blockDim.x + threadIdx.x;

    float out = 0.0f;

    for( int offset = 1; offset<=span; offset += 2 ) {
        const float u    = filter[offset];
        const float off  = offset + ( 1.0f - u );
        const float val = readTex( src_linear_tex, off_x - off, blockIdx.y, src_level )
                        + readTex( src_linear_tex, off_x + off, blockIdx.y, src_level );

        const float v = filter[offset+1];
        out += val * v;
    }
    const float& g  = filter[0];
    const float v3 = readTex( src_linear_tex, off_x, blockIdx.y, src_level );
    out += ( v3 * g );

    surf2DLayeredwrite( out, dst_data, off_x*4, blockIdx.y, dst_level, cudaBoundaryModeZero );
}

__global__ void vert(cudaTextureObject_t src_linear_tex, cudaSurfaceObject_t dst_data, int dst_level)
{
    const int    span   =  d_gauss.inc.i_span[dst_level];
    const float* filter = &d_gauss.inc.i_filter[dst_level*GAUSS_ALIGN];
    int block_x   = blockIdx.x * blockDim.y;
    int block_y   = blockIdx.y * blockDim.x;
    const int idx = threadIdx.y;
    const int idy = threadIdx.x;

    float out = 0;

    for( int offset = 1; offset<=span; offset += 2 ) {
        const float u    = filter[offset];
        const float off  = offset + ( 1.0f - u );
        const float val = readTex( src_linear_tex, block_x + idx, block_y + idy - off, dst_level )
                        + readTex( src_linear_tex, block_x + idx, block_y + idy + off, dst_level );

        const float v = filter[offset+1];
        out += val * v;
    }

    float g   = filter[0];
    float val = readTex( src_linear_tex, block_x + idx, block_y + idy, dst_level );
    out += ( val * g );

    surf2DLayeredwrite( out, dst_data, (block_x+idx)*4, block_y+idy, dst_level, cudaBoundaryModeZero );
}

__global__ void vert_abs0(cudaTextureObject_t src_linear_tex, cudaSurfaceObject_t dst_data, int dst_level)
{
    const int    span   =  d_gauss.abs_o0.i_span[dst_level];
    const float* filter = &d_gauss.abs_o0.i_filter[dst_level*GAUSS_ALIGN];
    const int block_x   = blockIdx.x * blockDim.y;
    const int block_y   = blockIdx.y * blockDim.x;
    const int idx       = threadIdx.y;
    const int idy       = threadIdx.x;

    float out = 0;

    for( int offset = 1; offset<=span; offset += 2 ) {
        const float u    = filter[offset];
        const float off  = offset + ( 1.0f - u );
        const float val = readTex( src_linear_tex, block_x + idx, block_y + idy - off, dst_level )
                        + readTex( src_linear_tex, block_x + idx, block_y + idy + off, dst_level );

        const float v = filter[offset+1];
        out += val * v;
    }

    float g   = filter[0];
    float val = readTex( src_linear_tex, block_x + idx, block_y + idy, dst_level );
    out += ( val * g );

    surf2DLayeredwrite( out, dst_data, (block_x+idx)*4, block_y+idy, dst_level, cudaBoundaryModeZero );
}

__global__ void vert_all_abs0(cudaTextureObject_t src_linear_tex,
                              cudaSurfaceObject_t dst_data,
                              int start_level,
                              int max_level)
{
    const int block_x = blockIdx.x * blockDim.y;
    const int block_y = blockIdx.y * blockDim.x;
    const int idx     = threadIdx.y;
    const int idy     = threadIdx.x;

    for( int dst_level=start_level; dst_level<max_level; dst_level++ )
    {
        const int    span   =  d_gauss.abs_o0.i_span[dst_level];
        const float* filter = &d_gauss.abs_o0.i_filter[dst_level*GAUSS_ALIGN];

        float out = 0;

        for( int offset = 1; offset<=span; offset += 2 ) {
            const float u    = filter[offset];
            const float off  = offset + ( 1.0f - u );
            const float val = readTex( src_linear_tex, block_x + idx, block_y + idy - off, dst_level )
                            + readTex( src_linear_tex, block_x + idx, block_y + idy + off, dst_level );

            const float v = filter[offset+1];
            out += val * v;
        }

        const float& g   = filter[0];
        float        val = readTex( src_linear_tex, block_x + idx, block_y + idy, dst_level );
        out += ( val * g );

        surf2DLayeredwrite( out, dst_data, (block_x+idx)*4, block_y+idy, dst_level, cudaBoundaryModeZero );
    }
}

} // namespace absoluteSourceInterpolated
} // namespace gauss
} // namespace popsift

