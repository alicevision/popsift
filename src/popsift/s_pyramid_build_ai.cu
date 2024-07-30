/*
 * Copyright 2016-2017, Simula Research Laboratory
 *           2018-2024, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "gauss_filter.h"
#include "sift_pyramid.h"
#include "sift_constants.h"

namespace popsift {
namespace absoluteSourceInterpolated {

__global__ static void horiz(cudaTextureObject_t src_linear_tex, cudaSurfaceObject_t dst_data, int dst_level)
{
    const int    src_level = dst_level - 1;
    const int    span      =  d_gauss.inc.i_span[dst_level];
    const float* filter    = &d_gauss.inc.i_filter[dst_level*GAUSS_ALIGN];
    const int    idx       = blockIdx.x * blockDim.x + threadIdx.x;
    const int    idy       = blockIdx.x * blockDim.x + threadIdx.x;

    float out = 0.0f;

    for( int offset = 1; offset<=span; offset += 2 ) {
        const float u    = filter[offset];
        const float off  = offset + ( 1.0f - u );
        const float val = readTex( src_linear_tex, idx - off, idy, src_level )
                        + readTex( src_linear_tex, idx + off, idy, src_level );

        const float v = filter[offset+1];
        out += val * v;
    }
    const float& g  = filter[0];
    const float v3 = readTex( src_linear_tex, idx, idy, src_level );
    out += ( v3 * g );

    surf2DLayeredwrite( out, dst_data, idx*4, idy, dst_level, cudaBoundaryModeZero );
}

__global__ static void vert(cudaTextureObject_t src_linear_tex, cudaSurfaceObject_t dst_data, int dst_level)
{
    const int    span   =  d_gauss.inc.i_span[dst_level];
    const float* filter = &d_gauss.inc.i_filter[dst_level*GAUSS_ALIGN];
    const int    idx    = blockIdx.y * blockDim.y + threadIdx.y;
    const int    idy    = blockIdx.x * blockDim.x + threadIdx.x;

    float out = 0;

    for( int offset = 1; offset<=span; offset += 2 ) {
        const float u    = filter[offset];
        const float off  = offset + ( 1.0f - u );
        const float val = readTex( src_linear_tex, idx, idy - off, dst_level )
                        + readTex( src_linear_tex, idx, idy + off, dst_level );

        const float v = filter[offset+1];
        out += val * v;
    }

    const float g   = filter[0];
    const float v3 = readTex( src_linear_tex, idx, idy, dst_level );
    out += ( v3 * g );

    surf2DLayeredwrite( out, dst_data, idx*4, idy, dst_level, cudaBoundaryModeZero );
}

} // namespace absoluteSourceInterpolated

__host__
void Pyramid::horiz_from_prev_level_pairs( int octave, int level, cudaStream_t stream )
{
    Octave&      oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    dim3 block( 128, 1 );
    dim3 grid;
    grid.x  = grid_divide( width,  128 );
    grid.y  = height;

    absoluteSourceInterpolated::horiz
        <<<grid,block,0,stream>>>
        ( oct_obj.getDataTexLinear( ).tex,
          oct_obj.getIntermediateSurface( ),
          level );
    POP_SYNC_CHK;
}

__host__
void Pyramid::vert_from_interm_pairs( int octave, int level, cudaStream_t stream )
{
    Octave& oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    dim3 block( 4, 32 );
    dim3 grid;
    grid.y = (unsigned int)grid_divide( width,  block.y );
    grid.x = (unsigned int)grid_divide( height, block.x );

    absoluteSourceInterpolated::vert
        <<<grid,block,0,stream>>>
        ( oct_obj.getIntermDataTexLinear( ).tex,
          oct_obj.getDataSurface( ),
          level );
    POP_SYNC_CHK;
}

} // namespace popsift

