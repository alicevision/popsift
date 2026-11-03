/*
 * Copyright 2016-2017, Simula Research Laboratory
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
namespace gauss {
namespace absoluteSourceInterpolated {

__global__
static void horiz(LayeredReadTex src_linear_tex, cudaSurfaceObject_t dst_data, int dst_level)
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

__global__
static void vert(LayeredReadTex src_linear_tex, cudaSurfaceObject_t dst_data, int dst_level)
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

__global__
static void vert_abs0(LayeredReadTex src_linear_tex, cudaSurfaceObject_t dst_data, int dst_level)
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

__global__
static void vert_all_abs0(LayeredReadTex src_linear_tex,
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

namespace absoluteSource {

__global__
static void horiz(LayeredReadTex src_point_texture, cudaSurfaceObject_t dst_data, int dst_level)
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

    // Horizontal Gauss tap exchange via warp shuffles. The block is
    // (32,blockDim.y) and each threadIdx.y row convolves one image row. The
    // shuffle width is that row width, not the hardware warp size: on a 64-lane
    // wavefront two rows share a wavefront and a lane would otherwise pull a
    // neighbour from the wrong image row.
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

    surf2DLayeredwrite( out, dst_data, off_x*4, off_y, dst_level, cudaBoundaryModeZero );
}

__global__
static void vert(LayeredReadTex src_point_texture, cudaSurfaceObject_t dst_data, int dst_level)
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

__global__
static void vert_abs0(LayeredReadTex src_point_texture, cudaSurfaceObject_t dst_data, int dst_level)
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

__global__
static void vert_all_abs0(LayeredReadTex src_point_texture,
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

__host__
void Pyramid::horiz_from_prev_level( int octave, int level, cudaStream_t stream, GaussTableChoice useInterpolatedGauss )
{
    Octave&      oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    switch( useInterpolatedGauss )
    {
    case Interpolated_FromPrevious :
        {
            dim3 block( 128, 1 );
            dim3 grid;
            grid.x  = grid_divide( width,  128 );
            grid.y  = height;

            gauss::absoluteSourceInterpolated::horiz
                <<<grid,block,0,stream>>>
                ( oct_obj.getDataReadTexLinear( ),
                  oct_obj.getIntermediateSurface( ),
                  level );
        }
        break;
    case NotInterpolated_FromPrevious :
        {
            dim3 block( 32,  8 ); // most stable good perf on GTX 980 TI
            // similar speed: dim3 block( 32,  4 ); dim3 block( 32,  3 ); dim3 block( 32,  2 );

            dim3 grid;
            grid.x  = grid_divide( width,  32 );
            grid.y  = grid_divide( height, block.y );

            gauss::absoluteSource::horiz
                <<<grid,block,0,stream>>>
                ( oct_obj.getDataReadTexPoint( ),
                  oct_obj.getIntermediateSurface( ),
                  level );
        }
        break;
    case Interpolated_FromFirst :
    case NotInterpolated_FromFirst :
        POP_FATAL( "Case horizontal Gauss filtering from first level makes not sense in case horizontal Gauss filter from previous level" );
        break;
    default :
        POP_FATAL( "Missing case in horizontal Gauss filter from previous level" );
        break;
    }
    POP_SYNC_CHK;
}

__host__
void Pyramid::vert_from_interm( int octave, int level, cudaStream_t stream, GaussTableChoice useInterpolatedGauss )
{
    Octave& oct_obj = _octaves[octave];

    /* waiting for any events is not necessary, it's in the same stream as horiz
     */

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    switch( useInterpolatedGauss )
    {
    case Interpolated_FromPrevious :
        {
            dim3 block( 4, 32 );
            dim3 grid;
            grid.x = (unsigned int)grid_divide( width,  block.y );
            grid.y = (unsigned int)grid_divide( height, block.x );

            gauss::absoluteSourceInterpolated::vert
                <<<grid,block,0,stream>>>
                ( oct_obj.getIntermReadTexLinear( ),
                  oct_obj.getDataSurface( ),
                  level );
        }
        break;
    case Interpolated_FromFirst :
        {
            dim3 block( 4, 32 );
            dim3 grid;
            grid.x = (unsigned int)grid_divide( width,  block.y );
            grid.y = (unsigned int)grid_divide( height, block.x );

            gauss::absoluteSourceInterpolated::vert_abs0
                <<<grid,block,0,stream>>>
                ( oct_obj.getIntermReadTexLinear( ),
                  oct_obj.getDataSurface( ),
                  level );
        }
        break;
    case NotInterpolated_FromPrevious :
        {
            dim3 block( 64, 2 );
            dim3 grid;
            grid.x = (unsigned int)grid_divide( width,  block.x );
            grid.y = (unsigned int)grid_divide( height, block.y );

            gauss::absoluteSource::vert
                <<<grid,block,0,stream>>>
                ( oct_obj.getIntermReadTexPoint( ),
                  oct_obj.getDataSurface( ),
                  level );
        }
        break;
    case NotInterpolated_FromFirst :
        {
            dim3 block( 64, 2 );
            dim3 grid;
            grid.x = (unsigned int)grid_divide( width,  block.x );
            grid.y = (unsigned int)grid_divide( height, block.y );

            gauss::absoluteSource::vert_abs0
                <<<grid,block,0,stream>>>
                ( oct_obj.getIntermReadTexPoint( ),
                  oct_obj.getDataSurface( ),
                  level );
        }
        break;
    default :
        {
            POP_FATAL( "Missing case in vertical Gauss filter from intermediate buffer" );
        }
        break;
    }
    POP_SYNC_CHK;
}

__host__
void Pyramid::vert_all_from_interm( int octave, int start_level, int max_level, cudaStream_t stream, GaussTableChoice useInterpolatedGauss )
{
    Octave& oct_obj = _octaves[octave];

    /* waiting for any events is not necessary, it's in the same stream as horiz
     */

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    switch( useInterpolatedGauss )
    {
    case Interpolated_FromFirst :
        {
            dim3 block( 4, 32 );
            dim3 grid;
            grid.x = (unsigned int)grid_divide( width,  block.y );
            grid.y = (unsigned int)grid_divide( height, block.x );

            gauss::absoluteSourceInterpolated::vert_all_abs0
                <<<grid,block,0,stream>>>
                ( oct_obj.getIntermReadTexLinear( ),
                  oct_obj.getDataSurface( ),
                  start_level,
                  max_level );
        }
        break;
    case NotInterpolated_FromFirst :
        {
            dim3 block( 64, 2 );
            dim3 grid;
            grid.x = (unsigned int)grid_divide( width,  block.x );
            grid.y = (unsigned int)grid_divide( height, block.y );

            gauss::absoluteSource::vert_all_abs0
                <<<grid,block,0,stream>>>
                ( oct_obj.getIntermReadTexPoint( ),
                  oct_obj.getDataSurface( ),
                  start_level,
                  max_level );
        }
        break;
    case Interpolated_FromPrevious :
    case NotInterpolated_FromPrevious :
        POP_FATAL( "Case horizontal Gauss filtering from intermediate level makes not sense in case vertial-all Gauss filter from previous level" );
        break;
    default :
        POP_FATAL( "Missing case in vertical-all Gauss filter from intermediate buffer" );
        break;
    }
    POP_SYNC_CHK;
}

} // namespace popsift

