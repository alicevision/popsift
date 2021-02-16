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

namespace absoluteSource
{
template<class Weight>
__global__ void horiz(cudaTextureObject_t src_point_tex, cudaSurfaceObject_t dst_data, int dst_level);

template<class Weight>
__global__ void vert(cudaTextureObject_t src_point_tex, cudaSurfaceObject_t dst_data, int dst_level);

template<class Weight>
__global__ void vert_all(cudaTextureObject_t src_point_tex,
                         cudaSurfaceObject_t dst_data,
                         int start_level,
                         int max_level);
} // namespace absoluteSource

namespace absoluteSourceInterpolatedFilter
{
template<class Weight>
__global__ void horiz(cudaTextureObject_t src_linear_tex, cudaSurfaceObject_t dst_data, int dst_level);

template<class Weight>
__global__ void vert(cudaTextureObject_t src_linear_tex, cudaSurfaceObject_t dst_data, int dst_level);

template<class Weight>
__global__ void vert_all(cudaTextureObject_t src_linear_tex,
                         cudaSurfaceObject_t dst_data,
                         int start_level,
                         int max_level);
} // namespace absoluteSourceInterpolatedFilter

/*********************************************************************************
 * Gaussian filter options
 *********************************************************************************/

enum FilterStyle
{
    DirectFilter,      // for kernels that use every cell of the Gaussian filter
    InterpolatedFilter // for kernels that use pixel interpolation for fewer mult-ops
};

enum FilterSrc
{
    Incremental, // for narrow kernels that blur incrementally
    Level0       // for wide kernels that blur from level 0 of the octave
};

template<enum FilterStyle,enum FilterSrc>
struct Weight
{
    __device__ inline static int getSpan( int lvl );
    __device__ inline static const float* getFilter( int lvl );
};

template<> struct Weight<DirectFilter,Incremental>
{
    __device__ inline static int getSpan( int lvl )
    { return d_gauss.inc.span[lvl]; }

    __device__ inline static const float* getFilter( int lvl )
    { return &d_gauss.inc.filter[lvl*GAUSS_ALIGN]; }
};

template<> struct Weight<InterpolatedFilter,Incremental>
{
    __device__ inline static int getSpan( int lvl )
    { return d_gauss.inc.i_span[lvl]; }

    __device__ inline static const float* getFilter( int lvl )
    { return &d_gauss.inc.i_filter[lvl*GAUSS_ALIGN]; }
};

template<> struct Weight<DirectFilter,Level0>
{
    __device__ inline static int getSpan( int lvl )
    { return d_gauss.abs_o0.span[lvl]; }

    __device__ inline static const float* getFilter( int lvl )
    { return &d_gauss.abs_o0.filter[lvl*GAUSS_ALIGN]; }
};

template<> struct Weight<InterpolatedFilter,Level0>
{
    __device__ inline static int getSpan( int lvl )
    { return d_gauss.abs_o0.i_span[lvl]; }

    __device__ inline static const float* getFilter( int lvl )
    { return &d_gauss.abs_o0.i_filter[lvl*GAUSS_ALIGN]; }
};


/*********************************************************************************
 * AbsoluteSource
 *********************************************************************************/

void AbsoluteSource::horiz( cudaTextureObject_t src,
                            cudaSurfaceObject_t dst,
                            int                 level )
{
    dim3 block( 32,  8 ); // most stable good perf on GTX 980 TI
    // similar speed: dim3 block( 32,  4 ); dim3 block( 32,  3 ); dim3 block( 32,  2 );

    dim3 grid;
    grid.x  = grid_divide( _width,  block.x );
    grid.y  = grid_divide( _height, block.y );

    typedef Weight<DirectFilter,Incremental> filter_t;

    absoluteSource::horiz<filter_t>
        <<<grid,block,0,_stream>>>
        ( src,
          dst,
          level );
}

void AbsoluteSource::vert( cudaTextureObject_t src,
                           cudaSurfaceObject_t dst,
                           int                 level )
{
    dim3 block( 64, 2 );
    dim3 grid;
    grid.x = (unsigned int)grid_divide( _width,  block.x );
    grid.y = (unsigned int)grid_divide( _height, block.y );

    typedef Weight<DirectFilter,Incremental> filter_t;

    absoluteSource::vert<filter_t>
        <<<grid,block,0,_stream>>>
        ( src,
          dst,
          level );
}

/*********************************************************************************
 * AbsoluteSourceLevel0
 *********************************************************************************/

void AbsoluteSourceLevel0::vert( cudaTextureObject_t src,
                                 cudaSurfaceObject_t dst,
                                 int                 level )
{
    dim3 block( 64, 2 );
    dim3 grid;
    grid.x = (unsigned int)grid_divide( _width,  block.x );
    grid.y = (unsigned int)grid_divide( _height, block.y );

    typedef Weight<DirectFilter,Level0> filter_t;

    absoluteSource::vert<filter_t>
        <<<grid,block,0,_stream>>>
        ( src,
          dst,
          level );
}

void AbsoluteSourceLevel0::vert_all( cudaTextureObject_t src,
                                     cudaSurfaceObject_t dst,
                                     int                 start_level,
                                     int                 max_level )
{
    dim3 block( 64, 2 );
    dim3 grid;
    grid.x = (unsigned int)grid_divide( _width,  block.x );
    grid.y = (unsigned int)grid_divide( _height, block.y );

    typedef Weight<DirectFilter,Level0> filter_t;

    absoluteSource::vert_all<filter_t>
        <<<grid,block,0,_stream>>>
        ( src,
          dst,
          start_level,
          max_level );
}

/*********************************************************************************
 * AbsoluteSourceInterpolatedFilter
 *********************************************************************************/

void AbsoluteSourceInterpolatedFilter::horiz( cudaTextureObject_t src,
                                              cudaSurfaceObject_t dst,
                                              int                 level )
{
    dim3 block( 128, 1 );

    dim3 grid;
    grid.x  = grid_divide( _width,  block.x );
    grid.y  = grid_divide( _height, block.y );

    typedef Weight<InterpolatedFilter,Incremental> filter_t;

    absoluteSourceInterpolatedFilter::horiz<filter_t>
        <<<grid,block,0,_stream>>>
        ( src,
          dst,
          level );
}

void AbsoluteSourceInterpolatedFilter::vert( cudaTextureObject_t src,
                                             cudaSurfaceObject_t dst,
                                             int                 level )
{
    dim3 block( 4, 32 );
    dim3 grid;
    grid.x = (unsigned int)grid_divide( _width,  block.x );
    grid.y = (unsigned int)grid_divide( _height, block.y );

    typedef Weight<InterpolatedFilter,Incremental> filter_t;

    absoluteSourceInterpolatedFilter::vert<filter_t>
        <<<grid,block,0,_stream>>>
        ( src,
          dst,
          level );
}

/*********************************************************************************
 * AbsoluteSourceInterpolatedFilterLevel0
 *********************************************************************************/

void AbsoluteSourceInterpolatedFilterLevel0::vert( cudaTextureObject_t src,
                                                   cudaSurfaceObject_t dst,
                                                   int                 level )
{
    dim3 block( 4, 32 );
    dim3 grid;
    grid.x = (unsigned int)grid_divide( _width,  block.x );
    grid.y = (unsigned int)grid_divide( _height, block.y );

    typedef Weight<InterpolatedFilter,Level0> filter_t;

    absoluteSourceInterpolatedFilter::vert<filter_t>
        <<<grid,block,0,_stream>>>
        ( src,
          dst,
          level );
}

void AbsoluteSourceInterpolatedFilterLevel0::vert_all( cudaTextureObject_t src,
                                                       cudaSurfaceObject_t dst,
                                                       int                 start_level,
                                                       int                 max_level )
{
    dim3 block( 4, 32 );
    dim3 grid;
    grid.x = (unsigned int)grid_divide( _width,  block.x );
    grid.y = (unsigned int)grid_divide( _height, block.y );

    typedef Weight<InterpolatedFilter,Level0> filter_t;

    absoluteSource::vert_all<filter_t>
        <<<grid,block,0,_stream>>>
        ( src,
          dst,
          start_level,
          max_level );
}

/*********************************************************************************
 * kernels in namespace absoluteSource
 *********************************************************************************/

namespace absoluteSource {

template<class Weight>
__global__ void horiz(cudaTextureObject_t src_point_texture, cudaSurfaceObject_t dst_data, int dst_level)
{
#ifndef HORIZ_WITH_SHUFFLE
    const int    src_level = dst_level - 1;
    const int    span      = Weight::getSpan( dst_level );
    const float* filter    = Weight::getFilter( dst_level );
    const int off_x   = blockIdx.x * blockDim.x + threadIdx.x;
    const int off_y   = blockIdx.y * blockDim.y + threadIdx.y;

    float out = 0.0f;

    for( int offset = span; offset>0; offset-- ) {
        const float A = readTex( src_point_texture, off_x - offset, off_y, src_level );
        const float B = readTex( src_point_texture, off_x + offset, off_y, src_level );
        const float g = filter[offset];
        out += ( A + B ) * g;
    }

    const float g = filter[0];
    const float C = readTex( src_point_texture, off_x, off_y, src_level );
    out += ( C * g );

    surf2DLayeredwrite( out, dst_data, off_x*4, off_y, dst_level, cudaBoundaryModeZero );
#else
    const int    src_level = dst_level - 1;
    const int    span      = Weight::getSpan( dst_level );
    const float* filter    = Weight::getFilter( dst_level );

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
        const float D1 = popsift::shuffle_down( A, shiftval );
        const float D2 = popsift::shuffle_up  ( C, span - shiftval );
        const float D  = threadIdx.x < (32 - shiftval) ? D1 : D2;
        const float E1 = popsift::shuffle_up  ( B, shiftval );
        const float E2 = popsift::shuffle_down( C, span - shiftval );
        const float E  = threadIdx.x > shiftval        ? E1 : E2;
        g = filter[offset];
        out += ( D + E ) * g;
    }

    surf2DLayeredwrite( out, dst_data, off_x*4, off_y, dst_level, cudaBoundaryModeZero );
#endif
}

template<class Weight>
__global__ void vert(cudaTextureObject_t src_point_texture, cudaSurfaceObject_t dst_data, int dst_level)
{
    const int    span      = Weight::getSpan( dst_level );
    const float* filter    = Weight::getFilter( dst_level );
    const int off_x   = blockIdx.x * blockDim.x + threadIdx.x;
    const int off_y   = blockIdx.y * blockDim.y + threadIdx.y;

    float out = 0.0f;

    for( int offset = span; offset>0; offset-- ) {
        const float A = readTex( src_point_texture, off_x, off_y - offset, dst_level );
        const float B = readTex( src_point_texture, off_x, off_y + offset, dst_level );
        const float g = filter[offset];
        out += ( A + B ) * g;
    }

    const float g = filter[0];
    const float C = readTex( src_point_texture, off_x, off_y, dst_level );
    out += ( C * g );

    surf2DLayeredwrite( out, dst_data, off_x*4, off_y, dst_level, cudaBoundaryModeZero );
}

template<class Weight>
__global__ void vert_all( cudaTextureObject_t src_point_texture,
                          cudaSurfaceObject_t dst_data,
                          int start_level,
                          int max_level )
{
    const int off_x   = blockIdx.x * blockDim.x + threadIdx.x;
    const int off_y   = blockIdx.y * blockDim.y + threadIdx.y;

    for( int dst_level=start_level; dst_level<max_level; dst_level++ )
    {
        const int    span      = Weight::getSpan( dst_level );
        const float* filter    = Weight::getFilter( dst_level );

        float out = 0.0f;

        for( int offset = span; offset>0; offset-- ) {
            const float A = readTex( src_point_texture, off_x, off_y - offset, dst_level );
            const float B = readTex( src_point_texture, off_x, off_y + offset, dst_level );
            const float g  = filter[offset];
            out += ( A + B ) * g;
        }

        const float g  = filter[0];
        const float C = readTex( src_point_texture, off_x, off_y, dst_level );
        out += ( C * g );

        surf2DLayeredwrite( out, dst_data, off_x*4, off_y, dst_level, cudaBoundaryModeZero );
    }
}

} // namespace absoluteSource

/*********************************************************************************
 * kernels in namespace absoluteSourceInterpolatedFilter
 *********************************************************************************/

namespace absoluteSourceInterpolatedFilter
{
template<class Weight>
__global__ void horiz(cudaTextureObject_t src_linear_tex, cudaSurfaceObject_t dst_data, int dst_level)
{
    const int    src_level = dst_level - 1;
    const int    span      = Weight::getSpan( dst_level );
    const float* filter    = Weight::getFilter( dst_level );
    const int    off_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int    off_y     = blockIdx.y * blockDim.y + threadIdx.y;

    float out = 0.0f;

    for( int offset = 1; offset<=span; offset += 2 ) {
        // fetch fractional weights for two neighbouring cells
        const float& u   = filter[offset];
        const float  off = offset + ( 1.0f - u );

        // let the texture engine handle the 2 multiplications for weighting
        const float  A   = readTex( src_linear_tex, off_x - off, off_y, src_level );
        const float  B   = readTex( src_linear_tex, off_x + off, off_y, src_level );

        // multiply with the combined weight for both cells
        const float& v   = filter[offset+1];
        out += ( A + B ) * v;
    }
    const float& g = filter[0];
    const float  C = readTex( src_linear_tex, off_x, off_y, src_level );
    out += ( C * g );

    surf2DLayeredwrite( out, dst_data, off_x*4, off_y, dst_level, cudaBoundaryModeZero );
}

template<class Weight>
__global__ void vert(cudaTextureObject_t src_linear_tex, cudaSurfaceObject_t dst_data, int dst_level)
{
    const int    span      = Weight::getSpan( dst_level );
    const float* filter    = Weight::getFilter( dst_level );
    const int    off_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int    off_y     = blockIdx.y * blockDim.y + threadIdx.y;

    float out = 0.0f;

    for( int offset = 1; offset<=span; offset += 2 ) {
        const float& u   = filter[offset];
        const float  off = offset + ( 1.0f - u );
        const float A    = readTex( src_linear_tex, off_x, off_y - off, dst_level );
        const float B    = readTex( src_linear_tex, off_x, off_y + off, dst_level );
        const float& v   = filter[offset+1];
        out += ( A + B ) * v;
    }

    const float& g = filter[0];
    const float  C = readTex( src_linear_tex, off_x, off_y, dst_level );
    out += ( C * g );

    surf2DLayeredwrite( out, dst_data, off_x*4, off_y, dst_level, cudaBoundaryModeZero );
}

template<class Weight>
__global__ void vert_all(cudaTextureObject_t src_linear_tex,
                         cudaSurfaceObject_t dst_data,
                         int start_level,
                         int max_level)
{
    const int    off_x     = blockIdx.x * blockDim.x + threadIdx.x;
    const int    off_y     = blockIdx.y * blockDim.y + threadIdx.y;

    for( int dst_level=start_level; dst_level<max_level; dst_level++ )
    {
        const int    span   = Weight::getSpan( dst_level );
        const float* filter = Weight::getFilter( dst_level );

        float out = 0;

        for( int offset = 1; offset<=span; offset += 2 ) {
            const float& u  = filter[offset];
            const float off = offset + ( 1.0f - u );
            const float A   = readTex( src_linear_tex, off_x, off_y - off, dst_level );
            const float B   = readTex( src_linear_tex, off_x, off_y + off, dst_level );
            const float& v  = filter[offset+1];
            out += ( A + B ) * v;
        }

        const float& g = filter[0];
        const float  C = readTex( src_linear_tex, off_x, off_y, dst_level );
        out += ( C * g );

        surf2DLayeredwrite( out, dst_data, off_x*4, off_y, dst_level, cudaBoundaryModeZero );
    }
}

} //  absoluteSourceInterpolatedFilter

} // namespace gauss
} // namespace popsift

