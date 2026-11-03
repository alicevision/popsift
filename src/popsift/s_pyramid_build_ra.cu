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
namespace normalizedSource {

__global__
static void horiz( cudaTextureObject_t src_linear_tex,
            cudaSurfaceObject_t dst_data,
            int                 dst_w,
            int                 dst_h,
            float               shift )
{
    const int    write_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int    write_y = blockIdx.y;

    if( write_x >= dst_w ) return;

    const int    span    =  d_gauss.dd.span[0];
    const float* filter  = &d_gauss.dd.filter[0];
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
static void horiz_level( cudaTextureObject_t src_linear_tex,
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
static void horiz_all( cudaTextureObject_t src_linear_tex,
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

__host__
void Pyramid::horiz_from_input_image( const Config& conf, ImageBase* base, cudaStream_t stream )
{
    Octave&   oct_obj = _octaves[0];

    const int width   = oct_obj.getWidth();
    const int height  = oct_obj.getHeight();

    dim3 block( 128, 1 );
    dim3 grid;
    grid.x  = grid_divide( width,  128 );
    grid.y  = height;

    const Config::SiftMode& mode = conf.getSiftMode();
    float shift  = 0.5f;

    if( mode == Config::PopSift || mode == Config::VLFeat ) {
        shift  = 0.5f * powf( 2.0f, conf.getUpscaleFactor() );
    }

    gauss::normalizedSource::horiz
        <<<grid,block,0,stream>>>
        ( base->getInputTexture(),
          oct_obj.getIntermediateSurface(),
          width,
          height,
          shift );

    POP_SYNC_CHK;
}

__host__
void Pyramid::horiz_level_from_input_image( const Config& conf, ImageBase* base, int octave, int level, cudaStream_t stream )
{
    if( octave != 0 )
    {
        POP_FATAL( "Unsupported parameter octave != 0" );
    }

    Octave&   oct_obj = _octaves[octave];

    const int width   = oct_obj.getWidth();
    const int height  = oct_obj.getHeight();

    dim3 block( 128, 1 );
    dim3 grid;
    grid.x  = grid_divide( width,  128 );
    grid.y  = height;

    const Config::SiftMode& mode = conf.getSiftMode();
    float shift  = 0.5f;

    if( octave == 0 && ( mode == Config::PopSift || mode == Config::VLFeat ) ) {
        shift  = 0.5f * powf( 2.0f, conf.getUpscaleFactor() - octave );
    }

    gauss::normalizedSource::horiz_level
        <<<grid,block,0,stream>>>
        ( base->getInputTexture(),
          oct_obj.getIntermediateSurface(),
          width,
          height,
          octave,
          level,
          shift );

    POP_SYNC_CHK;
}

__host__
void Pyramid::horiz_all_from_input_image( const Config& conf, ImageBase* base, int octave, int startlevel, int maxlevel, cudaStream_t stream )
{
    if( octave != 0 )
    {
        POP_FATAL( "Unsupported parameter octave != 0" );
    }

    Octave&      oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    dim3 block( 128, 1 );
    dim3 grid;
    grid.x  = grid_divide( width,  128 );
    grid.y  = height;

    const Config::SiftMode& mode = conf.getSiftMode();
    float shift  = 0.5f;

    if( mode == Config::PopSift || mode == Config::VLFeat ) {
        shift  = 0.5f * powf( 2.0f, conf.getUpscaleFactor() );
    }

    gauss::normalizedSource::horiz_all
        <<<grid,block,0,stream>>>
        ( base->getInputTexture(),
          oct_obj.getIntermediateSurface( ),
          width,
          height,
          shift,
          maxlevel );

    POP_SYNC_CHK;
}

} // namespace popsift

