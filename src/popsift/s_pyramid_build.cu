/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "sift_pyramid.h"
#include "sift_constants.h"
#include "gauss_filter.h"
#include "common/debug_macros.h"
#include "common/assist.h"
#include "common/clamp.h"
#include "s_pyramid_build_aa.h"
#include "s_pyramid_build_ai.h"
#include "s_pyramid_build_ra.h"

#include <iostream>
#include <stdio.h>

/* It makes no sense whatsoever to change this value */
#define PREV_LEVEL 3

using std::cerr;
using std::endl;

namespace popsift {

namespace gauss {

__global__
void get_by_2_interpolate( cudaTextureObject_t src_data,
                           const int           src_level,
                           cudaSurfaceObject_t dst_data,
                           const int           dst_w,
                           const int           dst_h )
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if( idx >= dst_w ) return;
    if( idy >= dst_h ) return;

    const float val = readTex( src_data, 2.0f * idx + 1.0f, 2.0f * idy + 1.0f, src_level );

    surf2DLayeredwrite( val, dst_data, idx*4, idy, 0, cudaBoundaryModeZero ); // dst_data.ptr(idy)[idx] = val;
}

__global__
void get_by_2_pick_every_second( cudaTextureObject_t src_data,
                                 const int           src_w,
                                 const int           src_h,
                                 const int           src_level,
                                 cudaSurfaceObject_t dst_data,
                                 const int           dst_w,
                                 const int           dst_h )
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if( idx >= dst_w ) return;
    if( idy >= dst_h ) return;

    const int read_x = clamp( idx << 1, 0, src_w );
    const int read_y = clamp( idy << 1, 0, src_h );

    const float val = readTex( src_data, read_x, read_y, src_level );

    surf2DLayeredwrite( val, dst_data, idx*4, idy, 0, cudaBoundaryModeZero ); // dst_data.ptr(idy)[idx] = val;
}


__global__
void make_dog( cudaTextureObject_t src_data,
               cudaSurfaceObject_t dog_data,
               const int           w,
               const int           h,
               const int           max_level )
{
    const int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy   = blockIdx.y * blockDim.y + threadIdx.y;

    float a = readTex( src_data, idx, idy, 0 );
    for( int level=0; level<max_level-2; level++ )
    {
        const float b = readTex( src_data, idx, idy, level+1 );

        surf2DLayeredwrite( b-a, dog_data, idx*4, idy, level, cudaBoundaryModeZero );
        a = b;
    }
}

} // namespace gauss

__host__
inline void Pyramid::horiz_from_input_image( const Config& conf, Image* base, int octave, cudaStream_t stream )
{
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

    gauss::relativeSource::horiz
        <<<grid,block,0,stream>>>
        ( base->getInputTexture(),
          oct_obj.getIntermediateSurface(),
          oct_obj.getWidth(),
          oct_obj.getHeight(),
          octave,
          shift );

    POP_SYNC_CHK;
}


__host__
inline void Pyramid::downscale_from_prev_octave( int octave, cudaStream_t stream, Config::SiftMode mode )
{
    Octave&      oct_obj = _octaves[octave];
    Octave& prev_oct_obj = _octaves[octave-1];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    dim3 h_block( 64, 2 );
    dim3 h_grid;
    h_grid.x = (unsigned int)grid_divide( width,  h_block.x );
    h_grid.y = (unsigned int)grid_divide( height, h_block.y );

    switch( mode )
    {
    case Config::PopSift :
    case Config::VLFeat :
    case Config::OpenCV :
        gauss::get_by_2_pick_every_second
            <<<h_grid,h_block,0,stream>>>
            ( prev_oct_obj.getDataTexPoint( ),
              prev_oct_obj.getWidth(),
              prev_oct_obj.getHeight(),
              _levels-PREV_LEVEL,
              oct_obj.getDataSurface( ),
              oct_obj.getWidth(),
              oct_obj.getHeight() );

        POP_SYNC_CHK;
        break;
    default :
        gauss::get_by_2_interpolate
            <<<h_grid,h_block,0,stream>>>
            ( prev_oct_obj.getDataTexLinear( ).tex,
              _levels-PREV_LEVEL,
              oct_obj.getDataSurface( ),
              oct_obj.getWidth(),
              oct_obj.getHeight() );

        POP_SYNC_CHK;
        break;
    }
}

__host__
inline void Pyramid::horiz_from_prev_level( int octave, int level, cudaStream_t stream, bool useInterpolatedGauss )
{
    Octave&      oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    if( not useInterpolatedGauss ) {
        dim3 block( 32,  8 ); // most stable good perf on GTX 980 TI
        // similar speed: dim3 block( 32,  4 ); dim3 block( 32,  3 ); dim3 block( 32,  2 );

        dim3 grid;
        grid.x  = grid_divide( width,  32 );
        grid.y  = grid_divide( height, block.y );

        gauss::absoluteSource::horiz
            <<<grid,block,0,stream>>>
            ( oct_obj.getDataTexPoint( ),
              oct_obj.getIntermediateSurface( ),
              level );

        POP_SYNC_CHK;
    } else {
        dim3 block( 128, 1 );
        dim3 grid;
        grid.x  = grid_divide( width,  128 );
        grid.y  = height;

        gauss::absoluteSourceInterpolated::horiz
            <<<grid,block,0,stream>>>
            ( oct_obj.getDataTexLinear( ).tex,
              oct_obj.getIntermediateSurface( ),
              level );

        POP_SYNC_CHK;
    }
}

__host__
inline void Pyramid::vert_from_interm( int octave, int level, cudaStream_t stream, bool useInterpolatedGauss )
{
    Octave& oct_obj = _octaves[octave];

    /* waiting for any events is not necessary, it's in the same stream as horiz
     */

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    if( not useInterpolatedGauss ) {
        dim3 block( 64, 2 );
        dim3 grid;
        grid.x = (unsigned int)grid_divide( width,  block.x );
        grid.y = (unsigned int)grid_divide( height, block.y );

        gauss::absoluteSource::vert
            <<<grid,block,0,stream>>>
            ( oct_obj.getIntermDataTexPoint( ),
              oct_obj.getDataSurface( ),
              level );
        POP_SYNC_CHK;
    } else {
        dim3 block( 4, 32 );
        dim3 grid;
        grid.x = (unsigned int)grid_divide( width,  block.y );
        grid.y = (unsigned int)grid_divide( height, block.x );

        gauss::absoluteSourceInterpolated::vert
            <<<grid,block,0,stream>>>
            ( oct_obj.getIntermDataTexLinear( ).tex,
              oct_obj.getDataSurface( ),
              level );
        POP_SYNC_CHK;
    }
}

__host__
inline void Pyramid::dogs_from_blurred( int octave, int max_level, cudaStream_t stream )
{
    Octave&      oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    dim3 block( 128, 2 );
    dim3 grid;
    grid.x = grid_divide( width,  block.x );
    grid.y = grid_divide( height, block.y );
    grid.z = 1;

    gauss::make_dog
        <<<grid,block,0,stream>>>
        ( oct_obj.getDataTexPoint( ),
          oct_obj.getDogSurface( ),
          oct_obj.getWidth(),
          oct_obj.getHeight(),
          max_level );
    POP_SYNC_CHK;
}

/*************************************************************
 * V11: host side
 *************************************************************/
__host__
void Pyramid::build_pyramid( const Config& conf, Image* base )
{
#if (PYRAMID_PRINT_DEBUG==1)
    cerr << "Entering " << __FUNCTION__ << " with base image "  << endl
         << "    type size         : " << base->type_size << endl
         << "    aligned byte size : " << base->a_width << "x" << base->a_height << endl
         << "    pitch size        : " << base->pitch << "x" << base->a_height << endl
         << "    original byte size: " << base->u_width << "x" << base->u_height << endl
         << "    aligned pix size  : " << base->a_width/base->type_size << "x" << base->a_height << endl
         << "    original pix size : " << base->u_width/base->type_size << "x" << base->u_height << endl;
#endif // (PYRAMID_PRINT_DEBUG==1)

    cudaDeviceSynchronize();

    for( uint32_t octave=0; octave<_num_octaves; octave++ ) {
        Octave&      oct_obj = _octaves[octave];
        cudaStream_t stream  = oct_obj.getStream();

        if( ( conf.getScalingMode() == Config::ScaleDirect ) &&
            ( conf.getGaussMode() == Config::Fixed9 || conf.getGaussMode() == Config::Fixed15 ) ) {
            if( octave == 0 ) {
                make_octave( conf, base, oct_obj, stream, true );
            } else {
                horiz_from_input_image( conf, base, octave, stream );
                vert_from_interm( octave, 0, stream, DontUseInterpolatedGauss );
                make_octave( conf, base, oct_obj, stream, false );
            }
        } else if( conf.getGaussMode() == Config::Fixed9 || conf.getGaussMode() == Config::Fixed15 ) {
            if( octave == 0 ) {
                make_octave( conf, base, oct_obj, stream, true );
            } else {
                Octave& prev_oct_obj = _octaves[octave-1];
                cuda::event_wait( prev_oct_obj.getEventScaleDone(), stream, __FILE__, __LINE__ );

                downscale_from_prev_octave( octave, stream, conf.getSiftMode() );
                make_octave( conf, base, oct_obj, stream, false );
            }

            cuda::event_record( oct_obj.getEventScaleDone(), stream, __FILE__, __LINE__ );
        } else if( conf.getScalingMode() == Config::ScaleDirect ) {
            bool useGauss = ( conf.getGaussMode() == Config::VLFeat_Relative ) ? UseInterpolatedGauss
                                                                               : DontUseInterpolatedGauss;
            for( int level=0; level<_levels; level++ ) {
                const int width  = oct_obj.getWidth();
                const int height = oct_obj.getHeight();

                if( level == 0 )
                {
                    horiz_from_input_image( conf, base, octave, stream );
                    vert_from_interm( octave, level, stream, useGauss );
                }
                else
                {
                    horiz_from_prev_level( octave, level, stream, DontUseInterpolatedGauss );
                    vert_from_interm( octave, level, stream, useGauss );
                }
            }
        } else {
            bool useGauss = ( conf.getGaussMode() == Config::VLFeat_Relative ) ? UseInterpolatedGauss
                                                                               : DontUseInterpolatedGauss;
            for( int level=0; level<_levels; level++ ) {
                const int width  = oct_obj.getWidth();
                const int height = oct_obj.getHeight();

                if( level == 0 )
                {
                    if( octave == 0 )
                    {
                            horiz_from_input_image( conf, base, 0, stream );
                            vert_from_interm( octave, 0, stream, useGauss );
                    }
                    else
                    {
                        Octave& prev_oct_obj = _octaves[octave-1];
                        cuda::event_wait( prev_oct_obj.getEventScaleDone(), stream, __FILE__, __LINE__ );

                        downscale_from_prev_octave( octave, stream, conf.getSiftMode() );
                    }
                }
                else
                {
                    horiz_from_prev_level( octave, level, stream, useGauss );
                    vert_from_interm( octave, level, stream, useGauss );

                    if( level == _levels - PREV_LEVEL ) {
                        cuda::event_record( oct_obj.getEventScaleDone(), stream, __FILE__, __LINE__ );
                    }
                }
            }

        }
    }
    // for( int octave=_num_octaves-1; octave>=0; octave-- )
    for( int octave=0; octave<_num_octaves; octave++ )
    {
        if( conf.getGaussMode() == Config::Fixed9 || conf.getGaussMode() == Config::Fixed15 ) {
        } else {
            Octave&      oct_obj = _octaves[octave];
            cudaStream_t stream  = oct_obj.getStream();
            dogs_from_blurred( octave, _levels, stream );
        }
    }
    for( int octave=0; octave<_num_octaves; octave++ )
    // for( int octave=_num_octaves-1; octave>=0; octave-- )
    {
        Octave&      oct_obj = _octaves[octave];
        cudaStream_t stream  = oct_obj.getStream();
        cudaStreamSynchronize( stream );
    }
}

} // namespace popsift

