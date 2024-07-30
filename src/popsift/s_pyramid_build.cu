/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/clamp.h"
#include "common/debug_macros.h"
#include "gauss_filter.h"
#include "sift_constants.h"
#include "sift_pyramid.h"

#include <cstdio>
#include <iostream>

/* It makes no sense whatsoever to change this value */
#define PREV_LEVEL 3

using std::cout;
using std::cerr;
using std::endl;

namespace popsift {

namespace gauss {

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
    for( int level=0; level<max_level-1; level++ )
    {
        const float b = readTex( src_data, idx, idy, level+1 );

        surf2DLayeredwrite( b-a, dog_data, idx*4, idy, level, cudaBoundaryModeZero );
        a = b;
    }
}

} // namespace gauss

__host__
inline void Pyramid::downscale_from_prev_octave( int octave, cudaStream_t stream )
{
    Octave&      oct_obj = _octaves[octave];
    Octave& prev_oct_obj = _octaves[octave-1];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    dim3 h_block( 64, 2 );
    dim3 h_grid;
    h_grid.x = (unsigned int)grid_divide( width,  h_block.x );
    h_grid.y = (unsigned int)grid_divide( height, h_block.y );

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
}

__host__
inline void Pyramid::horiz_from_prev_level( int octave, int level, cudaStream_t stream, GaussTableChoice useInterpolatedGauss )
{
    switch( useInterpolatedGauss )
    {
    case Interpolated_FromPrevious :
        horiz_from_prev_level_pairs( octave, level, stream );
        break;
    case NotInterpolated_FromPrevious :
        horiz_from_prev_level_basic( octave, level, stream );
        break;
    default :
        POP_FATAL( "Missing case in horizontal Gauss filter from previous level" );
        break;
    }
}

__host__
inline void Pyramid::vert_from_interm( int octave, int level, cudaStream_t stream, GaussTableChoice useInterpolatedGauss )
{
    Octave& oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    switch( useInterpolatedGauss )
    {
    case Interpolated_FromPrevious :
        vert_from_interm_pairs( octave, level, stream );
        break;
    case NotInterpolated_FromPrevious :
        vert_from_interm_basic( octave, level, stream );
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
inline void Pyramid::dogs_from_blurred( int octave, int max_level, cudaStream_t stream )
{
    Octave&      oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    dim3 block( 1024, 1 );
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
void Pyramid::build_pyramid( const Config& conf, ImageBase* base )
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

    GaussTableChoice gaussTableChoice;

    if( conf.getGaussMode() == Config::VLFeat_Relative )
        gaussTableChoice = Interpolated_FromPrevious;
    else
        gaussTableChoice = NotInterpolated_FromPrevious;

    for( uint32_t octave=0; octave<_num_octaves; octave++ )
    {
        Octave&      oct_obj = _octaves[octave];
        cudaStream_t stream  = oct_obj.getStream();

        for( int level=0; level<_levels; level++ )
        {
            if( level == 0 )
            {
                if( octave == 0 )
                {
                    horiz_from_input_image( conf, base, stream );
                    vert_from_interm( octave, 0, stream, gaussTableChoice );
                }
                else
                {
                    Octave& prev_oct_obj = _octaves[octave-1];
                    cuda::event_wait( prev_oct_obj.getEventScaleDone(), stream, __FILE__, __LINE__ );
                    downscale_from_prev_octave( octave, stream );
                }
            }
            else
            {
                horiz_from_prev_level( octave, level, stream, gaussTableChoice );
                vert_from_interm( octave, level, stream, gaussTableChoice );
                if( level == _levels - PREV_LEVEL )
                {
                    cuda::event_record( oct_obj.getEventScaleDone(), stream, __FILE__, __LINE__ );
                }
            }
        }
    }

    for( int octave=0; octave<_num_octaves; octave++ )
    {
        Octave&      oct_obj = _octaves[octave];
        cudaStream_t stream  = oct_obj.getStream();
        dogs_from_blurred( octave, _levels, stream );
    }

    for( int octave=0; octave<_num_octaves; octave++ )
    {
        Octave&      oct_obj = _octaves[octave];
        cudaStream_t stream  = oct_obj.getStream();
        cudaStreamSynchronize( stream );
    }
}

} // namespace popsift

