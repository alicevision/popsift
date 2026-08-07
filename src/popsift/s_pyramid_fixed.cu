/*
 * Copyright 2017, Simula Research Laboratory
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

#include <iostream>
#include <cstdio>

namespace popsift {

namespace gauss {

namespace fixedSpan {

template<int SHIFT>
__device__
inline float octave_fixed_horiz( float fval, const float* filter )
{
    /* Example:
     * SHIFT is 4
     * input  fval of thread N is extracted from image index N-4
     * output fval of thread N should be filtered sum from N-4 to N+4
     */
    float out = fval * filter[0];
    #pragma unroll
    for( int i=1; i<=SHIFT; i++ ) {
        float val  = popsift::shuffle_up( fval, i ) + popsift::shuffle_down( fval, i );
        out += val * filter[i];
    }

    fval = popsift::shuffle_down( out, SHIFT );

    return fval;
}

namespace absoluteTexAddress {
/* read from point-addressable texture of image from previous octave */

template<int SHIFT>
__device__
inline float octave_fixed_vert( cudaTextureObject_t src_data, int idx, int idy, int level, const float* filter )
{
    /* Input thread N takes as input the (idx,idy) position of the pixel that it
     * will eventually write (The 2*SHIFT rightmost threads will not write anything).
     * Thread N computes and returns the vertical filter at position N-SHIFT.
     */
    float       val    = readTex( src_data, idx-SHIFT, idy, level );

    float       fval   = val * filter[0];
    #pragma unroll
    for( int i=1; i<=SHIFT; i++ ) {
        val   = readTex( src_data, idx-SHIFT, idy-i, level )
              + readTex( src_data, idx-SHIFT, idy+i, level );
        fval += val * filter[i];
    }

    return fval;
}

template<int SHIFT, int WIDTH, int HEIGHT, int LEVELS>
__global__
void octave_fixed( cudaTextureObject_t src_data,
                   cudaSurfaceObject_t dst_data,
                   const int           w,
                   const int           h,
                   cudaSurfaceObject_t dog_data )
{
    const int IDx   = threadIdx.x;
    const int IDy   = threadIdx.y;
    const int IDz   = threadIdx.z;
    const int level = IDz + 1;

    const float* filter = &d_gauss.abs_oN.filter[level*GAUSS_ALIGN];

    const int idx = blockIdx.x * WIDTH      + IDx;
    const int idy = blockIdx.y * blockDim.y + IDy;

    float fval;
    
    fval = octave_fixed_vert<SHIFT>( src_data, idx, idy, 0, filter );

    fval = octave_fixed_horiz<SHIFT>( fval, filter );

    __shared__ float lx_val[HEIGHT][WIDTH][LEVELS];

    if( IDx < WIDTH ) {
        lx_val[IDy][IDx][IDz] = fval;
    }
    __syncthreads();

    if( IDx < WIDTH ) {
        const float l0_val = readTex( src_data, idx, idy, 0 );
        const float dogval = ( IDz == 0 )
                           ? fval - l0_val
                           : fval - lx_val[IDy][IDx][IDz-1];

        const bool i_write = ( idx < w && idy < h );

        if( i_write ) {
            surf2DLayeredwrite( fval, dst_data,
                                idx*4, idy,
                                IDz + 1,
                                cudaBoundaryModeZero );

            surf2DLayeredwrite( dogval, dog_data,
                                idx*4, idy,
                                IDz,
                                cudaBoundaryModeZero );
        }
    }
}

} // namespace absoluteTexAddress

namespace relativeTexAddress {
/* read from ratio-addressable texture of input image */

/* reading from the texture laid over the input image */
template<int SHIFT>
__device__
inline float octave_fixed_vert( cudaTextureObject_t src_data, int idx, int idy, const float mul_w, const float mul_h, float tshift, const float* filter )
{
    /* Like above, but reading uses relative input image positions */
    const float xpos = ( idx - SHIFT + tshift ) * mul_w;
    const float ypos = ( idy + tshift ) * mul_h;
    float       val  = tex2D<float>( src_data, xpos, ypos );

    float       fval = val * filter[0];
    #pragma unroll
    for( int i=1; i<=SHIFT; i++ ) {
        val  = tex2D<float>( src_data, xpos, ypos - i * mul_h );
        val += tex2D<float>( src_data, xpos, ypos + i * mul_h );
        fval += val * filter[i];
    }

    return fval;
}

template<int SHIFT, int WIDTH, int HEIGHT, int LEVELS>
__global__
void octave_fixed( cudaTextureObject_t src_data,
                   cudaSurfaceObject_t dst_data,
                   cudaSurfaceObject_t dog_data,
                   const int           w,
                   const int           h,
                   const float         tshift )
{
    const int IDx   = threadIdx.x;
    const int IDy   = threadIdx.y;
    const int level = threadIdx.z;

    const float* filter = &d_gauss.abs_o0.filter[level*GAUSS_ALIGN];

    const int idx = blockIdx.x * WIDTH      + IDx;
    const int idy = blockIdx.y * blockDim.y + IDy;

    const float mul_w  = __frcp_rn( float(w) );
    const float mul_h  = __frcp_rn( float(h) );
    float fval;

    fval = octave_fixed_vert<SHIFT>( src_data, idx, idy, mul_w, mul_h, tshift, filter );

    fval = octave_fixed_horiz<SHIFT>( fval, filter );

    fval *= 255.0f; // don't forget to upscale

    __shared__ float lx_val[HEIGHT][WIDTH][LEVELS];

    if( IDx < WIDTH ) {
        lx_val[IDy][IDx][level] = fval;
    }
    __syncthreads();

    const bool i_write = ( idx < w && idy < h );

    if( IDx < WIDTH && i_write ) {
            // destination.ptr(idy)[idx] = fval;
            surf2DLayeredwrite( fval, dst_data,
                                idx*4, idy,
                                level,
                                cudaBoundaryModeZero );

        if( level > 0 ) {
            float dogval = fval - lx_val[IDy][IDx][level-1];
            // left side great
            // right side buggy
            surf2DLayeredwrite( dogval, dog_data,
                                idx*4, idy,
                                level-1,
                                cudaBoundaryModeZero );
        }
    }
}

} // namespace relativeTexAddress

} // namespace fixedSpan

} // namespace gauss

template<int SHIFT, bool OCT_0, int LEVELS>
__host__
inline void make_octave_sub( const Config& conf, ImageBase* base, Octave& oct_obj, cudaStream_t stream )
{
    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    if( OCT_0 ) {
        const int x_size = 32;
        const int l_conf = LEVELS;
        const int w_conf = x_size - 2 * SHIFT;
        const int h_conf = 1; // 1024 / ( x_size * l_conf );
        dim3 block( x_size, h_conf, l_conf );
        dim3 grid;
        grid.x = grid_divide( width, w_conf );
        grid.y = grid_divide( height, block.y );

        assert( block.x * block.y * block.z < 1024 );
        
        // cerr << "calling relative with " << block.x * block.y * block.z << " threads per block" << endl
             // << "                 and  " << grid.x * grid.y * grid.z << " blocks" << endl;

        const float tshift = 0.5f * powf( 2.0f, conf.getUpscaleFactor() );

        gauss::fixedSpan::relativeTexAddress::octave_fixed
            <SHIFT,w_conf,h_conf,l_conf>
            <<<grid,block,0,stream>>>
            ( base->getInputTexture( ),
              oct_obj.getDataSurface( ),
              oct_obj.getDogSurface( ),
              oct_obj.getWidth(),
              oct_obj.getHeight(),
              tshift );
    } else {
        const int x_size = 32;
        const int l_conf = LEVELS-1;
        const int w_conf = x_size - 2 * SHIFT;
        const int h_conf = 1024 / ( x_size * l_conf );
        dim3 block( x_size, h_conf, l_conf );
        dim3 grid;
        grid.x = grid_divide( width, w_conf );
        grid.y = grid_divide( height, block.y );

        assert( block.x * block.y * block.z < 1024 );

        // cerr << "calling absolute with " << block.x * block.y * block.z << " threads per block" << endl
             // << "                 and  " << grid.x * grid.y * grid.z << " blocks" << endl;

        gauss::fixedSpan::absoluteTexAddress::octave_fixed
            <SHIFT,w_conf,h_conf,l_conf>
            <<<grid,block,0,stream>>>
            ( oct_obj.getDataTexPoint( ),
              oct_obj.getDataSurface( ),
              oct_obj.getWidth(),
              oct_obj.getHeight(),
              oct_obj.getDogSurface( ) );
    }
}

void Pyramid::make_octave( const Config& conf, ImageBase* base, Octave& oct_obj, cudaStream_t stream, bool isOctaveZero )
{
    if( _levels == 6 ) {
        if( conf.getGaussMode() == Config::Fixed9 ) {
            if( isOctaveZero )
                make_octave_sub<4,true,6> ( conf, base, oct_obj, stream );
            else
                make_octave_sub<4,false,6>( conf, base, oct_obj, stream );
        } else if( conf.getGaussMode() == Config::Fixed15 ) {
            if( isOctaveZero )
                make_octave_sub<7,true,6> ( conf, base, oct_obj, stream );
            else
                make_octave_sub<7,false,6>( conf, base, oct_obj, stream );
        } else {
            POP_FATAL("Unsupported Gauss filter mode for making all octaves at once");
        }
    } else {
        POP_FATAL("Unsupported number of levels for making all octaves at once");
    }
}

} // namespace popsift

