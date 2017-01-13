/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "sift_pyramid.h"
#include "sift_constants.h"
#include "gauss_filter.h"
#include "common/debug_macros.h"
#include "assist.h"
#include "common/clamp.h"

#include <iostream>
#include <stdio.h>

/* It makes no sense whatsoever to change this value */
#define PREV_LEVEL 3

namespace popsift {

namespace gauss {

namespace variableSpan {

namespace absoluteTexAddress {
__global__
void horiz( cudaTextureObject_t src_data,
            Plane2D_float       dst_data,
            int                 level )
{
    const int dst_w = dst_data.getWidth();

    const int off_x = blockIdx.x * blockDim.x + threadIdx.x;

    if( off_x >= dst_w ) return;

    float out = 0.0f;

    #pragma unroll
    for( int offset = d_gauss.inc_span[level]; offset>0; offset-- ) {
        const float& g  = popsift::d_gauss.inc_filter[level*GAUSS_ALIGN + offset];
        const float  v1 = tex2D<float>( src_data, off_x - offset + 0.5f, blockIdx.y + 0.5f );
        out += ( v1 * g );

        const float  v2 = tex2D<float>( src_data, off_x + offset + 0.5f, blockIdx.y + 0.5f );
        out += ( v2 * g );
    }
    const float& g  = popsift::d_gauss.inc_filter[level*GAUSS_ALIGN];
    const float v3 = tex2D<float>( src_data, off_x+0.5f, blockIdx.y+0.5f );
    out += ( v3 * g );

    dst_data.ptr(blockIdx.y)[off_x] = out;
}

__device__
inline void vert( cudaTextureObject_t src_data,
                  Plane2D_float       dst_data,
                  int                 span,
                  float*              filter )
{
    const int dst_w = dst_data.getWidth();
    const int dst_h = dst_data.getHeight();

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
        val = tex2D<float>( src_data, block_x + idx + 0.5f, block_y + idy + 0.5f );
        out += ( val * g );

        idy = threadIdx.y + offset;
        val = tex2D<float>( src_data, block_x + idx + 0.5f, block_y + idy + 0.5f );
        out += ( val * g );
    }

    g  = filter[0];
    idy = threadIdx.y;
    val = tex2D<float>( src_data, block_x + idx + 0.5f, block_y + idy + 0.5f );
    out += ( val * g );

    idx = block_x+threadIdx.x;
    idy = block_y+threadIdx.y;

    if( idx >= dst_w ) return;
    if( idy >= dst_h ) return;

    dst_data.ptr(idy)[idx] = out;
}

__global__
void vert( cudaTextureObject_t src_data,
           Plane2D_float       dst_data,
           int                 level )
{
    vert( src_data, dst_data, d_gauss.inc_span[level], &popsift::d_gauss.inc_filter[level*GAUSS_ALIGN] );
}

} // namespace absoluteTexAddress

namespace relativeTexAddress {

__device__
inline void horiz( cudaTextureObject_t src_data,
                   Plane2D_float       dst_data,
                   float               shift,
                   int                 span,
                   float*              filter )
{
    const float dst_w  = dst_data.getWidth();
    const float dst_h  = dst_data.getHeight();
    const float read_y = ( blockIdx.y + shift ) / dst_h;

    const int off_x = blockIdx.x * blockDim.x + threadIdx.x;

    if( off_x >= dst_w ) return;

    float out = 0.0f;

    #pragma unroll
    for( int offset = span; offset>0; offset-- ) {
        const float& g  = filter[offset];
        const float read_x_l = ( off_x - offset );
        const float  v1 = tex2D<float>( src_data, ( read_x_l + shift ) / dst_w, read_y );
        out += ( v1 * g );

        const float read_x_r = ( off_x + offset );
        const float  v2 = tex2D<float>( src_data, ( read_x_r + shift ) / dst_w, read_y );
        out += ( v2 * g );
    }
    const float& g  = filter[0];
    const float read_x = off_x;
    const float v3 = tex2D<float>( src_data, ( read_x + shift ) / dst_w, read_y );
    out += ( v3 * g );

    dst_data.ptr(blockIdx.y)[off_x] = out * 255.0f;
}

__global__
void horiz( cudaTextureObject_t src_data,
            Plane2D_float       dst_data,
            float               shift )
{
    horiz( src_data, dst_data, shift, d_gauss.inc_span[0], &d_gauss.inc_filter[0*GAUSS_ALIGN] );
}

} // namespace relativeTexAddress

} // namespace variableSpan

namespace fixedSpan {

template<int SPAN, bool isOctave0>
__global__
void octave( Plane2D_float src_data,
             Plane2D_float dst_data,
             cudaSurfaceObject_t dog_data )
{
    /* Idea: Process a 16x16 square of the src image and compute
     *       all levels except level 0 (must be computed before)
     *       blockIdx.x and blockIdx.y are the X and Y blocks.
     *       threadIdx.x runs 0 ... 16 + 2*(span-1)
     *       threadIdx.y runs 0 ... LEVELS-2
     */
    const int stride = 16 + 2 * (SPAN-1);
    const int noff   = SPAN - 1;
    const int level  = threadIdx.y + 1;

    __shared__ float inn_block[stride][stride];
    __shared__ float out_block[MAX_LEVELS][stride][stride];

    const int w = src_data.getWidth();
    const int h = src_data.getHeight();

    Plane2D_float destination( w, h,
                               dst_data.ptr( threadIdx.y * h ),
                               dst_data.getPitch() );

    for( int row=threadIdx.y; row<stride; row+=blockDim.y ) {
        const int idx = blockIdx.x * 16 + threadIdx.x - noff;
        const int idy = blockIdx.y * 16 + row         - noff;
        inn_block[row][threadIdx.x] = src_data.ptr(idy)[idx];
    }
    __syncthreads();

    float* filter = isOctave0
                  ? &d_gauss.abs_filter_o0[ level * GAUSS_ALIGN ]
                  : &d_gauss.abs_filter_oN[ level * GAUSS_ALIGN ];

    for( int row=0; row<stride; row++ ) {
        float in = inn_block[row][threadIdx.x];
        float out = in * filter[0];
        float g;
        if( SPAN==4 ) {
            g = __shfl_up( in, 1 ) + __shfl_down( in, 1, 22 ); out += g * filter[1];
            g = __shfl_up( in, 2 ) + __shfl_down( in, 2, 22 ); out += g * filter[2];
            g = __shfl_up( in, 3 ) + __shfl_down( in, 3, 22 ); out += g * filter[3];
        }

#if 0
        #pragma unroll
        for( int s=1; s<SPAN; s++ ) {
            float g = __shfl_up( in, s ) + __shfl_down( in, s );
            out += g * filter[s];
        }
#endif
        out_block[level][row][threadIdx.x] = out;
    }
    __syncthreads();

#if 0
    float out_col[stride];
    for( int col=0; col<stride; col++ ) {
        float in = out_block[level][threadIdx.x][col];
        float out = in * filter[0];

        #pragma unroll
        for( int s=1; s<SPAN; s++ ) {
            float g = __shfl_up( in, -s ) + __shfl_down( in, s );
            out += g * filter[s];
        }
        out_col[col] = out;
    }
    __syncthreads();

    for( int col=0; col<stride; col++ ) {
        out_block[level][threadIdx.x][col] = out_col[col];
    }
#endif

    for( int row=0; row<16; row++ ) {
        const int base_x = blockIdx.x * 16;
        const int base_y = blockIdx.y * 16;
        if( threadIdx.x < 16 ) {
            float val = out_block[level][row+noff][threadIdx.x+noff];
            float dog = ( level == 0
                        ? inn_block[row+noff][threadIdx.x+noff]
                        : out_block[level-1][row+noff][threadIdx.x+noff] )
                      - val;
            if( base_y+row < h && base_x+threadIdx.x < w ) {
                int idx = base_x+threadIdx.x;
                int idy = base_y+row;

                destination.ptr(idy)[idx] = out_block[threadIdx.y][row+noff][threadIdx.x+noff];
                // destination.ptr(idy)[idx] = inn_block[row+noff][threadIdx.x+noff];
                // destination.ptr(idy)[idx] = src_data.ptr(idy)[idx];
                // destination.ptr(idy)[idx] = 0;
                // destination.ptr(idy)[idx] = val;

                surf2DLayeredwrite( dog,
                                    dog_data,
                                    idx*4,
                                    idy,
                                    level,
                                    cudaBoundaryModeZero );
            }
        }
    }
    __syncthreads();
}

} // namespace fixedSpan



__global__
void get_by_2_interpolate( cudaTextureObject_t src_data,
                           Plane2D_float       dst_data,
                           int                 level )
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const int dst_w = dst_data.getWidth();
    const int dst_h = dst_data.getHeight();
    if( idx >= dst_w ) return;
    if( idy >= dst_h ) return;

    const float val = tex2D<float>( src_data, 2.0f * idx + 1.0f, 2.0f * idy + 1.0f );
    dst_data.ptr(idy)[idx] = val;
}

__global__
void get_by_2_pick_every_second( Plane2D_float src_data,
                                 Plane2D_float dst_data,
                                 int           level )
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const int dst_w = dst_data.getWidth();
    const int dst_h = dst_data.getHeight();
    if( idx >= dst_w ) return;
    if( idy >= dst_h ) return;

    const int src_w = src_data.getWidth();
    const int src_h = src_data.getHeight();
    const int read_x = clamp( idx << 1, 0, src_w );
    const int read_y = clamp( idy << 1, 0, src_h );

    const float val = src_data.ptr(read_y)[read_x];

    dst_data.ptr(idy)[idx] = val;
}


__global__
void make_dog( Plane2D_float       this_data,
               Plane2D_float       top_data,
               cudaSurfaceObject_t dog_data,
               int                 level )
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const int cols = this_data.getWidth();
    const int rows = this_data.getHeight();
    
    const int r_x = clamp( idx, cols );
    const int r_y = clamp( idy, rows );

    const float b = this_data.ptr(r_y)[r_x];
    const float a = top_data .ptr(r_y)[r_x];
    const float c = b - a;

    surf2DLayeredwrite( c, dog_data, idx*4, idy, level, cudaBoundaryModeZero );
}

} // namespace gauss

__host__
inline void Pyramid::horiz_from_input_image( const Config& conf, Image* base, int octave, cudaStream_t stream, Config::SiftMode mode )
{
    Octave&   oct_obj = _octaves[octave];

    const int width   = oct_obj.getWidth();
    const int height  = oct_obj.getHeight();

    dim3 block( 128, 1 );
    dim3 grid;
    grid.x  = grid_divide( width,  128 );
    grid.y  = height;

    float shift  = 0.5f;

    if( octave == 0 && ( mode == Config::PopSift || mode == Config::VLFeat ) ) {
        shift  = 0.5f * powf( 2.0f, conf.getUpscaleFactor() );
    }

    gauss::variableSpan::relativeTexAddress::horiz
        <<<grid,block,0,stream>>>
        ( base->getInputTexture(),
          oct_obj.getIntermediateData( ),
          shift );
}


__host__
inline void Pyramid::downscale_from_prev_octave( int octave, int level, cudaStream_t stream, Config::SiftMode mode )
{
    Octave&      oct_obj = _octaves[octave];
    Octave& prev_oct_obj = _octaves[octave-1];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    /* Necessary to wait for a lower level in the previous octave */
    cudaEvent_t ev = prev_oct_obj.getEventGaussDone( _levels-PREV_LEVEL );
    cudaStreamWaitEvent( stream, ev, 0 );

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
            ( prev_oct_obj.getData( _levels-PREV_LEVEL ),
              oct_obj.getData( level ),
              level );
        break;
    default :
        gauss::get_by_2_interpolate
            <<<h_grid,h_block,0,stream>>>
            ( prev_oct_obj._data_tex[ _levels-PREV_LEVEL ],
              oct_obj.getData( level ),
              level );
        break;
    }
}

__host__
inline void Pyramid::horiz_from_prev_level( int octave, int level, cudaStream_t stream )
{
    Octave&      oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    /* waiting for previous level in same octave */
    cudaEvent_t ev = oct_obj.getEventGaussDone( level-1 );
    cudaStreamWaitEvent( stream, ev, 0 );

    dim3 block( 128, 1 );
    dim3 grid;
    grid.x  = grid_divide( width,  128 );
    grid.y  = height;

    gauss::variableSpan::absoluteTexAddress::horiz
        <<<grid,block,0,stream>>>
        ( oct_obj._data_tex[ level-1 ],
          oct_obj.getIntermediateData( ),
          level );
}

__host__
inline void Pyramid::vert_from_interm( int octave, int level, cudaStream_t stream )
{
    Octave& oct_obj = _octaves[octave];

    /* waiting for any events is not necessary, it's in the same stream as horiz
     */

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    dim3 block( 64, 2 );
    dim3 grid;
    grid.x = (unsigned int)grid_divide( width,  block.x );
    grid.y = (unsigned int)grid_divide( height, block.y );

    gauss::variableSpan::absoluteTexAddress::vert
        <<<grid,block,0,stream>>>
        ( oct_obj._interm_data_tex,
          oct_obj.getData( level ),
          level );
}

__host__
inline void Pyramid::dog_from_blurred( int octave, int level, cudaStream_t stream )
{
    Octave&      oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    dim3 block( 128, 2 );
    dim3 grid;
    grid.x = grid_divide( width,  block.x );
    grid.y = grid_divide( height, block.y );

    /* waiting for lower level is automatic, it's in the same stream.
     * waiting for upper level is necessary, it's in another stream.
     */
    cudaEvent_t  ev     = oct_obj.getEventGaussDone( level-1 );
    cudaStreamWaitEvent( stream, ev, 0 );

    gauss::make_dog
        <<<grid,block,0,stream>>>
        ( oct_obj.getData(level),
          oct_obj.getData(level-1),
          oct_obj.getDogSurface( ),
          level-1 );
}

__host__
inline void Pyramid::make_octave( const Config& conf, Octave& oct_obj, cudaStream_t stream, bool isOctaveZero )
{
    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    if( conf.getGaussMode() == Config::Fixed4 ) {
        dim3 block( 22, _levels-1 );
        dim3 grid;
        grid.x = grid_divide( width,  16 );
        grid.y = grid_divide( height, 16 );

        if( isOctaveZero ) {
            gauss::fixedSpan::octave
                <4,true>
                <<<grid,block,0,stream>>>
                ( oct_obj.getData(0),
                  oct_obj.getData(1),
                  oct_obj.getDogSurface( ) );
        } else {
            gauss::fixedSpan::octave
                <4,false>
                <<<grid,block,0,stream>>>
                ( oct_obj.getData(0),
                  oct_obj.getData(1),
                  oct_obj.getDogSurface( ) );
        }
    } else {
        dim3 block( 30, _levels-1 );
        dim3 grid;
        grid.x = grid_divide( width,  16 );
        grid.y = grid_divide( height, 16 );

        if( isOctaveZero ) {
            gauss::fixedSpan::octave
                <8,true>
                <<<grid,block,0,stream>>>
                ( oct_obj.getData(0),
                  oct_obj.getData(1),
                  oct_obj.getDogSurface( ) );
        } else {
            gauss::fixedSpan::octave
                <8,false>
                <<<grid,block,0,stream>>>
                ( oct_obj.getData(0),
                  oct_obj.getData(1),
                  oct_obj.getDogSurface( ) );
        }
    }
}

/*************************************************************
 * V11: host side
 *************************************************************/
__host__
void Pyramid::build_pyramid( const Config& conf, Image* base )
{
    cudaError_t err;

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
      Octave& oct_obj   = _octaves[octave];

      if( conf.getGaussMode() == Config::Fixed4 || conf.getGaussMode() == Config::Fixed8 ) {
        cudaStream_t stream = oct_obj.getStream(0);
        if( octave == 0 ) {
            horiz_from_input_image( conf, base, 0, stream, conf.getSiftMode() );
            vert_from_interm( octave, 0, stream );
            make_octave( conf, oct_obj, stream, true );
        } else {
            downscale_from_prev_octave( octave, 0, stream, conf.getSiftMode() );
            make_octave( conf, oct_obj, stream, false );
        }

        /*
         * The multitude of events may be required by the following stages.
         */
        for( uint32_t level=0; level<_levels; level++ ) {
            cudaEvent_t  ev     = oct_obj.getEventGaussDone(level);
            cudaEvent_t  dog_ev = oct_obj.getEventDogDone(level);
            err = cudaEventRecord( ev, stream );
            POP_CUDA_FATAL_TEST( err, "Could not record a Gauss done event: " );
            err = cudaEventRecord( dog_ev, stream );
            POP_CUDA_FATAL_TEST( err, "Could not record a Gauss done event: " );
        }
      } else {

        for( uint32_t level=0; level<_levels; level++ ) {
            const int width  = oct_obj.getWidth();
            const int height = oct_obj.getHeight();

            cudaStream_t stream = oct_obj.getStream(level);
            cudaEvent_t  ev     = oct_obj.getEventGaussDone(level);
            cudaEvent_t  dog_ev = oct_obj.getEventDogDone(level);

            if( level == 0 )
            {
                if( octave == 0 )
                {
                    horiz_from_input_image( conf, base, 0, stream, conf.getSiftMode() );
                    vert_from_interm( octave, 0, stream );
                }
                else
                {
                    switch( _scaling_mode )
                    {
                    case Config::ScaleDirect :
                        // Does not work yet
                        horiz_from_input_image( conf, base, octave, stream, conf.getSiftMode() );
                        vert_from_interm( octave, level, stream );
                        break;
                    case Config::ScaleDefault :
                    default :
                        downscale_from_prev_octave( octave, level, stream, conf.getSiftMode() );
                        break;
                    }
                }
            }
            else
            {
                horiz_from_prev_level( octave, level, stream );
                vert_from_interm( octave, level, stream );
            }


            err = cudaEventRecord( ev, stream );
            POP_CUDA_FATAL_TEST( err, "Could not record a Gauss done event: " );

            if( level > 0 ) {
                dog_from_blurred( octave, level, stream );

                err = cudaEventRecord( dog_ev, stream );
                POP_CUDA_FATAL_TEST( err, "Could not record a Gauss done event: " );
            }
        }
      }
    }
}

} // namespace popsift

