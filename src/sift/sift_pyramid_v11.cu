#include "sift_pyramid.h"
#include "sift_constants.h"
#include "gauss_filter.h"
#include "debug_macros.h"
#include "assist.h"
#include "clamp.h"

#include <iostream>
#include <stdio.h>

#define PREV_LEVEL 3

/*************************************************************
 * V11: device side
 *************************************************************/

namespace popart {

namespace gauss {
namespace v11 {

namespace absoluteLinearTex {
__global__
void horiz_128x1( cudaTextureObject_t src_data,
                  Plane2D_float       dst_data,
                  int                 level )
{
    const int dst_w = dst_data.getWidth();

    const int off_x = blockIdx.x * blockDim.x + threadIdx.x;

    if( off_x >= dst_w ) return;

    float out = 0.0f;

    #pragma unroll
    for( int offset = d_gauss.span[level]; offset>0; offset-- ) {
        const float& g  = popart::d_gauss.filter[level*GAUSS_ALIGN + offset];
        const float  v1 = tex2D<float>( src_data, off_x - offset + 0.5f, blockIdx.y + 0.5f );
        out += ( v1 * g );

        const float  v2 = tex2D<float>( src_data, off_x + offset + 0.5f, blockIdx.y + 0.5f );
        out += ( v2 * g );
    }
    const float& g  = popart::d_gauss.filter[level*GAUSS_ALIGN];
    const float v3 = tex2D<float>( src_data, off_x+0.5f, blockIdx.y+0.5f );
    out += ( v3 * g );

    dst_data.ptr(blockIdx.y)[off_x] = out;
}
} // namespace absoluteLinearTex

__global__
void horiz_tex_128x1( cudaTextureObject_t src_data,
                      Plane2D_float       dst_data,
                      float               shift )
{
    const float dst_w  = dst_data.getWidth();
    const float dst_h  = dst_data.getHeight();
    const float read_y = ( blockIdx.y + shift ) / dst_h;

    const int off_x = blockIdx.x * blockDim.x + threadIdx.x;

    if( off_x >= dst_w ) return;

    float out = 0.0f;

    #pragma unroll
    for( int offset = d_gauss.span[0]; offset>0; offset-- ) {
        const float& g  = popart::d_gauss.filter[0*GAUSS_ALIGN + offset];
        const float read_x_l = ( off_x - offset );
        const float  v1 = tex2D<float>( src_data, ( read_x_l + shift ) / dst_w, read_y );
        out += ( v1 * g );

        const float read_x_r = ( off_x + offset );
        const float  v2 = tex2D<float>( src_data, ( read_x_r + shift ) / dst_w, read_y );
        out += ( v2 * g );
    }
    const float& g  = popart::d_gauss.filter[0*GAUSS_ALIGN];
    const float read_x = off_x;
    const float v3 = tex2D<float>( src_data, ( read_x + shift ) / dst_w, read_y );
    out += ( v3 * g );

    dst_data.ptr(blockIdx.y)[off_x] = out * 255.0f; // scalbnf( out, 8 );
    // dst_data.ptr(blockIdx.y)[off_x] = scalbnf( out, 8 );
}

__global__
void horiz_tex_128x1_initial_blur( cudaTextureObject_t src_data,
                                   Plane2D_float       dst_data,
                                   float               shift )
{
    const float dst_w  = dst_data.getWidth();
    const float dst_h  = dst_data.getHeight();
    const float read_y = ( blockIdx.y + shift ) / dst_h;

    const int off_x = blockIdx.x * blockDim.x + threadIdx.x;

    if( off_x >= dst_w ) return;

    float out = 0.0f;

    #pragma unroll
    for( int offset = d_gauss.initial_span; offset>0; offset-- ) {
        const float& g  = popart::d_gauss.filter_initial_blur[offset];
        const float read_x_l = ( off_x - offset );
        const float  v1 = tex2D<float>( src_data, ( read_x_l + shift ) / dst_w, read_y );
        out += ( v1 * g );

        const float read_x_r = ( off_x + offset );
        const float  v2 = tex2D<float>( src_data, ( read_x_r + shift ) / dst_w, read_y );
        out += ( v2 * g );
    }
    const float& g  = popart::d_gauss.filter_initial_blur[0];
    const float read_x = off_x;
    const float v3 = tex2D<float>( src_data, ( read_x + shift ) / dst_w, read_y );
    out += ( v3 * g );

    dst_data.ptr(blockIdx.y)[off_x] = out * 255.0f; // scalbnf( out, 8 );
    // dst_data.ptr(blockIdx.y)[off_x] = scalbnf( out, 8 );
}


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
void vert( cudaTextureObject_t src_data,
           Plane2D_float       dst_data,
           int level )
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

#ifdef GAUSS_INTERM_FILTER_MODE_POINT
    for( int offset = d_gauss.span[level]; offset>0; offset-- ) {
        g  = popart::d_gauss.filter[level*GAUSS_ALIGN + offset];

        idy = threadIdx.y - offset;
        val = tex2D<float>( src_data, block_x + idx, block_y + idy );
        out += ( val * g );

        idy = threadIdx.y + offset;
        val = tex2D<float>( src_data, block_x + idx, block_y + idy );
        out += ( val * g );
    }

    g  = popart::d_gauss.filter[level*GAUSS_ALIGN];
    idy = threadIdx.y;
    val = tex2D<float>( src_data, block_x + idx, block_y + idy );
    out += ( val * g );
#else // not GAUSS_INTERM_FILTER_MODE_POINT
    for( int offset = d_gauss.span[level]; offset>0; offset-- ) {
        g  = popart::d_gauss.filter[level*GAUSS_ALIGN + offset];

        idy = threadIdx.y - offset;
        val = tex2D<float>( src_data, block_x + idx + 0.5f, block_y + idy + 0.5f );
        out += ( val * g );

        idy = threadIdx.y + offset;
        val = tex2D<float>( src_data, block_x + idx + 0.5f, block_y + idy + 0.5f );
        out += ( val * g );
    }

    g  = popart::d_gauss.filter[level*GAUSS_ALIGN];
    idy = threadIdx.y;
    val = tex2D<float>( src_data, block_x + idx + 0.5f, block_y + idy + 0.5f );
    out += ( val * g );
#endif // not GAUSS_INTERM_FILTER_MODE_POINT

    idx = block_x+threadIdx.x;
    idy = block_y+threadIdx.y;

    if( idx >= dst_w ) return;
    if( idy >= dst_h ) return;

    dst_data.ptr(idy)[idx] = out;
}

__global__
void vert_initial_blur( cudaTextureObject_t src_data,
                        Plane2D_float       dst_data )
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

#ifdef GAUSS_INTERM_FILTER_MODE_POINT
    for( int offset = d_gauss.initial_span; offset>0; offset-- ) {
        g  = popart::d_gauss.filter_initial_blur[offset];

        idy = threadIdx.y - offset;
        val = tex2D<float>( src_data, block_x + idx, block_y + idy );
        out += ( val * g );

        idy = threadIdx.y + offset;
        val = tex2D<float>( src_data, block_x + idx, block_y + idy );
        out += ( val * g );
    }

    g  = popart::d_gauss.filter_initial_blur[0];
    idy = threadIdx.y;
    val = tex2D<float>( src_data, block_x + idx, block_y + idy );
    out += ( val * g );
#else // not GAUSS_INTERM_FILTER_MODE_POINT
    for( int offset = d_gauss.initial_span; offset>0; offset-- ) {
        g  = popart::d_gauss.filter_initial_blur[offset];

        idy = threadIdx.y - offset;
        val = tex2D<float>( src_data, block_x + idx + 0.5f, block_y + idy + 0.5f );
        out += ( val * g );

        idy = threadIdx.y + offset;
        val = tex2D<float>( src_data, block_x + idx + 0.5f, block_y + idy + 0.5f );
        out += ( val * g );
    }

    g  = popart::d_gauss.filter_initial_blur[0];
    idy = threadIdx.y;
    val = tex2D<float>( src_data, block_x + idx + 0.5f, block_y + idy + 0.5f );
    out += ( val * g );
#endif // not GAUSS_INTERM_FILTER_MODE_POINT

    idx = block_x+threadIdx.x;
    idy = block_y+threadIdx.y;

    if( idx >= dst_w ) return;
    if( idy >= dst_h ) return;

    dst_data.ptr(idy)[idx] = out;
}


#if 1
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
    const float c = b - a; // c = fabs( a - b );

    surf2DLayeredwrite( c, dog_data, idx*4, idy, level, cudaBoundaryModeZero );
}
#else
__global__
void make_dog( cudaTextureObject_t this_data,
               cudaTextureObject_t top_data,
               cudaSurfaceObject_t dog_data,
               int                 level )
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const float b = tex2D<float>( this_data, idx, idy );
    const float a = tex2D<float>( top_data, idx, idy );
    const float c = b - a; // c = fabs( a - b );

    surf2DLayeredwrite( c, dog_data, idx*4, idy, level, cudaBoundaryModeZero );
}
#endif

} // namespace gauss
} // namespace v11

__host__
inline void Pyramid::horiz_from_input_image( const Config& conf, Image* base, int octave, cudaStream_t stream, Config::SiftMode mode, bool initial_blur )
{
    Octave&      oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();

    /* I believe that waiting is not necessary because image is upscaled
     * in default stream */

    dim3 block( 128, 1 );
    dim3 grid;
    grid.x  = grid_divide( width,  128 );
    grid.y  = height;

    float shift = 0.5f;

    if( octave == 0 ) {
        switch( mode )
        {
        case Config::PopSift :
        case Config::VLFeat :
            shift = 0.5f * powf( 2.0f, conf.getUpscaleFactor() );
            break;
        case Config::OpenCV :
        default :
            break;
        }
    } else {
        // This code is only called for direct downscaling from the original image
        shift = 0.5f;
    }

    if( initial_blur )
        gauss::v11::horiz_tex_128x1_initial_blur
            <<<grid,block,0,stream>>>
            ( base->getInputTexture(),
              oct_obj.getIntermediateData( ),
              shift );
    else
        gauss::v11::horiz_tex_128x1
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
        gauss::v11::get_by_2_pick_every_second
            <<<h_grid,h_block,0,stream>>>
            ( prev_oct_obj.getData( _levels-PREV_LEVEL ),
              oct_obj.getData( level ),
              level );
        break;
    default :
        gauss::v11::get_by_2_interpolate
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

    gauss::v11::absoluteLinearTex::horiz_128x1
        <<<grid,block,0,stream>>>
        ( oct_obj._data_tex[ level-1 ],
          oct_obj.getIntermediateData( ),
          level );
}

__host__
inline void Pyramid::vert_from_interm( int octave, int level, cudaStream_t stream, bool initial_blur )
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

    if( initial_blur )
        gauss::v11::vert_initial_blur
            <<<grid,block,0,stream>>>
            ( oct_obj._interm_data_tex,
              oct_obj.getData( 0 ) );
    else
        gauss::v11::vert
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

#if 1
    gauss::v11::make_dog
        <<<grid,block,0,stream>>>
        ( oct_obj.getData(level),
          oct_obj.getData(level-1),
          oct_obj.getDogSurface( ),
          level-1 );
#else
    gauss::v11::make_dog
        <<<grid,block,0,stream>>>
        ( oct_obj._data_tex[level],
          oct_obj._data_tex[level-1],
          oct_obj.getDogSurface( ),
          level-1 );
#endif
}

/*************************************************************
 * V11: host side
 *************************************************************/
__host__
void Pyramid::build_v11( const Config& conf, Image* base )
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
                    horiz_from_input_image( conf, base, 0, stream, conf.getSiftMode(), _assume_initial_blur );
                    vert_from_interm( octave, level, stream, _assume_initial_blur );
                }
                else 
                {
                    switch( _scaling_mode )
                    {
                    case Config::DirectDownscaling :
                        {
                            horiz_from_input_image( conf, base, octave, stream, conf.getSiftMode(), false );
                            vert_from_interm( octave, level, stream, false );
                        }
                        break;
                    case Config::IndirectUnfilteredDownscaling :
                        downscale_from_prev_octave( octave, level, stream, conf.getSiftMode() );
                        break;
                    default :
                        cerr << __FILE__ << ":" << __LINE__ << ": unknown scaling mode" << endl;
                        break;
                    }
                }
            }
            else
            {
                horiz_from_prev_level( octave, level, stream );
                vert_from_interm( octave, level, stream, false );
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

} // namespace popart

