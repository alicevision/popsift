#include "sift_pyramid.h"
#include "gauss_filter.h"
#include "assist.h"

#include <iostream>

/*************************************************************
 * V11: device side
 *************************************************************/

namespace popart {
namespace gauss {
namespace v11 {

__global__
void horiz_tex_128x1( cudaTextureObject_t src_data,
                      Plane2D_float       dst_data,
                      int                 level )
{
    const float dst_w  = dst_data.getWidth();
    const float dst_h  = dst_data.getHeight();
    const float read_y = ( blockIdx.y + 0.5f ) / dst_h;

    const int off_x = blockIdx.x * blockDim.x + threadIdx.x;

    if( off_x >= dst_w ) return;

    float out = 0.0f;

    #pragma unroll
    for( int offset = GAUSS_SPAN; offset>0; offset-- ) {
        const float& g  = popart::d_gauss_filter[level*GAUSS_ALIGN + offset];
        const float read_x_l = ( off_x - offset );
        const float  v1 = tex2D<float>( src_data, ( read_x_l + 0.5f ) / dst_w, read_y );
        out += ( v1 * g );

        const float read_x_r = ( off_x + offset );
        const float  v2 = tex2D<float>( src_data, ( read_x_r + 0.5f ) / dst_w, read_y );
        out += ( v2 * g );
    }
    const float& g  = popart::d_gauss_filter[level*GAUSS_ALIGN];
    const float read_x = off_x;
    const float v3 = tex2D<float>( src_data, ( read_x + 0.5f ) / dst_w, read_y );
    out += ( v3 * g );

    dst_data.ptr(blockIdx.y)[off_x] = out;
}


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
    for( int offset = GAUSS_SPAN; offset>0; offset-- ) {
        const float& g  = popart::d_gauss_filter[level*GAUSS_ALIGN + offset];
        const float  v1 = tex2D<float>( src_data, off_x - offset + 0.5f, blockIdx.y + 0.5f );
        out += ( v1 * g );

        const float  v2 = tex2D<float>( src_data, off_x + offset + 0.5f, blockIdx.y + 0.5f );
        out += ( v2 * g );
    }
    const float& g  = popart::d_gauss_filter[level*GAUSS_ALIGN];
    const float v3 = tex2D<float>( src_data, off_x+0.5f, blockIdx.y+0.5f );
    out += ( v3 * g );

    dst_data.ptr(blockIdx.y)[off_x] = out;
}

__global__
void get_by_2( cudaTextureObject_t src_data,
               Plane2D_float       dst_data,
               int level )
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
void horiz_by_2( cudaTextureObject_t src_data,
                 Plane2D_float       dst_data,
                 int level )
{
    int block_x = blockIdx.x * blockDim.x;
    int block_y = blockIdx.y * blockDim.y;
    int idx;
    int idy     = threadIdx.y;

    float g;
    float val;
    float out = 0;

    for( int offset = GAUSS_SPAN; offset>0; offset-- ) {
        g  = popart::d_gauss_filter[level*GAUSS_ALIGN + offset];

        idx = threadIdx.x - offset;
        // add +1.0f because we must shift by 0.5 pixels upscaled by 2 in the previous octave
        val = tex2D<float>( src_data, 2 * ( block_x + idx ) + 1.0f, 2 * ( block_y + idy ) + 1.0f );
        out += ( val * g );

        idx = threadIdx.x + offset;
        val = tex2D<float>( src_data, 2 * ( block_x + idx ) + 1.0f, 2 * ( block_y + idy ) + 1.0f );
        out += ( val * g );
    }

    g  = popart::d_gauss_filter[level*GAUSS_ALIGN];
    idx = threadIdx.x;
    val = tex2D<float>( src_data, 2 * ( block_x + idx ) + 1.0f, 2 * ( block_y + idy ) + 1.0f );
    out += ( val * g );

    idx = block_x+threadIdx.x;
    idy = block_y+threadIdx.y;
    const int dst_w = dst_data.getWidth();
    const int dst_h = dst_data.getHeight();
    if( idx >= dst_w ) return;
    if( idy >= dst_h ) return;

    dst_data.ptr(idy)[idx] = out;
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
    for( int offset = GAUSS_SPAN; offset>0; offset-- ) {
        g  = popart::d_gauss_filter[level*GAUSS_ALIGN + offset];

        idy = threadIdx.y - offset;
        val = tex2D<float>( src_data, block_x + idx, block_y + idy );
        out += ( val * g );

        idy = threadIdx.y + offset;
        val = tex2D<float>( src_data, block_x + idx, block_y + idy );
        out += ( val * g );
    }

    g  = popart::d_gauss_filter[level*GAUSS_ALIGN];
    idy = threadIdx.y;
    val = tex2D<float>( src_data, block_x + idx, block_y + idy );
    out += ( val * g );
#else // not GAUSS_INTERM_FILTER_MODE_POINT
    for( int offset = GAUSS_SPAN; offset>0; offset-- ) {
        g  = popart::d_gauss_filter[level*GAUSS_ALIGN + offset];

        idy = threadIdx.y - offset;
        val = tex2D<float>( src_data, block_x + idx + 0.5f, block_y + idy + 0.5f );
        out += ( val * g );

        idy = threadIdx.y + offset;
        val = tex2D<float>( src_data, block_x + idx + 0.5f, block_y + idy + 0.5f );
        out += ( val * g );
    }

    g  = popart::d_gauss_filter[level*GAUSS_ALIGN];
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
void make_dog( cudaTextureObject_t this_data,
               cudaTextureObject_t top_data,
               cudaSurfaceObject_t dog_data,
               int                 level )
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const float b = tex2D<float>( this_data, idx, idy );
    const float a = tex2D<float>( top_data, idx, idy );
    const float c = a - b; // c = fabs( a - b );

    surf2DLayeredwrite( c, dog_data, idx*4, idy, level, cudaBoundaryModeZero );
}

} // namespace gauss
} // namespace v11

__host__
inline void Pyramid::horiz_from_upscaled_orig_tex( cudaTextureObject_t src_data,
                                            int                 octave )
{
    Octave&      oct_obj   = _octaves[octave];
    cudaStream_t oct_str_0 = oct_obj.getStream(0);

    const int width  = _octaves[octave].getWidth();
    const int height = _octaves[octave].getHeight();

    dim3 block( 128, 1 );
    dim3 grid;
    grid.x  = grid_divide( width,  128 );
    grid.y  = height;
    gauss::v11::horiz_tex_128x1
        <<<grid,block,0,oct_str_0>>>
        ( src_data,
          oct_obj.getIntermediateData( ),
          0 ); // level is always 0
}

#define PREV_LEVEL 3
// #define PREV_LEVEL 5

__host__
inline void Pyramid::downscale_from_prev_octave( int octave, int level )
{
    Octave&      oct_obj   = _octaves[octave];
    cudaStream_t oct_str_0 = oct_obj.getStream(0);

    Octave& prev_oct_obj  = _octaves[octave-1];
    cudaStreamWaitEvent( oct_str_0, prev_oct_obj.getEventGaussDone( _levels-PREV_LEVEL ), 0 );

    const int width  = _octaves[octave].getWidth();
    const int height = _octaves[octave].getHeight();

    dim3 h_block( 64, 2 );
    dim3 h_grid;
    h_grid.x = (unsigned int)grid_divide( width,  h_block.x );
    h_grid.y = (unsigned int)grid_divide( height, h_block.y );

    gauss::v11::get_by_2
        <<<h_grid,h_block,0,oct_str_0>>>
        ( prev_oct_obj._data_tex[ _levels-PREV_LEVEL ],
          oct_obj.getData( level ),
          level );
}

__host__
inline void Pyramid::downscale_from_prev_octave_and_horiz_blur( int octave, int level )
{
    Octave&      oct_obj   = _octaves[octave];
    cudaStream_t oct_str_0 = oct_obj.getStream(0);

    Octave& prev_oct_obj  = _octaves[octave-1];
    cudaStreamWaitEvent( oct_str_0, prev_oct_obj.getEventGaussDone( _levels-PREV_LEVEL ), 0 );

    const int width  = _octaves[octave].getWidth();
    const int height = _octaves[octave].getHeight();

    dim3 h_block( 64, 2 );
    dim3 h_grid;
    h_grid.x = (unsigned int)grid_divide( width,  h_block.x );
    h_grid.y = (unsigned int)grid_divide( height, h_block.y );

    gauss::v11::horiz_by_2
        <<<h_grid,h_block,0,oct_str_0>>>
        ( prev_oct_obj._data_tex[ _levels-PREV_LEVEL ],
          oct_obj.getIntermediateData( ),
          level );
}

__host__
inline void Pyramid::horiz_from_prev_level( int octave, int level )
{
    Octave&      oct_obj   = _octaves[octave];
    cudaStream_t oct_str_0 = oct_obj.getStream(0);

    const int width  = _octaves[octave].getWidth();
    const int height = _octaves[octave].getHeight();

    dim3 block( 128, 1 );
    dim3 grid;
    grid.x  = grid_divide( width,  128 );
    grid.y  = height;
    gauss::v11::horiz_128x1
        <<<grid,block,0,oct_str_0>>>
        ( oct_obj._data_tex[ level-1 ],
          oct_obj.getIntermediateData( ),
          level );
}

__host__
inline void Pyramid::vert_from_interm( int octave, int level )
{
    Octave&      oct_obj   = _octaves[octave];
    cudaStream_t oct_str_0 = oct_obj.getStream(0);

    const int width  = _octaves[octave].getWidth();
    const int height = _octaves[octave].getHeight();

    dim3 v_block( 64, 2 );
    dim3 v_grid;
    v_grid.x = (unsigned int)grid_divide( width,  v_block.x );
    v_grid.y = (unsigned int)grid_divide( height, v_block.y );

    gauss::v11::vert
        <<<v_grid,v_block,0,oct_str_0>>>
        ( oct_obj._interm_data_tex,
          oct_obj.getData( level ),
          level );
}

__host__
inline void Pyramid::dog_from_blurred( int octave, int level )
{
    Octave&      oct_obj   = _octaves[octave];
    cudaStream_t oct_str_0 = oct_obj.getStream(0);

    const int width  = _octaves[octave].getWidth();
    const int height = _octaves[octave].getHeight();

    dim3 block( 128, 2 );
    dim3 grid;
    grid.x = grid_divide( width,  block.x );
    grid.y = grid_divide( height, block.y );

    gauss::v11::make_dog
        <<<grid,block,0,oct_str_0>>>
        ( oct_obj._data_tex[level],
          oct_obj._data_tex[level-1],
          oct_obj.getDogSurface( ),
          level-1 );
}

/*************************************************************
 * V11: host side
 *************************************************************/
__host__
void Pyramid::build_v11( Image* base )
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

    for( uint32_t octave=0; octave<_num_octaves; octave++ ) {
        for( uint32_t level=0; level<_levels; level++ ) {

            const int width  = _octaves[octave].getWidth();
            const int height = _octaves[octave].getHeight();

            Octave&      oct_obj   = _octaves[octave];
            cudaStream_t oct_str_0 = oct_obj.getStream(0);

            if( level == 0 )
            {
                if( octave == 0 )
                {
                    cudaTextureObject_t& tex = base->getUpscaledTexture();
                    horiz_from_upscaled_orig_tex( tex, octave );
                    vert_from_interm( octave, level );
                }
                else 
                {
                    switch( _scaling_mode )
                    {
                    case Config::DirectDownscaling :
                        {
                            cudaTextureObject_t& tex = base->getUpscaledTexture();
                            horiz_from_upscaled_orig_tex( tex, octave );
                            vert_from_interm( octave, level );
                        }
                        break;
                    case Config::IndirectUnfilteredDownscaling :
                        downscale_from_prev_octave( octave, level );
                        break;
                    case Config::IndirectDownscaling :
                        downscale_from_prev_octave_and_horiz_blur( octave, level );
                        vert_from_interm( octave, level );
                        break;
                    default :
                        cerr << __FILE__ << ":" << __LINE__ << ": unknown scaling mode" << endl;
                        break;
                    }
                }
            }
            else
            {
                horiz_from_prev_level( octave, level );
                vert_from_interm( octave, level );
            }

        }
    }
    cudaDeviceSynchronize();
    for( uint32_t octave=0; octave<_num_octaves; octave++ ) {
        for( uint32_t level=1; level<_levels; level++ ) {
            dog_from_blurred( octave, level );
        }
    }
    cudaDeviceSynchronize();
    for( uint32_t octave=0; octave<_num_octaves; octave++ ) {
        for( uint32_t level=0; level<_levels; level++ ) {
            Octave&      oct_obj   = _octaves[octave];
            cudaStream_t oct_str_0 = oct_obj.getStream(0);

            cudaEventRecord( oct_obj.getEventGaussDone( level ), oct_str_0 );
        }
    }
}

} // namespace popart

