#include "s_pyramid.h"

#include "gauss_filter.h"
#include "clamp.h"
#include "debug_macros.h"
#include "assist.h"

/*************************************************************
 * V7: device side
 *************************************************************/

#define V7_WIDTH    32
#define V7_RANGE    4 // RANGES from 1 to 12 are possible
#define V7_GAUSS_BASE   ( GAUSS_ONE_SIDE_RANGE - V7_RANGE )
#define V7_FILTERSIZE   ( V7_RANGE + 1        + V7_RANGE )
#define V7_READ_RANGE   ( V7_RANGE + V7_WIDTH + V7_RANGE )
#define V7_LEVELS       _levels

namespace popart {

__global__
void filter_gauss_horiz_v7( Plane2D_float src_data,
                            Plane2D_float dst_data )
{
    int block_x = blockIdx.x * V7_WIDTH;
    int block_y = blockIdx.y;
    int idx;

    float g;
    float val;
    float out = 0;

    const int width  = src_data.getWidth();
    const int height = src_data.getHeight();

    for( int offset = V7_RANGE; offset>0; offset-- ) {
        g  = popart::d_gauss_filter[GAUSS_ONE_SIDE_RANGE - offset];

        idx = clamp( block_x + threadIdx.x - offset, width );
        val = src_data.ptr(block_y)[idx];
        out += ( val * g );

        idx = clamp( block_x + threadIdx.x + offset, width );
        val = src_data.ptr(block_y)[idx];
        out += ( val * g );
    }

    g  = popart::d_gauss_filter[GAUSS_ONE_SIDE_RANGE];
    idx = clamp( block_x + threadIdx.x, width );
    val = src_data.ptr(block_y)[idx];
    out += ( val * g );

    if( block_y >= height ) return;
    if( idx     >= width  ) return;

    dst_data.ptr(block_y)[idx] = out;
}

__device__
void filter_gauss_vert_v7_sub( Plane2D_float&  src_data,
                               Plane2D_float&  dst_data )
{
    const int block_x = blockIdx.x * V7_WIDTH;
    const int block_y = blockIdx.y;
    const int idx     = block_x + threadIdx.x;
    int idy;

    const int width  = src_data.getWidth();
    const int height = src_data.getHeight();

    if( idx >= width ) return;

    float g;
    float val;
    float out = 0;

    for( int offset = V7_RANGE; offset>0; offset-- ) {
        g  = popart::d_gauss_filter[GAUSS_ONE_SIDE_RANGE - offset];

        idy = clamp( block_y - offset, height );
        val = src_data.ptr(idy)[idx];
        out += ( val * g );

        idy = clamp( block_y + offset, height );
        val = src_data.ptr(idy)[idx];
        out += ( val * g );
    }

    g  = popart::d_gauss_filter[GAUSS_ONE_SIDE_RANGE];
    idy = clamp( block_y, height );
    val = src_data.ptr(idy)[idx];
    out += ( val * g );

    if( idy >= height ) return;

    dst_data.ptr(idy)[idx] = out;
}

__global__
void filter_gauss_vert_v7( Plane2D_float   src_data,
                           Plane2D_float   dst_data )
{
    filter_gauss_vert_v7_sub( src_data, dst_data );
}

__global__
void filter_gauss_vert_v7_and_dog( Plane2D_float   src_data,
                                   Plane2D_float   dst_data,
                                   Plane2D_float   higher_level_data,
                                   Plane2D_float   dog_data )
{
    filter_gauss_vert_v7_sub( src_data, dst_data );

    const int idx = blockIdx.x * V7_WIDTH + threadIdx.x;
    const int idy = blockIdx.y;

    const int width  = src_data.getWidth();
    const int height = src_data.getHeight();

    if( idx >= width ) return;
    if( idy >= height ) return;

    float a, b;
    a = dst_data.ptr(idy)[idx];
    b = higher_level_data.ptr(idy)[idx];
    a = fabs( a - b );
    dog_data.ptr(idy)[idx] = a;
}

__global__
void filter_gauss_horiz_v7_by_2( Plane2D_float   src_data,
                                 Plane2D_float   dst_data )
{
    int block_x = blockIdx.x * V7_WIDTH;
    int block_y = blockIdx.y;

    const int src_w   = src_data.getWidth();
    const int src_h   = src_data.getHeight();
    int       src_idx;
    const int src_idy = clamp( 2 * block_y, src_h );
    const int dst_w   = dst_data.getWidth();
    const int dst_h   = dst_data.getHeight();
    const int dst_idx = block_x + threadIdx.x;
    const int dst_idy = block_y;

    if( dst_idx >= dst_w ) return;
    if( dst_idy >= dst_h ) return;

    float g;
    float val;
    float out = 0;

    for( int offset = V7_RANGE; offset>0; offset-- ) {
        g  = popart::d_gauss_filter[GAUSS_ONE_SIDE_RANGE - offset];

        src_idx = clamp( 2 * ( dst_idx - offset ), src_w );
        val = src_data.ptr(src_idy)[src_idx];
        out += ( val * g );

        src_idx = clamp( 2 * ( dst_idx + offset ), src_w );
        val = src_data.ptr(src_idy)[src_idx];
        out += ( val * g );
    }

    g  = popart::d_gauss_filter[GAUSS_ONE_SIDE_RANGE];
    src_idx = clamp( 2 * dst_idx, src_w );
    val = src_data.ptr(src_idy)[src_idx];
    out += ( val * g );

    dst_data.ptr(dst_idy)[dst_idx] = out;
}

/*************************************************************
 * V7: host side
 *************************************************************/
__host__
void Pyramid::build_v7( Image* base )
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

    dim3 block;
    block.x = V7_WIDTH;

    for( int octave=0; octave<_num_octaves; octave++ ) {

        for( int level=0; level<V7_LEVELS; level++ ) {
            dim3 grid;
            grid.x  = grid_divide(_octaves[octave].getData(level).getWidth(), V7_WIDTH);
            grid.y  = _octaves[octave].getData(level).getHeight();
#if 0
        cerr << "Configuration for octave " << octave << endl
             << "  Horiz: layer size: "
             << _octaves[octave].getData(level).getWidth() << "x" << _octaves[octave].getData(level).getHeight() << endl
             << "  Vert: layer size: "
             << _octaves[octave].getData2(level).getWidth() << "x" << _octaves[octave].getData2(level).getHeight() << endl
             << "  grid: "
             << "(" << grid.x << "," << grid.y << "," << grid.z << ")"
             << " block: "
             << "(" << block.x << "," << block.y << "," << block.z << ")" << endl;
#endif


            if( level == 0 ) {
                if( octave == 0 ) {
                    filter_gauss_horiz_v7
                        <<<grid,block>>>
                        ( base->array,
                          _octaves[octave].getIntermediateData( ) );
                } else {
                    filter_gauss_horiz_v7_by_2
                        <<<grid,block>>>
                        ( _octaves[octave-1].getData( V7_LEVELS-3 ),
                          _octaves[octave].getIntermediateData() );
                }
            } else {
                filter_gauss_horiz_v7
                    <<<grid,block>>>
                    ( _octaves[octave].getData( level-1 ),
                      _octaves[octave].getIntermediateData() );
            }
            cudaDeviceSynchronize( );
            cudaError_t err = cudaGetLastError();
            POP_CUDA_FATAL_TEST( err, "filter_gauss_horiz_v7 failed: " );

            if( level == 0 ) {
                filter_gauss_vert_v7
                    <<<grid,block>>>
                    ( _octaves[octave].getIntermediateData(),
                      _octaves[octave].getData( level ) );
            } else {
                filter_gauss_vert_v7_and_dog
                    <<<grid,block>>>
                    ( _octaves[octave].getIntermediateData(),
                      _octaves[octave].getData( level ),
                      _octaves[octave].getData( level-1 ),
                      _octaves[octave].getDogData( level-1 ) );
            }
            cudaDeviceSynchronize( );
            err = cudaGetLastError();
            POP_CUDA_FATAL_TEST( err, "filter_gauss_horiz_v7 failed: " );
        }
    }
}

} // namespace popart

