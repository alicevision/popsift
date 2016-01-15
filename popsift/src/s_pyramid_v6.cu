#include "s_pyramid.h"

#include "assist.h"
#include "gauss_filter.h"
#include "clamp.h"
#include "debug_macros.h"

/*************************************************************
 * V6: device side
 *************************************************************/

#define V6_WIDTH    128
#define V6_RANGE    4 // RANGES from 1 to 12 are possible
#define V6_GAUSS_BASE   ( GAUSS_ONE_SIDE_RANGE - V6_RANGE )
#define V6_FILTERSIZE   ( V6_RANGE + 1        + V6_RANGE )
#define V6_READ_RANGE   ( V6_RANGE + V6_WIDTH + V6_RANGE )
#define V6_LEVELS       _levels

namespace popart {

__global__
void filter_gauss_horiz_v6( Plane2D_float src_data,
                            Plane2D_float dst_data )
{
    int32_t block_x = blockIdx.x * V6_WIDTH; // blockDim.x; <- wrong, it's 136
    int32_t block_y = blockIdx.y;            // blockDim.y; <- it's 1, trivial

    __shared__ float px[V6_READ_RANGE];

    const int src_w = src_data.getWidth();
    const int src_h = src_data.getHeight();
    const int dst_w = dst_data.getWidth();
    const int dst_h = dst_data.getHeight();

    int32_t idx     = threadIdx.x - V6_RANGE;
    int32_t src_idx = clamp( block_x + idx, src_w );
    px[threadIdx.x] = src_data.ptr(block_y)[src_idx];
    __syncthreads();

    if( threadIdx.x >= V6_WIDTH ) return;

    float out = 0;
    #pragma unroll
    for( int i=0; i<V6_FILTERSIZE; i++ ) {
        out += px[threadIdx.x+i] * popart::d_gauss_filter[V6_GAUSS_BASE+i];
    }

    int dst_row   = block_x + threadIdx.x;
    int dst_col   = block_y;
    if( dst_col < 0 || dst_col >= dst_w ) return;
    if( dst_row < 0 || dst_row >= dst_h ) return;

    dst_data.ptr(dst_row)[dst_col] = out;
}

#ifdef USE_DOG_ARRAY
__global__
void filter_gauss_horiz_v6_and_dog( Plane2D_float src_data,
                                    Plane2D_float dst_data,
                                    Plane2D_float higher_level_data,
                                    cudaSurfaceObject_t dog_data,
                                    int                 level )
{
    int32_t block_x = blockIdx.x * V6_WIDTH;
    int32_t block_y = blockIdx.y;

    __shared__ float px[V6_READ_RANGE];

    const int src_w = src_data.getWidth();
    const int src_h = src_data.getHeight();
    const int dst_w = dst_data.getWidth();
    const int dst_h = dst_data.getHeight();

    int32_t idx     = threadIdx.x - V6_RANGE;
    int32_t src_idx = clamp( block_x + idx, src_w );
    px[threadIdx.x] = src_data.ptr(block_y)[src_idx];
    __syncthreads();

    if( threadIdx.x >= V6_WIDTH ) return;

    float out = 0;
    #pragma unroll
    for( int i=0; i<V6_FILTERSIZE; i++ ) {
        out += px[threadIdx.x+i] * popart::d_gauss_filter[V6_GAUSS_BASE+i];
    }

    int dst_row   = block_x + threadIdx.x;
    int dst_col   = block_y;
    if( dst_col < 0 || dst_col >= dst_w ) return;
    if( dst_row < 0 || dst_row >= dst_h ) return;

    dst_data.ptr(dst_row)[dst_col] = out;

    float cmp;
    cmp = higher_level_data.ptr(dst_row)[dst_col];
    out -= cmp;
    out = fabs(out);

    surf2DLayeredwrite( out, dog_data,
                        dst_col*4, dst_row, level,
                        cudaBoundaryModeZero );
}
#else // not USE_DOG_ARRAY
__global__
void filter_gauss_horiz_v6_and_dog( Plane2D_float src_data,
                                    Plane2D_float dst_data,
                                    Plane2D_float higher_level_data,
                                    Plane2D_float dog_data )
{
    int32_t block_x = blockIdx.x * V6_WIDTH;
    int32_t block_y = blockIdx.y;

    __shared__ float px[V6_READ_RANGE];

    const int src_w = src_data.getWidth();
    const int src_h = src_data.getHeight();
    const int dst_w = dst_data.getWidth();
    const int dst_h = dst_data.getHeight();

    int32_t idx     = threadIdx.x - V6_RANGE;
    int32_t src_idx = clamp( block_x + idx, src_w );
    px[threadIdx.x] = src_data.ptr(block_y)[src_idx];
    __syncthreads();

    if( threadIdx.x >= V6_WIDTH ) return;

    float out = 0;
    #pragma unroll
    for( int i=0; i<V6_FILTERSIZE; i++ ) {
        out += px[threadIdx.x+i] * popart::d_gauss_filter[V6_GAUSS_BASE+i];
    }

    int dst_row   = block_x + threadIdx.x;
    int dst_col   = block_y;
    if( dst_col < 0 || dst_col >= dst_w ) return;
    if( dst_row < 0 || dst_row >= dst_h ) return;

    dst_data.ptr(dst_row)[dst_col] = out;

    float cmp;
    cmp = higher_level_data.ptr(dst_row)[dst_col];
    out -= cmp;
    out = fabs(out);
    dog_data.ptr(dst_row)[dst_col] = out;
}
#endif // not USE_DOG_ARRAY

__global__
void filter_gauss_horiz_v6_by_2( Plane2D_float src_data,
                                 Plane2D_float dst_data )
{
    if( threadIdx.x >= V6_READ_RANGE ) return;
    int32_t block_x = blockIdx.x * V6_WIDTH;
    int32_t block_y = blockIdx.y;

    __shared__ float px[V6_READ_RANGE];

    const int src_w = src_data.getWidth();
    const int src_h = src_data.getHeight();
    const int dst_w = dst_data.getWidth();
    const int dst_h = dst_data.getHeight();

    int32_t idx     = threadIdx.x - V6_RANGE;
    int32_t src_idx = clamp( 2*(block_x + idx), src_w );
    int32_t src_y   = clamp( 2*block_y, src_h );
    float value     = src_data.ptr(src_y)[src_idx];
    px[threadIdx.x] = value;
    __syncthreads();

    if( threadIdx.x >= V6_WIDTH ) return;

    float out = 0;
    #pragma unroll
    for( int i=0; i<V6_FILTERSIZE; i++ ) {
        out += px[threadIdx.x+i] * popart::d_gauss_filter[V6_GAUSS_BASE+i];
    }

    int dst_row   = block_x + threadIdx.x;
    int dst_col   = block_y;
    if( dst_col < 0 || dst_col >= dst_w ) return;
    if( dst_row < 0 || dst_row >= dst_h ) return;

    dst_data.ptr(dst_row)[dst_col] = out;
}

/*************************************************************
 * V6: host side
 *************************************************************/
__host__
void Pyramid::build_v6( Image* base )
{
#if 0
    cerr << "Entering " << __FUNCTION__ << " with base image "  << endl
         << "    type size         : " << base->type_size << endl
         << "    aligned byte size : " << base->a_width << "x" << base->a_height << endl
         << "    pitch size        : " << base->pitch << "x" << base->a_height << endl
         << "    original byte size: " << base->u_width << "x" << base->u_height << endl
         << "    aligned pix size  : " << base->a_width/base->type_size << "x" << base->a_height << endl
         << "    original pix size : " << base->u_width/base->type_size << "x" << base->u_height << endl;

#endif

    dim3 block;
    block.x = V6_READ_RANGE;

    for( int octave=0; octave<_num_octaves; octave++ ) {
        Plane2D_float s_data( _octaves[octave].getData(0) );
        Plane2D_float t_data( _octaves[octave].getTransposedData(0) );
        dim3 grid_t;
        dim3 grid;
        grid_t.x  = grid_divide( s_data.getWidth(), V6_WIDTH );
        grid_t.y  = s_data.getHeight();
        grid.x    = grid_divide( t_data.getWidth(), V6_WIDTH );
        grid.y    = t_data.getHeight();

#if 0
        cerr << "Configuration for octave " << octave << endl
             << "  Normal-to-transposed: layer size: "
             << _octaves[octave].getData(0).getWidth() << "x" << _octaves[octave].getData(0).getHeight() << endl
             << "                        grid: "
             << "(" << grid_t.x << "," << grid_t.y << "," << grid_t.z << ")"
             << " block: "
             << "(" << block.x << "," << block.y << "," << block.z << ")" << endl
             << "  Transposed-to-normal: layer size: "
             << _octaves[octave].getTransposedData(0).getWidth() << "x" << _octaves[octave].getTransposedData(0).getHeight() << endl
             << "                        grid: "
             << "(" << grid.x << "," << grid.y << "," << grid.z << ")"
             << " block: "
             << "(" << block.x << "," << block.y << "," << block.z << ")" << endl;
#endif

        for( int level=0; level<V6_LEVELS; level++ ) {

            if( level == 0 ) {
                if( octave == 0 ) {
                    filter_gauss_horiz_v6
                        <<<grid_t,block>>>
                        ( base->array,
                          _octaves[octave].getTransposedData( level ) );
                } else {
                    filter_gauss_horiz_v6_by_2
                        <<<grid_t,block>>>
                        ( _octaves[octave-1].getData( V6_LEVELS-1 ),
                          _octaves[octave].getTransposedData( level ) );
                }
            } else {
                filter_gauss_horiz_v6
                    <<<grid_t,block>>>
                    ( _octaves[octave].getData( level-1 ),
                      _octaves[octave].getTransposedData( level ) );
            }
            cudaDeviceSynchronize( );
            cudaError_t err = cudaGetLastError();
            POP_CUDA_FATAL_TEST( err, "filter_gauss_horiz_v6 failed: " );

            if( level == 0 ) {
                filter_gauss_horiz_v6
                    <<<grid,block>>>
                    ( _octaves[octave].getTransposedData( level ),
                      _octaves[octave].getData( level ) );
            } else {
#ifdef USE_DOG_ARRAY
                filter_gauss_horiz_v6_and_dog
                    <<<grid,block>>>
                    ( _octaves[octave].getTransposedData( level ),
                      _octaves[octave].getData( level ),
                      _octaves[octave].getData( level-1 ),
                      _octaves[octave].getDogSurface(),
                      level-1 );
#else // not USE_DOG_ARRAY
                filter_gauss_horiz_v6_and_dog
                    <<<grid,block>>>
                    ( _octaves[octave].getTransposedData( level ),
                      _octaves[octave].getData( level ),
                      _octaves[octave].getData( level-1 ),
                      _octaves[octave].getDogData( level-1 ) );
#endif // not USE_DOG_ARRAY
            }
            cudaDeviceSynchronize( );
            err = cudaGetLastError();
            POP_CUDA_FATAL_TEST( err, "filter_gauss_horiz_v6 failed: " );
        }
    }
}

} // namespace popart

