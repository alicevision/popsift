#include "s_pyramid.h"

#include "gauss_filter.h"
#include "clamp.h"
#include "debug_macros.h"
#include "assist.h"
#include <cuda_runtime.h>

/*************************************************************
 * V12: device side
 *************************************************************/

#define V12_EDGE_LEN    32
#define V12_RANGE    4 // RANGES from 1 to 8 are possible
#define V12_GAUSS_BASE   ( GAUSS_ONE_SIDE_RANGE - V12_RANGE )
#define V12_FILTERSIZE   ( V12_RANGE + 1        + V12_RANGE )
#define V12_LEVELS       _levels

namespace popart {

__global__
void filter_gauss_horiz_v12( Plane2D_float src_data,
                            Plane2D_float dst_data )
{

    __shared__ float loaddata[V12_EDGE_LEN][V12_RANGE + V12_EDGE_LEN + V12_RANGE];

    const int src_w = src_data.getWidth();
    const int src_h = src_data.getHeight();

    int block_x = blockIdx.x * V12_EDGE_LEN;
    int block_y = blockIdx.y * V12_EDGE_LEN;
    int idx     = threadIdx.x;
    int idy     = threadIdx.y;
    for( ; idx < V12_EDGE_LEN+2*V12_RANGE; idx += V12_EDGE_LEN) {
        int read_x = clamp( block_x + idx - V12_RANGE, src_w );
        int read_y = clamp( block_y + idy,            src_h );
        loaddata[idy][idx] = src_data.ptr(read_y)[read_x];
    }
    __syncthreads();

    float g;
    float val;
    float out = 0;

    for( int offset = V12_RANGE; offset>0; offset-- ) {
        g  = popart::d_gauss_filter[GAUSS_ONE_SIDE_RANGE - offset];

        idx = threadIdx.x - offset;
        val = loaddata[threadIdx.y][idx+V12_RANGE];
        out += ( val * g );

        idx = threadIdx.x + offset;
        val = loaddata[threadIdx.y][idx+V12_RANGE];
        out += ( val * g );
    }

    g  = popart::d_gauss_filter[GAUSS_ONE_SIDE_RANGE];
    idx = threadIdx.x;
    val = loaddata[threadIdx.y][idx+V12_RANGE];
    out += ( val * g );

    idx = block_x+threadIdx.x;
    idy = block_y+threadIdx.y;
    if( idx >= src_w ) return;
    if( idy >= src_h ) return;

    dst_data.ptr(idy)[idx] = out;
}
__global__
void filter_gauss_horiz_v12( cudaTextureObject_t src_data,
                             Plane2D_float       dst_data )
{
    __shared__ float loaddata[V12_EDGE_LEN][V12_RANGE + V12_EDGE_LEN + V12_RANGE];

    const int dst_w = dst_data.getWidth();
    const int dst_h = dst_data.getHeight();

    int block_x = blockIdx.x * V12_EDGE_LEN;
    int block_y = blockIdx.y * V12_EDGE_LEN;
    int idx     = threadIdx.x;
    int idy     = threadIdx.y;
    for( ; idx < V12_EDGE_LEN+2*V12_RANGE; idx += V12_EDGE_LEN) {
        int read_x = block_x + idx - V12_RANGE;
        int read_y = block_y + idy;
        float f = tex2D<float>( src_data, read_x, read_y );
        loaddata[idy][idx] = f;
    }
    __syncthreads();

    float g;
    float val;
    float out = 0;

    for( int offset = V12_RANGE; offset>0; offset-- ) {
        g  = popart::d_gauss_filter[GAUSS_ONE_SIDE_RANGE - offset];

        idx = threadIdx.x - offset;
        val = loaddata[threadIdx.y][idx+V12_RANGE];
        out += ( val * g );

        idx = threadIdx.x + offset;
        val = loaddata[threadIdx.y][idx+V12_RANGE];
        out += ( val * g );
    }

    g  = popart::d_gauss_filter[GAUSS_ONE_SIDE_RANGE];
    idx = threadIdx.x;
    val = loaddata[threadIdx.y][idx+V12_RANGE];
    out += ( val * g );

    idx = block_x+threadIdx.x;
    idy = block_y+threadIdx.y;
    if( idx >= dst_w ) return;
    if( idy >= dst_h ) return;

    dst_data.ptr(idy)[idx] = out;
}

__device__
void filter_gauss_vert_v12_sub( Plane2D_float&  src_data,
                               Plane2D_float&  dst_data )
{
    // does not work on Mac !
    // assert( blockDim.x == blockDim.y );
    const int src_w   = src_data.getWidth();
    const int src_h   = src_data.getHeight();

    /* loaddata is transposed with respect to the src plane */
    __shared__ float loaddata[V12_EDGE_LEN][V12_RANGE + V12_EDGE_LEN + V12_RANGE];

    int block_x = blockIdx.x * V12_EDGE_LEN;
    int block_y = blockIdx.y * V12_EDGE_LEN;
    int idx     = threadIdx.x;
    int idy     = threadIdx.y;
    for( ; idy < V12_EDGE_LEN+2*V12_RANGE; idy += V12_EDGE_LEN) {
        int read_x = clamp( block_x + idx,            src_w );
        int read_y = clamp( block_y + idy - V12_RANGE, src_h );
        loaddata[idx][idy] = src_data.ptr(read_y)[read_x];
    }
    __syncthreads();

    float g;
    float val;
    float out = 0;

    for( int offset = V12_RANGE; offset>0; offset-- ) {
        g  = popart::d_gauss_filter[GAUSS_ONE_SIDE_RANGE - offset];

        idx = threadIdx.x - offset;
        val = loaddata[threadIdx.y][idx+V12_RANGE];
        out += ( val * g );

        idx = threadIdx.x + offset;
        val = loaddata[threadIdx.y][idx+V12_RANGE];
        out += ( val * g );
    }

    g  = popart::d_gauss_filter[GAUSS_ONE_SIDE_RANGE];
    idx = threadIdx.x;
    val = loaddata[threadIdx.y][idx+V12_RANGE];
    out += ( val * g );

    idx = block_x+threadIdx.x;
    idy = block_y+threadIdx.y;
    if( idx >= src_w ) return;
    if( idy >= src_h ) return;

    dst_data.ptr(idy)[idx] = out;
}

__global__
void filter_gauss_vert_v12( Plane2D_float   src_data,
                           Plane2D_float   dst_data )
{
    filter_gauss_vert_v12_sub( src_data, dst_data );
}

__global__
void filter_gauss_vert_v12_and_dog( Plane2D_float   src_data,
                                   Plane2D_float   dst_data,
                                   Plane2D_float   higher_level_data,
                                   Plane2D_float   dog_data )
{
    filter_gauss_vert_v12_sub( src_data, dst_data );

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
void filter_gauss_horiz_v12_by_2( cudaTextureObject_t src_data,
                                  Plane2D_float       dst_data )
{
    int block_x = blockIdx.x * blockDim.x;
    int block_y = blockIdx.y;

    int       src_idx;
    const int src_idy = 2 * block_y;
    const int dst_w   = dst_data.getWidth();
    const int dst_h   = dst_data.getHeight();
    const int dst_idx = block_x + threadIdx.x;
    const int dst_idy = block_y;

    if( dst_idx >= dst_w ) return;
    if( dst_idy >= dst_h ) return;

    float g;
    float val;
    float out = 0;

    for( int offset = V12_RANGE; offset>0; offset-- ) {
        g  = popart::d_gauss_filter[GAUSS_ONE_SIDE_RANGE - offset];

        src_idx = 2 * ( dst_idx - offset );
        val = tex2D<float>( src_data, src_idy, src_idx );
        out += ( val * g );

        src_idx = 2 * ( dst_idx + offset );
        val = tex2D<float>( src_data, src_idy, src_idx );
        out += ( val * g );
    }

    g  = popart::d_gauss_filter[GAUSS_ONE_SIDE_RANGE];
    src_idx = 2 * dst_idx;
    val = tex2D<float>( src_data, src_idy, src_idx );
    out += ( val * g );

    dst_data.ptr(dst_idy)[dst_idx] = out;
}

/*************************************************************
 * V12: host side
 *************************************************************/
__host__
void Pyramid::build_v12( Image* base )
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

    for( int octave=0; octave<_num_octaves; octave++ ) {
        dim3 block;
        block.x = V12_EDGE_LEN;
        block.y = V12_EDGE_LEN;

        dim3 grid;
        const int width  = _octaves[octave].getData(0).getWidth();
        const int height = _octaves[octave].getData(0).getHeight();
        grid.x = grid_divide( width,  V12_EDGE_LEN );
        grid.y = grid_divide( height, V12_EDGE_LEN );

        for( int level=0; level<V12_LEVELS; level++ ) {
#if 0
        cerr << "Configuration for octave " << octave << endl
             << "  Horiz: layer size: "
             << _octaves[octave].getData(level).getWidth() << "x" << _octaves[octave].getData(level).getHeight() << endl
             << "  Vert: layer size: "
             << _octaves[octave].getIntermediateData().getWidth() << "x" << _octaves[octave].getIntermediateData().getHeight() << endl
             << "  grid: "
             << "(" << grid.x << "," << grid.y << "," << grid.z << ")"
             << " block: "
             << "(" << block.x << "," << block.y << "," << block.z << ")" << endl;
#endif


            if( level == 0 ) {
                if( octave == 0 ) {
                    filter_gauss_horiz_v12
                        <<<grid,block>>>
                        ( base->array,
                          _octaves[octave].getIntermediateData( ) );
                } else {
                    filter_gauss_horiz_v12_by_2
                        <<<grid,block>>>
                        ( _octaves[octave-1]._data_tex[ V12_LEVELS-3 ],
                          _octaves[octave].getIntermediateData( ) );
                }
            } else {
                filter_gauss_horiz_v12
                    <<<grid,block>>>
                    ( _octaves[octave]._data_tex[ level-1 ],
                      _octaves[octave].getIntermediateData( ) );
            }
            // cudaDeviceSynchronize( );
            // cudaError_t err = cudaGetLastError();
            // POP_CUDA_FATAL_TEST( err, "filter_gauss_horiz_v12 failed: " );

            if( level == 0 ) {
                filter_gauss_vert_v12
                    <<<grid,block>>>
                    ( _octaves[octave].getIntermediateData( ),
                      _octaves[octave].getData( level ) );
            } else {
                filter_gauss_vert_v12_and_dog
                    <<<grid,block>>>
                    ( _octaves[octave].getIntermediateData( ),
                      _octaves[octave].getData( level ),
                      _octaves[octave].getData( level-1 ),
                      _octaves[octave].getDogData( level-1 ) );
            }
            // cudaDeviceSynchronize( );
            // err = cudaGetLastError();
            // POP_CUDA_FATAL_TEST( err, "filter_gauss_horiz_v12 failed: " );
        }
    }
    cudaDeviceSynchronize( );
    cudaError_t err = cudaGetLastError();
    POP_CUDA_FATAL_TEST( err, "filter_gauss_horiz_v12 failed: " );
}

} // namespace popart

