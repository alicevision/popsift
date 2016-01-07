#include "s_image.hpp"
#include "clamp.hpp"
#include "assist.h"

#include <iostream>

using namespace std;

namespace popart {

__global__
void p_upscale_1( Plane2D_float dst, Plane2D_uint8 src )
{
    uint32_t y_ko_0 = clamp( blockIdx.y*blockDim.y+threadIdx.y, src.getRows() );
    uint32_t x_ko_0 = clamp( blockIdx.x*blockDim.x+threadIdx.x, src.getCols() );
    uint32_t y_ko_1 = clamp( y_ko_0 + 1, src.getRows() );
    uint32_t x_ko_1 = clamp( x_ko_0 + 1, src.getCols() );
    uint32_t c00 = src.ptr(y_ko_0)[x_ko_0];
    uint32_t c02 = src.ptr(y_ko_0)[x_ko_1];
    uint32_t c20 = src.ptr(y_ko_1)[x_ko_0];
    uint32_t c22 = src.ptr(y_ko_1)[x_ko_1];

    float g00 = (float)c00;
    // float g02 = (float)c02;
    // float g20 = (float)c20;
    // float g22 = (float)c22;
    float g01 = (float)(c00+c02) * 0.5f; // / 2.0;
    float g10 = (float)(c00+c20) * 0.5f; // / 2.0;
    float g12 = (float)(c02+c22) * 0.5f; // / 2.0;
    float g21 = (float)(c20+c22) * 0.5f; // / 2.0;
    float g11 = ( g01 + g10 + g12 + g21 ) * 0.25f; // / 4.0;

    y_ko_0 = ( blockIdx.y*blockDim.y+threadIdx.y ) << 1; // * 2
    x_ko_0 = ( blockIdx.x*blockDim.x+threadIdx.x ) << 1; // * 2
    y_ko_1 = y_ko_0 + 1;
    if( y_ko_0 >= dst.getRows() ) return;
    if( x_ko_0 >= dst.getCols() ) return;
    dst.ptr(y_ko_0)[x_ko_0] = g00;
    if( y_ko_1 < dst.getRows() ) dst.ptr(y_ko_1)[x_ko_0] = g10;
    x_ko_1 = x_ko_0 + 1;
    if( x_ko_1 >= dst.getCols() ) return;
    dst.ptr(y_ko_0)[x_ko_1] = g01;
    if( y_ko_1 < dst.getRows() ) dst.ptr(y_ko_1)[x_ko_1] = g11;
}

__host__
void Image::upscale_v1( Image_uint8& src, cudaStream_t stream )
{
    cerr << "Upscaling method 1" << endl;
    dim3 grid( grid_divide( src.array.getCols(), 128 ), src.array.getRows() );
    dim3 block( 128 );

    p_upscale_1
        <<<grid,block,0,stream>>>
        ( this->array,
          src.array );

    test_last_error( __FILE__,  __LINE__ );
}

} // namespace popart

