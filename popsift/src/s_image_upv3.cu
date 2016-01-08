#include "s_image.h"
#include "clamp.h"
#include "assist.h"

#include <iostream>

using namespace std;

namespace popart {

__global__
void p_upscale_3_even( Plane2D_float dst, Plane2D_uint8 src )
{
    const int src_yko = clamp( blockIdx.y,     src.getRows() );
    const int dst_yko = clamp( blockIdx.y * 2, dst.getRows() );

    const int  dst_xko = blockIdx.x * blockDim.x + threadIdx.x;

    if( dst_xko >= dst.getCols() ) return;
    if( dst_yko >= dst.getRows() ) return;

    const int src_xko = clamp( dst_xko >> 1, src.getRows() );
    const int in1     = src.ptr(src_yko)[src_xko  ];
    const int in2     = src.ptr(src_yko)[src_xko+1];

    const float out1 = ( in1 + in2 ) * 0.5f; // / 2.0;
    dst.ptr(dst_yko)[dst_xko]   = in1;
    dst.ptr(dst_yko)[dst_xko+1] = out1;
}

__global__
void p_upscale_3_odd( Plane2D_float dst )
{
    const int ypos0 = blockIdx.y * 2;
    const int ypos1 = ypos0 + 1;
    if( ypos1 >= dst.getRows() ) return;
    const int ypos2 = clamp( ypos0 + 2, dst.getRows()-1 );

    const int xpos  = blockIdx.x * blockDim.x + threadIdx.x;
    if( xpos >= dst.getCols() ) return;

    const float v0 = dst.ptr(ypos0)[xpos];
    const float v2 = dst.ptr(ypos2)[xpos];
    const float v1 = ( v0 + v2 ) * 0.5f;

    dst.ptr(ypos1)[xpos] = v1;
}

__host__
void Image::upscale_v3( Plane2D_uint8 & src, cudaStream_t stream )
{
    cerr << "Separated even-odd method" << endl;

    dim3 grid( grid_divide( this->array.getCols(), 128 ),
               grid_divide( this->array.getRows(), 2 ) );
    dim3 block( 128 );

    p_upscale_3_even
        <<<grid,block,0,stream>>>
        ( this->array,
          src );

    p_upscale_3_odd
        <<<grid,block,0,stream>>>
        ( this->array );

    test_last_error( __FILE__,  __LINE__ );
}

} // namespace popart

