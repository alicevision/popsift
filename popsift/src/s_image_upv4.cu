#include "s_image.hpp"
#include "clamp.hpp"
#include "assist.h"

#include <iostream>

using namespace std;

namespace popart {

__global__
void p_upscale_4( Plane2D_float dst, Plane2D_uint8 src )
{
    int dst_xko  = blockIdx.x * blockDim.x + threadIdx.x;
    if( dst_xko >= dst.getCols() ) return;

    const int dst_ypos_1 = blockIdx.y * 2;
    if( dst_ypos_1 >= dst.getRows() ) return;
    const int dst_ypos_2 = dst_ypos_1 + 1;
    const int src_ypos_1 = clamp( blockIdx.y,   src.getRows() );
    const int src_ypos_2 = clamp( blockIdx.y+1, src.getRows() );

    /* we get copy operation in X direction automatically:
     * example:
     *  10 >> 1 -> 5 and 11 >> 1 -> 5
     *  11 >> 1 -> 5 and 12 >> 1 -> 6
     */
    const int src_xpos_1 = clamp(  dst_xko    >> 1, src.getCols() );
    const int src_xpos_2 = clamp( (dst_xko+1) >> 1, src.getCols() );

    const int v00  = src.ptr(src_ypos_1)[src_xpos_1];
    const int v01  = src.ptr(src_ypos_1)[src_xpos_2];
    const int v10  = src.ptr(src_ypos_2)[src_xpos_1];
    const int v11  = src.ptr(src_ypos_2)[src_xpos_2];

    const float dst_1 = ( v00 + v01 ) * 0.5f;
    const float dst_2 = ( v00 + v01 + v10 + v11 ) * 0.25f;

    dst.ptr(dst_ypos_1)[dst_xko] = dst_1;
    dst.ptr(dst_ypos_2)[dst_xko] = dst_2;
}

__host__
void Image::upscale_v4( Image_uint8& src )
{
    cerr << "Merged even-odd method" << endl;

    dim3 grid( grid_divide( this->array.getCols(), 128 ),
               grid_divide( this->array.getRows(), 2 ) );
    dim3 block( 128 );

    p_upscale_4
        <<<grid,block,0,stream>>>
        ( this->array,
          src.array );

    test_last_error( __FILE__,  __LINE__ );
}

} // namespace popart

