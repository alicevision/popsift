#include "s_image.h"
#include "clamp.h"
#include "assist.h"

#include <iostream>

using namespace std;

namespace popart {

__global__
void p_upscale_2( Plane2D_float_4 dst, Plane2D_uchar_2 src )
{
    int column = blockIdx.x*blockDim.x+threadIdx.x;
    bool nix   = ( column > src.getCols() );
    column     = clamp( column, src.getCols() );

    const int startline = blockIdx.y;
    int       dstline   = startline * 2;
    const int endline   = startline + src.getRows() / blockDim.y;
    register uchar2   c0[2];
    register uchar2   c1[2];
    register float4   g0[3];
    register float    g1[3];
    c0[0]   = src.ptr(startline)[column  ];
    c1[0]   = src.ptr(startline)[column+1];
    g0[0].x = c0[0].x;
    g0[0].z = c0[0].y;
    g1[0]   = c1[0].x;
    g0[0].y = ( g0[0].x + g0[0].z ) * 0.5f; //  / 2.0;
    g0[0].w = ( g0[0].z + g1[0]   ) * 0.5f; //  / 2.0;
    dst.ptr(dstline)[column] = nix ? make_float4(0,0,0,0) : g0[0];

    register int      toggle = 1;
    for( int srcline=startline+1; srcline<endline; srcline += 2 )
    {
        if( srcline > src.getRows() ) return;

        dstline  = srcline * 2;
        const int toggleX2 = toggle << 1;
        c0[toggle]     = src.ptr(srcline)[column  ];
        c1[toggle]     = src.ptr(srcline)[column+1];
        g0[toggleX2].x = c0[toggle].x;
        g0[toggleX2].z = c0[toggle].y;
        g1[toggleX2]   = c1[toggle].x;
        g0[toggleX2].y = ( g0[toggleX2].x + g0[toggleX2].z ) * 0.5f; //  / 2.0;
        g0[toggleX2].w = ( g0[toggleX2].z + g1[toggleX2]   ) * 0.5f; //  / 2.0;
        g0[1].x        = ( g0[0].x + g0[2].x ) * 0.5f; //  / 2.0;
        g0[1].y        = ( g0[0].y + g0[2].y ) * 0.5f; //  / 2.0;
        g0[1].z        = ( g0[0].z + g0[2].z ) * 0.5f; //  / 2.0;
        g0[1].w        = ( g0[0].w + g0[2].w ) * 0.5f; //  / 2.0;
        dst.ptr(dstline-1)[column] = nix ? make_float4(0,0,0,0) : g0[1];
        dst.ptr(dstline  )[column] = nix ? make_float4(0,0,0,0) : g0[toggle];
        toggle = !toggle;
    }
}

__host__
void Image::upscale_v2( Plane2D_uint8 & src )
{
    cerr << "Upscaling method 2" << endl;
    dim3 grid( grid_divide( src.getCols(), 2*64 ), 32 );
    dim3 block( 64 );

    Plane2D_float_4 dest( this->array );
    Plane2D_uchar_2 source( src );

    p_upscale_2
        <<<grid,block>>>
        ( dest,
          source );

    test_last_error( __FILE__,  __LINE__ );
}

} // namespace popart

