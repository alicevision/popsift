#include "s_image.hpp"
#include "keep_time.hpp"
#include <iostream>
#include <fstream>
#include "debug_macros.hpp"
#include "align_macro.hpp"
#include "assist.h"
#include <stdio.h>
#include <assert.h>

using namespace std;

namespace popart {

__global__
void p_upscale_5( float* dst, uint32_t dst_pitch, uchar* src, uint32_t src_pitch, uint32_t src_width, uint32_t src_height )
{
    uint32_t src_ypos = min( src_height, blockIdx.y ) * src_pitch;
    uint32_t src_ypo2 = min( src_height, blockIdx.y+1 ) * src_pitch;

    uint32_t dst_xko  = blockIdx.x * blockDim.x + threadIdx.x;
    bool     nix      = ( dst_xko >= ( src_width << 1 ) );

    uint32_t src_1    = min( src_width, dst_xko / 2 );
    uint32_t src_2    = min( src_width, src_1 + ( dst_xko % 2 != 0 ) );

    uint32_t src_v_1  = src[ src_ypos + src_1 ];
    uint32_t src_v_2  = src[ src_ypos + src_2 ];
    uint32_t src_v_3  = src[ src_ypo2 + src_1 ];
    uint32_t src_v_4  = src[ src_ypo2 + src_2 ];

    uint32_t dst_ypos = blockIdx.y * 2 * dst_pitch;
    uint32_t dst_ypo2 = dst_ypos + dst_pitch;
    // assert( dst_xko < dst_pitch ); // alignment should guarantee this
    float    dst_1    = ( src_v_1 + src_v_2 ) / 2.0;
    float    dst_2    = ( src_v_1 + src_v_2 + src_v_3 + src_v_4 ) / 4.0;

    dst[ dst_ypos + dst_xko ] = nix ? 0 : dst_1;
    dst[ dst_ypo2 + dst_xko ] = nix ? 0 : dst_2;
}

__global__
void p_upscale_4_even( float* dst, uint32_t dst_pitch, uchar* src, uint32_t src_pitch, uint32_t src_width, uint32_t src_height )
{
    uint32_t src_ypos = min( src_height, blockIdx.y ) * src_pitch;
    uint32_t dst_ypos = blockIdx.y * 2 * dst_pitch;

    uint32_t dst_xko  = blockIdx.x * blockDim.x + threadIdx.x;
    bool     nix      = ( dst_xko >= ( src_width << 1 ) );
    // assert( dst_xko < dst_pitch ); // alignment should guarantee this

    uint32_t src_1    = min( src_width, dst_xko / 2 );
    uint32_t src_2    = min( src_width, src_1 + ( dst_xko % 2 != 0 ) );
    src_1             = src[ src_ypos + src_1 ];
    src_2             = src[ src_ypos + src_2 ];

    float    dst_1    = ( src_1 + src_2 ) / 2.0;
    dst[ dst_ypos + dst_xko ] = nix ? 0 : dst_1;
}

__global__
void p_upscale_4_odd( float* dst, uint32_t width, uint32_t pitch, uint32_t height )
{
    uint32_t ypos  = blockIdx.y * 2;
    if( ypos + 1 >= height ) return;

    uint32_t xpos  = blockIdx.x * blockDim.x + threadIdx.x;
    bool     nix   = ( xpos >= width );

    float    value = dst[ ypos * pitch + xpos ];
    value         += dst[ min( ypos + 2, height - 1) * pitch + xpos ];
    value         /= 2.0;
    __syncthreads();
    dst[ ( ypos + 1 ) * pitch + xpos ] = nix ? 0 : value;
}

__global__
void p_upscale_3( float4* dst, uint32_t dst_pitch, const uchar2* src, uint32_t src_width, uint32_t src_pitch, uint32_t src_height )
{
    int column = blockIdx.x*blockDim.x+threadIdx.x;
    bool nix   = ( column > src_width );
    column     = min( column, src_pitch );

    const int startline = blockIdx.y;
    const int endline   = src_height / blockDim.y;
    register uchar2   c0[2];
    register uchar2   c1[2];
    register float4   g0[3];
    register float    g1[3];
    c0[0]   = src[startline*src_pitch+column];
    c1[0]   = src[startline*src_pitch+column+1];
    g0[0].x = c0[0].x;
    g0[0].z = c0[0].y;
    g1[0]   = c1[0].x;
    g0[0].y = ( g0[0].x + g0[0].z ) / 2.0;
    g0[0].w = ( g0[0].z + g1[0]   ) / 2.0;
    dst[ column ] = nix ? make_float4(0,0,0,0) : g0[0];

    register int      toggle = 1;
    for( int srcline=startline+1; srcline<endline; srcline += 2 )
    {
        if( srcline > src_height ) return;

        const int dstline  = srcline * 2;
        const int toggleX2 = toggle << 1;
        c0[toggle]     = src[ srcline * src_pitch + column ];
        c1[toggle]     = src[ srcline * src_pitch + column + 1 ];
        g0[toggleX2].x = c0[toggle].x;
        g0[toggleX2].z = c0[toggle].y;
        g1[toggleX2]   = c1[toggle].x;
        g0[toggleX2].y = ( g0[toggleX2].x + g0[toggleX2].z ) / 2.0;
        g0[toggleX2].w = ( g0[toggleX2].z + g1[toggleX2]   ) / 2.0;
        g0[1].x        = ( g0[0].x + g0[2].x ) / 2.0;
        g0[1].y        = ( g0[0].y + g0[2].y ) / 2.0;
        g0[1].z        = ( g0[0].z + g0[2].z ) / 2.0;
        g0[1].w        = ( g0[0].w + g0[2].w ) / 2.0;
        dst[ (dstline-1) * dst_pitch + column ] = nix ? make_float4(0,0,0,0) : g0[1];
        dst[  dstline    * dst_pitch + column ] = nix ? make_float4(0,0,0,0) : g0[toggle];
        toggle = !toggle;
    }
}

__global__
void p_upscale_2_very_simple( float* dst, uint32_t dst_pitch, const uchar* src, uint32_t src_pitch, uint32_t src_height )
{
    uint32_t y_ko_0 = blockIdx.y*blockDim.y+threadIdx.y;
    uint32_t x_ko_0 = blockIdx.x*blockDim.x+threadIdx.x;
    uint32_t y_ko_1 = min( y_ko_0 + 1, src_height );
    uint32_t x_ko_1 = min( x_ko_0 + 1, src_pitch );
    uint32_t c00 = src[ (y_ko_0) * src_pitch + (x_ko_0) ];
    uint32_t c02 = src[ (y_ko_0) * src_pitch + (x_ko_1) ];
    uint32_t c20 = src[ (y_ko_1) * src_pitch + (x_ko_0) ];
    uint32_t c22 = src[ (y_ko_1) * src_pitch + (x_ko_1) ];

    float g00 = (float)c00;
    // float g02 = (float)c02;
    // float g20 = (float)c20;
    // float g22 = (float)c22;
    float g01 = (float)(c00+c02) / 2.0;
    float g10 = (float)(c00+c20) / 2.0;
    float g12 = (float)(c02+c22) / 2.0;
    float g21 = (float)(c20+c22) / 2.0;
    float g11 = ( g01 + g10 + g12 + g21 ) / 4.0;

    y_ko_0 <<= 1;
    x_ko_0 <<= 1;
    y_ko_1 = min( y_ko_0 + 1, src_height<<1 );
    x_ko_1 = min( x_ko_0 + 1, dst_pitch );
    dst[y_ko_0 * dst_pitch + x_ko_0] = g00;
    dst[y_ko_0 * dst_pitch + x_ko_1] = g01;
    dst[y_ko_1 * dst_pitch + x_ko_0] = g10;
    dst[y_ko_1 * dst_pitch + x_ko_1] = g11;
}

__host__
void Image::upscale_v1( Image& src )
{
    printf("Method 1\n");
    dim3 grid( src.u_width, src.u_height );
    dim3 block( 1 );
    printf("Grid (%d,%d,%d) block (%d,%d,%d)\n",grid.x,grid.y,grid.z,block.x,block.y,block.z);

    _keep_time_image_v2.start();
    p_upscale_2_very_simple
        <<<grid,block,0,stream>>>
        ( (float*)this->array,
          this->pitch/sizeof(float),
          src.array,
          src.pitch,
          src.a_height );
    _keep_time_image_v2.stop();

    test_last_error( __FILE__,  __LINE__ );
}

__host__
void Image::upscale_v2( Image& src )
{
    printf("Method 2\n");
    dim3 grid( src.u_width/128/2, 32 );
    dim3 block( 128 );

    float4* dest   = (float4*)(this->array);
    uchar2* source = (uchar2*)(src.array);

    _keep_time_image_v3.start();
    p_upscale_3
        <<<grid,block,0,stream>>>
        ( dest,
          this->pitch/sizeof(float4),
          source,
          src.u_width/sizeof(uchar2),
          src.pitch/sizeof(uchar2),
          src.u_height );
    _keep_time_image_v3.stop();

    test_last_error( __FILE__,  __LINE__ );
}

__host__
void Image::upscale_v3( Image& src )
{
    printf("Method 2\n");
    dim3 grid( src.u_width/128/2, 32 );
    dim3 block( 128 );

    float4* dest   = (float4*)(this->array);
    uchar2* source = (uchar2*)(src.array);

    _keep_time_image_v3.start();
    p_upscale_3
        <<<grid,block,0,stream>>>
        ( dest,
          this->pitch/sizeof(float4),
          source,
          src.u_width/sizeof(uchar2),
          src.pitch/sizeof(uchar2),
          src.u_height );
    _keep_time_image_v3.stop();

    test_last_error( __FILE__,  __LINE__ );
}

__host__
void Image::upscale_v4( Image& src )
{
    printf("Seperated even-odd method\n");
    dim3 grid( this->a_width/128/sizeof(float), this->a_height/2 );
    dim3 block( 128 );

    float* dest   = (float*)(this->array);
    uchar* source = src.array;

    _keep_time_image_v4.start();
    p_upscale_4_even
        <<<grid,block,0,stream>>>
        ( dest,
          this->pitch/sizeof(float),
          source,
          src.pitch,
          src.u_width,
          src.u_height );

    p_upscale_4_odd
        <<<grid,block,0,stream>>>
        ( dest,
          src.u_width << 1,
          this->pitch/sizeof(float4),
          this->a_height );
    _keep_time_image_v4.stop();

    test_last_error( __FILE__,  __LINE__ );
}

__host__
void Image::upscale_v5( Image& src )
{
    printf("Merged even-odd method\n");
    dim3 grid( grid_divide( this->a_width, 128*sizeof(float) ), grid_divide( this->a_height, 2 ) );
    dim3 block( 128 );

    float* dest   = (float*)(this->array);
    uchar* source = src.array;

    _keep_time_image_v5.start();
    p_upscale_5
        <<<grid,block,0,stream>>>
        ( dest,
          this->pitch/sizeof(float),
          source,
          src.pitch,
          src.u_width,
          src.u_height );
    _keep_time_image_v5.stop();

    test_last_error( __FILE__,  __LINE__ );
}

__host__
void Image::upscale( Image& src, size_t scalefactor )
{
    if( scalefactor != 2 ) {
        cerr << "Scale factor is " << scalefactor << endl;
        cerr << "Currently only 2 is supported" << endl;
        exit( -__LINE__ );
    }
    assert( this->type_size == sizeof(float) );
    assert( src.type_size   == sizeof(uchar) );

    if( false ) upscale_v1( src );
    if( false ) upscale_v2( src );
    if( false ) upscale_v3( src );
    if( false ) upscale_v4( src );
    if( true  ) upscale_v5( src );
}

void Image::report_times( )
{
    if( false ) _keep_time_image_v1.report( "    V1, Time for image upscale: " );
    if( false ) _keep_time_image_v2.report( "    V2, Time for image upscale: " );
    if( false ) _keep_time_image_v3.report( "    V3, Time for image upscale: " );
    if( false ) _keep_time_image_v4.report( "    V4, Time for image upscale: " );
    if( true  ) _keep_time_image_v5.report( "    V5, Time for image upscale: " );
}

void Image::test_last_error( const char* file, int line )
{
    cudaError_t err;
    err = cudaGetLastError();
    if( err != cudaSuccess ) {
        printf("Error in %s:%d\n     CUDA failed: %s\n", file, line, cudaGetErrorString(err) );
        exit( -__LINE__ );
    }
}

void Image::download_and_save_array( const char* filename )
{
    cerr << "Downloading image from GPU to CPU and writing to file " << filename << endl;
    int width = a_width / sizeof(float);
    float* f = new float[ width * a_height ];
    POP_CUDA_MEMCPY_ASYNC( f,
                           this->array,
                           a_width * a_height,
                           cudaMemcpyDeviceToHost,
                           0,
                           false );
    unsigned char* c = new unsigned char[ width * a_height ];
    for( int i=0; i<width * a_height; i++ ) {
        c[i] = (unsigned char)(f[i]);
    }
    ofstream of( filename );
    of << "P5" << endl
       << width << " " << a_height << endl
       << "255" << endl;
    of.write( (char*)c, width * a_height );
    delete [] c;
    delete [] f;
}

Image::Image( size_t w, size_t h, size_t t, cudaStream_t s )
    : stream( s )
    , a_width ( w*t )
    , a_height( h )
    , u_width ( w*t )
    , u_height( h )
    , type_size( t )
    , _keep_time_image_v1( s )
    , _keep_time_image_v2( s )
    , _keep_time_image_v3( s )
    , _keep_time_image_v4( s )
    , _keep_time_image_v5( s )
{
    align( a_width,  128 ); // 0x80
    align( a_height, 128 ); // 0x80
    cudaError_t err;
    err = cudaMallocPitch( &array, &pitch, a_width, a_height );
    POP_CUDA_FATAL_TEST( err, "cudaMallocPitch failed for array: " );
}

Image::Image( imgStream& gray, cudaStream_t s )
    : stream( s )
    , a_width ( gray.width )
    , a_height( gray.height )
    , u_width ( gray.width )
    , u_height( gray.height )
    , type_size( sizeof(uchar) )
    , _keep_time_image_v1( s )
    , _keep_time_image_v2( s )
    , _keep_time_image_v3( s )
    , _keep_time_image_v4( s )
    , _keep_time_image_v5( s )
{
    align( a_width,  128 ); // 0x80
    align( a_height, 128 ); // 0x80
    cudaError_t err;
    err = cudaMallocPitch( &array, &pitch, a_width, a_height );
    POP_CUDA_FATAL_TEST( err, "cudaMallocPitch failed for array: " );

    err = cudaMemcpy2DAsync( array,       pitch,
                             gray.data_r, gray.width,
                             gray.width,
                             gray.height,
                             cudaMemcpyHostToDevice,
                             stream );
    POP_CUDA_FATAL_TEST( err, "cudaMemcpy2D failed for image upload: " );

}

Image::~Image( )
{
    cudaFree( array );
}

} // namespace popart

