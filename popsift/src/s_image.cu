#include "s_image.h"
#include <iostream>
#include <fstream>
#include "debug_macros.h"
#include "align_macro.h"
#include "assist.h"
#include <stdio.h>
#include <assert.h>

using namespace std;

namespace popart {

__host__
void Image::upscale( Plane2D_uint8 & src, size_t scalefactor, cudaStream_t stream )
{
    if( scalefactor != 2 ) {
        cerr << "Scale factor is " << scalefactor << endl;
        cerr << "Currently only 2 is supported" << endl;
        exit( -__LINE__ );
    }

    if( false ) upscale_v1( src, stream );
    if( false ) upscale_v2( src, stream );
    if( false ) upscale_v3( src, stream );
    if( true  ) upscale_v4( src, stream );
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

    Plane2D_float f;
    f.allocHost( this->array.getCols(), this->array.getRows(), PageAligned );

    f.memcpyFromDevice( array );

    unsigned char* c = new unsigned char[ f.getCols() * f.getRows() ];
    for( int y=0; y<f.getRows(); y++ ) {
        for( int x=0; x<f.getCols(); x++ ) {
            c[y*f.getCols()+x] = (unsigned char)(f.ptr(y)[x]);
        }
    }
    ofstream of( filename );
    of << "P5" << endl
       << f.getCols() << " " << f.getRows() << endl
       << "255" << endl;
    of.write( (char*)c, f.getCols() * f.getRows() );
    delete [] c;

    f.freeHost( PageAligned );
}

Image::Image( size_t w, size_t h )
{
    array.allocDev( w, h );
}

Image::~Image( )
{
    array.freeDev( );
}

} // namespace popart

