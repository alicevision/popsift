#include "s_image.h"
#include <iostream>
#include <fstream>
#include "debug_macros.h"
#include "assist.h"
#include <stdio.h>
#include <assert.h>

using namespace std;

namespace popart {

void Image::test_last_error( const char* file, int line )
{
    cudaError_t err;
    err = cudaGetLastError();
    if( err != cudaSuccess ) {
        printf("Error in %s:%d\n     CUDA failed: %s\n", file, line, cudaGetErrorString(err) );
        exit( -__LINE__ );
    }
}

Image::Image( size_t w, size_t h )
    : _w(w), _h(h)
{
    cudaError_t err;

    _input_image_h.allocHost( w, h, popart::CudaAllocated );

    _input_image_d.allocDev( w, h );

    cout << "Upscaled size of the input image: " << 2*w << "X" << 2*h << endl;

    /* initializing texture for upscaling
     */
    memset( &_input_image_texDesc, 0, sizeof(cudaTextureDesc) );
    _input_image_texDesc.normalizedCoords = 1; // address 0..1 instead of 0..width/height
    _input_image_texDesc.addressMode[0]   = cudaAddressModeClamp;
    _input_image_texDesc.addressMode[1]   = cudaAddressModeClamp;
    _input_image_texDesc.addressMode[2]   = cudaAddressModeClamp;
    _input_image_texDesc.readMode         = cudaReadModeNormalizedFloat; // automatic conversion from uchar to float
    _input_image_texDesc.filterMode       = cudaFilterModeLinear; // bilinear interpolation
    // _input_image_texDesc.filterMode       = cudaFilterModePoint; // nearest neighbour mode

    memset( &_input_image_resDesc, 0, sizeof(cudaResourceDesc) );
    _input_image_resDesc.resType                  = cudaResourceTypePitch2D;
    _input_image_resDesc.res.pitch2D.devPtr       = _input_image_d.data;
    _input_image_resDesc.res.pitch2D.desc.f       = cudaChannelFormatKindUnsigned;
    _input_image_resDesc.res.pitch2D.desc.x       = 8; // sizeof(uint8_t)*8
    _input_image_resDesc.res.pitch2D.desc.y       = 0;
    _input_image_resDesc.res.pitch2D.desc.z       = 0;
    _input_image_resDesc.res.pitch2D.desc.w       = 0;
    assert( _input_image_d.elemSize() == 1 );
    _input_image_resDesc.res.pitch2D.pitchInBytes = _input_image_d.step;
    _input_image_resDesc.res.pitch2D.width        = _input_image_d.getCols();
    _input_image_resDesc.res.pitch2D.height       = _input_image_d.getRows();

    err = cudaCreateTextureObject( &_input_image_tex, &_input_image_resDesc, &_input_image_texDesc, 0 );
    POP_CUDA_FATAL_TEST( err, "Could not create texture object: " );
}

Image::~Image( )
{
    cudaError_t err;
    err = cudaDestroyTextureObject( _input_image_tex );
    POP_CUDA_FATAL_TEST( err, "Could not destroy texture object: " );

    _input_image_d   .freeDev( );
    _input_image_h   .freeHost( popart::CudaAllocated );
}

void Image::load( const Config& conf, const unsigned char* input ) // const imgStream& inp )
{
    // memcpy( _input_image_h.data, inp.data_r, _w*_h );
    memcpy( _input_image_h.data, input, _w*_h );
    _input_image_h.memcpyToDevice( _input_image_d );
}

} // namespace popart

