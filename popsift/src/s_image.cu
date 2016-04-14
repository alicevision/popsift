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

    _upscaled_image_d.allocDev( 2 * w, 2 * h );

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

    /* initializing texture for access by octaves
     */
    memset( &_upscaled_image_texDesc, 0, sizeof(cudaTextureDesc) );
    _upscaled_image_texDesc.normalizedCoords = 1; // address 0..1 instead of 0..width/height
    _upscaled_image_texDesc.addressMode[0]   = cudaAddressModeClamp;
    _upscaled_image_texDesc.addressMode[1]   = cudaAddressModeClamp;
    _upscaled_image_texDesc.addressMode[2]   = cudaAddressModeClamp;
    _upscaled_image_texDesc.readMode         = cudaReadModeElementType; // no conversion, this is float
    _upscaled_image_texDesc.filterMode       = cudaFilterModeLinear; // bilinear interpolation

    memset( &_upscaled_image_resDesc, 0, sizeof(cudaResourceDesc) );
    _upscaled_image_resDesc.resType                  = cudaResourceTypePitch2D;
    _upscaled_image_resDesc.res.pitch2D.devPtr       = _upscaled_image_d.data;
    _upscaled_image_resDesc.res.pitch2D.desc.f       = cudaChannelFormatKindFloat; // float
    _upscaled_image_resDesc.res.pitch2D.desc.x       = 32; // sizeof(float)*8
    _upscaled_image_resDesc.res.pitch2D.desc.y       = 0;
    _upscaled_image_resDesc.res.pitch2D.desc.z       = 0;
    _upscaled_image_resDesc.res.pitch2D.desc.w       = 0;
    assert( _upscaled_image_d.elemSize() == 4 );
    _upscaled_image_resDesc.res.pitch2D.pitchInBytes = _upscaled_image_d.step;
    _upscaled_image_resDesc.res.pitch2D.width        = _upscaled_image_d.getCols();
    _upscaled_image_resDesc.res.pitch2D.height       = _upscaled_image_d.getRows();

    err = cudaCreateTextureObject( &_upscaled_image_tex, &_upscaled_image_resDesc, &_upscaled_image_texDesc, 0 );
    POP_CUDA_FATAL_TEST( err, "Could not create texture object: " );
}

Image::~Image( )
{
    cudaError_t err;
    err = cudaDestroyTextureObject( _input_image_tex );
    POP_CUDA_FATAL_TEST( err, "Could not destroy texture object: " );

    err = cudaDestroyTextureObject( _upscaled_image_tex );
    POP_CUDA_FATAL_TEST( err, "Could not destroy texture object: " );

    _upscaled_image_d.freeDev( );
    _input_image_d   .freeDev( );
    _input_image_h   .freeHost( popart::CudaAllocated );
}

void Image::load( imgStream inp )
{
    memcpy( _input_image_h.data, inp.data_r, _w*_h );
    _input_image_h.memcpyToDevice( _input_image_d );
    upscale_v5( _input_image_tex );
}

} // namespace popart

