#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <assert.h>

#include "SIFT.h"
#include "debug_macros.h"
#include "write_plane_2d.h"

using namespace std;

PopSift::PopSift( int num_octaves,
                  int S_,
                  int upscale_factor,
                  float threshold,
                  float edgeThreshold,
                  float sigma )
    : _octaves( num_octaves )
    , _scales( max(2,S_) ) // min is 2, GPU restriction */
    , up( upscale_factor )
    , _sigma( sigma )
    , _threshold( threshold ) // SIFT parameter
    , _edgeLimit( edgeThreshold ) // SIFT parameter
{
}

PopSift::~PopSift()
{ }

#define TRY_IMAGE_TWICE 0

void PopSift::init( int w, int h )
{
    if (_octaves < 0) {
        _octaves = max(int (floor( logf( (float)min( w, h ) )
                                   / logf( 2.0f ) ) - 3 + up), 1);
    }

    _upscaled_width  = w << up;
    _upscaled_height = h << up;

    _hst_input_image.allocHost( w, h, popart::CudaAllocated );
    _dev_input_image.allocDev( w, h );

    float sigma = 1.0;

    popart::Pyramid::init_filter( sigma, _scales );
    popart::Pyramid::init_sigma(  sigma, _scales );

    _baseImg = new popart::Image( _upscaled_width, _upscaled_height );
    _pyramid = new popart::Pyramid( _baseImg, _octaves, _scales );

    /* initializing texture for upscale V5
     */

    memset( &_texDesc, 0, sizeof(cudaTextureDesc) );
    _texDesc.normalizedCoords = 1; // address 0..1 instead of 0..width/height
    _texDesc.addressMode[0]   = cudaAddressModeClamp;
    _texDesc.addressMode[1]   = cudaAddressModeClamp;
    _texDesc.addressMode[2]   = cudaAddressModeClamp;
    _texDesc.readMode         = cudaReadModeNormalizedFloat; // automatic conversion from uchar to float
    _texDesc.filterMode       = cudaFilterModeLinear; // bilinear interpolation

    memset( &_resDesc, 0, sizeof(cudaResourceDesc) );
    _resDesc.resType                  = cudaResourceTypePitch2D;
    _resDesc.res.pitch2D.devPtr       = _dev_input_image.data;
    _resDesc.res.pitch2D.desc.f       = cudaChannelFormatKindUnsigned;
    _resDesc.res.pitch2D.desc.x       = 8;
    _resDesc.res.pitch2D.desc.y       = 0;
    _resDesc.res.pitch2D.desc.z       = 0;
    _resDesc.res.pitch2D.desc.w       = 0;
    assert( _dev_input_image.elemSize() == 1 );
    _resDesc.res.pitch2D.pitchInBytes = _dev_input_image.step;
    _resDesc.res.pitch2D.width        = _dev_input_image.getCols();
    _resDesc.res.pitch2D.height       = _dev_input_image.getRows();

    cudaError_t err;
    err = cudaCreateTextureObject( &_texture, &_resDesc, &_texDesc, 0 );
    POP_CUDA_FATAL_TEST( err, "Could not create texture object: " );
}

void PopSift::uninit( )
{
    cudaError_t err;
    err = cudaDestroyTextureObject( _texture );
    POP_CUDA_FATAL_TEST( err, "Could not destroy texture object: " );

    _hst_input_image.freeHost( popart::CudaAllocated );
    _dev_input_image.freeDev( );

    delete _baseImg;
    delete _pyramid;
}

void PopSift::execute( imgStream inp )
{
    assert( inp.data_g == 0 );
    assert( inp.data_b == 0 );

    memcpy( _hst_input_image.data, inp.data_r, inp.width * inp.height );
    _hst_input_image.memcpyToDevice( _dev_input_image );
    _baseImg->upscale( _dev_input_image, _texture, 1<<up );

    _pyramid->build( _baseImg );

    _pyramid->find_extrema( _edgeLimit, _threshold );

    if( log_to_file ) {
        popart::write_plane2D( "upscaled-input-image.pgm",
                               true, // is stored on device
                               _baseImg->array );

        for( int o=0; o<_octaves; o++ ) {
            for( int s=0; s<_scales+3; s++ ) {
                _pyramid->download_and_save_array( "pyramid", o, s );
            }
        }
        for( int o=0; o<_octaves; o++ ) {
            _pyramid->download_and_save_descriptors( "pyramid", o );
        }
    }
}

