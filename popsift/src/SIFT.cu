#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <assert.h>

#include "SIFT.h"
#include "s_sigma.h"
#include "gauss_filter.h"
#include "debug_macros.h"
#include "write_plane_2d.h"

using namespace std;

PopSift::PopSift( popart::Config config )
    : _octaves( config.octaves )
    , _levels( max( 2, config.levels ) ) // min is 2, GPU restriction */
    , up( config.start_sampling )
    , _sigma( config.sigma )
    , _threshold( config.threshold ) // SIFT parameter
    , _edgeLimit( config.edge_limit ) // SIFT parameter
    , _vlfeat_mode( config.sift_mode == popart::Config::VLFeat )
    , _log_to_file( config.log_mode == popart::Config::All )
    , _verbose( config.verbose )
{
}

PopSift::~PopSift()
{ }

void PopSift::baseInit( )
{
    popart::init_filter( _sigma, _levels, _vlfeat_mode );
    popart::init_sigma(  _sigma, _levels );
}

void PopSift::init( int w, int h )
{
    if (_octaves < 0) {
        _octaves = max(int (floor( logf( (float)min( w, h ) )
                                   / logf( 2.0f ) ) - 3 + 1.0/pow(2.0f,up) ), 1);
    }

    // _upscaled_width  = w << up;
    // _upscaled_height = h << up;
    _upscaled_width  = w / pow( 2.0f, up );
    _upscaled_height = h / pow( 2.0f, up );
    cerr << "Upscaling WxH = " << _upscaled_width << "x" << _upscaled_height << endl;

    _hst_input_image.allocHost( w, h, popart::CudaAllocated );
    _dev_input_image.allocDev( w, h );

    _baseImg = new popart::Image( _upscaled_width, _upscaled_height );
    _pyramid = new popart::Pyramid( _baseImg, _octaves, _levels );

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
    _baseImg->upscale( _dev_input_image, _texture, 1.0 / pow( 2.0f, up ) );

    _pyramid->build( _baseImg );

    _pyramid->find_extrema( _edgeLimit, _threshold );

    if( _log_to_file ) {
        popart::write_plane2D( "upscaled-input-image.pgm",
                               true, // is stored on device
                               _baseImg->array );

        for( int o=0; o<_octaves; o++ ) {
            for( int s=0; s<_levels+3; s++ ) {
                _pyramid->download_and_save_array( "pyramid", o, s );
            }
        }
        for( int o=0; o<_octaves; o++ ) {
            _pyramid->download_and_save_descriptors( "pyramid", o );
        }
    }
}

