#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <assert.h>

#include "SIFT.h"
#include "debug_macros.h"

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
{
    _hst_input_image.freeHost( popart::CudaAllocated );
    _dev_input_image.freeDev( );
}

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

    POP_CUDA_STREAM_CREATE( &_stream );

    _initTime    = new popart::KeepTime( _stream );
    _uploadTime  = new popart::KeepTime( _stream );
    _pyramidTime = new popart::KeepTime( _stream );
    _extremaTime = new popart::KeepTime( _stream );

    float sigma = 1.0;

    _initTime->start();

    popart::Pyramid::init_filter( sigma, _scales, _stream );
    popart::Pyramid::init_sigma(  sigma, _scales, _stream );

    _initTime->stop();

    _baseImg = new popart::Image( _upscaled_width, _upscaled_height );
    _pyramid = new popart::Pyramid( _baseImg, _octaves, _scales, _stream );
}

void PopSift::uninit( )
{
    cudaStreamSynchronize( _stream );

    _hst_input_image.freeHost( popart::CudaAllocated );
    _dev_input_image.freeDev( );

    cudaStreamDestroy( _stream );

    _initTime   ->report( "Time to initialize:    " );
    _uploadTime ->report( "Time to upload:        " );
    _pyramidTime->report( "Time to build pyramid: " );
    _extremaTime->report( "Time to find extrema:  " );

    delete _initTime;
    delete _uploadTime;
    delete _pyramidTime;
    delete _extremaTime;

    delete _baseImg;
    delete _pyramid;
}

void PopSift::execute( imgStream inp )
{
    assert( inp.data_g == 0 );
    assert( inp.data_b == 0 );

    _uploadTime->start();
    memcpy( _hst_input_image.data, inp.data_r, inp.width * inp.height );
    _hst_input_image.memcpyToDevice( _dev_input_image, _stream );
    _baseImg->upscale( _dev_input_image, 1<<up, _stream );
    _uploadTime->stop();

    _pyramidTime->start();
    _pyramid->build( _baseImg );
    _pyramidTime->stop();

    _extremaTime->start();
    _pyramid->find_extrema( _edgeLimit, _threshold );
    _extremaTime->stop();

    if( log_to_file ) {
        _baseImg->download_and_save_array( "upscaled-input-image.pgm" );

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

