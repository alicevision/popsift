#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <assert.h>

#include "SIFT.hpp"
#include "debug_macros.hpp"
#include "keep_time.hpp"

using namespace std;

PopSift::PopSift(int O_,
               int S_,
               int up_,
               float threshold,
               float edgeThreshold,
               float sigma )
    : _octaves( O_ )
    , _scales( max(2,S_) ) // min is 2, GPU restriction */
    , up( up_ )
    , _sigma( sigma )
{
    /* SIFT parameters */
    _threshold = threshold;
    _edgeLimit = edgeThreshold;
}

PopSift::~PopSift()
{
}

#define TRY_IMAGE_TWICE 0

void PopSift::execute( imgStream inp )
{
    _inp = inp;

    if (_octaves < 0)
        _octaves = max(int (floor( logf( (float)min(_inp.width, _inp.height) )
                                   / logf( 2.0f ) ) - 3 + up), 1);

    cudaError_t err;

    /* setup CUDA device */
    err = cudaSetDevice( choose );
    POP_CUDA_FATAL_TEST( err, "Failed to set CUDA device" );

#if (TRY_IMAGE_TWICE==1)
    const bool secondImage = true;
#else
    const bool secondImage = false;
#endif // (TRY_IMAGE_TWICE==1)

    _upscaled_width  = _inp.width  << up;
    _upscaled_height = _inp.height << up;

    cudaEvent_t  event_0;
    cudaStream_t stream_0;
    cudaStream_t stream_1;
    cudaStream_t stream_2;
    POP_CUDA_STREAM_CREATE( &stream_0 );
    POP_CUDA_STREAM_CREATE( &stream_1 );
    POP_CUDA_STREAM_CREATE( &stream_2 );

    err = cudaEventCreate( &event_0 );

    popart::Image  input1( _inp, stream_1 );
    popart::Image* input2;
    if( secondImage ) {
        input2 = new popart::Image( _inp, stream_2 );
    }

    popart::KeepTime globalKeepTime( stream_1 );
    globalKeepTime.start();

    float sigma = 1.0;

    _base1 = new popart::Image( _upscaled_width, _upscaled_height, sizeof(float), stream_1 );
    _pyramid1 = new popart::Pyramid( _base1, _octaves, _scales, stream_1 );
    _base1->upscale( input1, 1<<up );

    if( secondImage ) {
        _base2 = new popart::Image( _upscaled_width, _upscaled_height, sizeof(float), stream_2 );
        _pyramid2 = new popart::Pyramid( _base2, _octaves, _scales, stream_2 );
        _base2->upscale( *input2, 1<<up );
    }

    popart::Pyramid::init_filter( sigma, _scales, stream_0 );
    popart::Pyramid::init_sigma(  sigma, _scales, stream_0 );
    cudaEventRecord( event_0, stream_0 );

    cudaStreamWaitEvent( stream_1, event_0, 0 );
    _pyramid1->build( _base1, 0 );

    _pyramid1->find_extrema( _edgeLimit, _threshold );

    if( secondImage ) {
        cudaStreamWaitEvent( stream_2, event_0, 0 );
        _pyramid2->build( _base2, 0 );
        _pyramid2->find_extrema( _edgeLimit, _threshold );
        globalKeepTime.waitFor( stream_2 );
    }

#if 1
    cerr << "stopping global timer" << endl;
    globalKeepTime.stop( );
    cerr << "stopped global timer" << endl;
    globalKeepTime.report( "Combined overall time: " );

    _base1->report_times();
    if( secondImage )
        _base2->report_times();
    _pyramid1->report_times();
    if( secondImage )
        _pyramid2->report_times();
#endif

    if( log_to_file ) {
        _base1->download_and_save_array( "upscaled-input-image.pgm" );
        if( secondImage ) {
            _base2->download_and_save_array( "upscaled-input-image2.pgm" );
        }

        for( int o=0; o<_octaves; o++ ) {
            for( int s=0; s<_scales+3; s++ ) {
                _pyramid1->download_and_save_array( "pyramid", o, s );
                if( secondImage ) {
                    _pyramid2->download_and_save_array( "pyramid2", o, s );
                }
            }
        }
        for( int o=0; o<_octaves; o++ ) {
            _pyramid1->download_and_save_descriptors( "pyramid", o );
            if( secondImage ) {
                _pyramid2->download_and_save_descriptors( "pyramid2", o );
            }
        }
    }

    cudaEventDestroy( event_0 );
    cudaStreamDestroy( stream_0 );
    cudaStreamDestroy( stream_1 );
    if( secondImage ) {
        cudaStreamDestroy( stream_2 );
    }

printf("Everything OK until here, quitting\n");
}

