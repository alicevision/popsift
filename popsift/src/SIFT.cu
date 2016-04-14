#include "SIFT.h"
#include "s_sigma.h"
#include "gauss_filter.h"
#include "write_plane_2d.h"

using namespace std;

PopSift::PopSift( popart::Config config )
    : _config( config )
{
    _config.levels = max( 2, config.levels );

    const bool vlfeat_mode = ( config.sift_mode == popart::Config::VLFeat );
    popart::init_filter( _config.sigma, _config.levels, vlfeat_mode );
    popart::init_sigma(  _config.sigma, _config.levels );
}

PopSift::~PopSift()
{ }

bool PopSift::init( int pipe, int w, int h )
{
    if( pipe < 0 && pipe >= MAX_PIPES ) {
        return false;
    }

    int octaves = _config.octaves;

    /* up=-1 -> scale factor=2
     * up= 0 -> scale factor=1
     * up= 1 -> scale factor=0.5
     */
    float scaleFactor = 1.0 / pow( 2.0, _config.start_sampling );

    if( octaves < 0 ) {
        octaves = max(int (floor( logf( (float)min( w, h ) )
                                / logf( 2.0f ) ) - 3.0f + scaleFactor ), 1);
    }

    _pipe[pipe]._inputImage = new popart::Image( w, h );
    _pipe[pipe]._pyramid = new popart::Pyramid( _pipe[pipe]._inputImage,
                                                octaves,
                                                _config.levels,
                                                ceilf( w * scaleFactor ),
                                                ceilf( h * scaleFactor ),
                                                _config.scaling_mode );

    return true;
}

void PopSift::uninit( int pipe )
{
    if( pipe < 0 && pipe >= MAX_PIPES ) return;

    delete _pipe[pipe]._inputImage;
    delete _pipe[pipe]._pyramid;
}

void PopSift::execute( int pipe, imgStream inp )
{
    if( pipe < 0 && pipe >= MAX_PIPES ) return;

    assert( inp.data_g == 0 );
    assert( inp.data_b == 0 );

    _pipe[pipe]._inputImage->load( inp );

    _pipe[pipe]._pyramid->build( _pipe[pipe]._inputImage );

    _pipe[pipe]._pyramid->find_extrema( _config.edge_limit, _config.threshold );

    bool log_to_file = ( _config.log_mode == popart::Config::All );
    if( log_to_file ) {
        popart::write_plane2D( "upscaled-input-image.pgm",
                               true, // is stored on device
                               _pipe[pipe]._inputImage->getUpscaledImage() );

        int octaves = _pipe[pipe]._pyramid->getNumOctaves();
        int levels  = _pipe[pipe]._pyramid->getNumLevels();

        for( int o=0; o<octaves; o++ ) {
            for( int s=0; s<levels+3; s++ ) {
                _pipe[pipe]._pyramid->download_and_save_array( "pyramid", o, s );
            }
        }
        for( int o=0; o<octaves; o++ ) {
            _pipe[pipe]._pyramid->download_and_save_descriptors( "pyramid", o );
        }
    }
}

