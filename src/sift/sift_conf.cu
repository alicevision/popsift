#include "sift_conf.h"
#include "debug_macros.h"

namespace popart
{

Config::Config( )
    : _upscale_factor( 1.0f )
    , octaves( -1 )
    , levels( 3 )
    , sigma( 1.6f )
    , _edge_limit( 10.0f )
    , _threshold( 0.04 ) // ( 10.0f / 256.0f )
    , _sift_mode( Config::PopSift )
    , log_mode( Config::None )
    , scaling_mode( Config::ScaleDefault )
    , verbose( false )
    , gauss_group_size( 1 )
    , _assume_initial_blur( false )
    , _initial_blur( 0.0f )
    , _print_gauss_tables( false )
    , _dp_orientation( true )
    , _dp_descriptors( true )
    , _dp_capable( true )
{
    int            currentDev;
    cudaDeviceProp currentProp;
    cudaError_t    err;

    err = cudaGetDevice( &currentDev );
    POP_CUDA_FATAL_TEST( err, "Could not get current device ID" );

    err = cudaGetDeviceProperties( &currentProp, currentDev );
    POP_CUDA_FATAL_TEST( err, "Could not get current device properties" );

    if( currentProp.major < 3 || ( currentProp.major == 3 && currentProp.minor < 5 ) ) {
        _dp_capable = false;
    }
}

void Config::setMode( Config::SiftMode m )
{
    _sift_mode = m;
}

void Config::setVerbose( bool on )
{
    verbose = on;
}

void Config::setLogMode( LogMode mode )
{
    log_mode = mode;
}

void Config::setScalingMode( ScalingMode mode )
{
    scaling_mode = mode;
}

void Config::setDownsampling( float v ) { _upscale_factor = -v; }
void Config::setOctaves( int v ) { octaves = v; }
void Config::setLevels( int v ) { levels = v; }
void Config::setSigma( float v ) { sigma = v; }
void Config::setEdgeLimit( float v ) { _edge_limit = v; }
void Config::setThreshold( float v ) { _threshold = v; }
void Config::setPrintGaussTables() { _print_gauss_tables = true; }
void Config::setDPOrientation( bool onoff ) { _dp_orientation = onoff; }
void Config::setDPDescriptors( bool onoff ) { _dp_descriptors = onoff; }


void Config::setInitialBlur( float blur )
{
    _assume_initial_blur = true;
    _initial_blur        = blur;
}

void Config::setGaussGroup( int groupsize )
{
    gauss_group_size = groupsize;
}

int  Config::getGaussGroup( ) const
{
    return gauss_group_size;
}

bool Config::hasInitialBlur( ) const
{
    return _assume_initial_blur;
}

float Config::getInitialBlur( ) const
{
    return _initial_blur;
}

float Config::getPeakThreshold() const
{
    return ( _threshold * 0.5f * 255.0f / levels );
}

bool Config::ifPrintGaussTables() const
{
    return _print_gauss_tables;
}

Config::SiftMode Config::getSiftMode() const
{
    return _sift_mode;
}


}; // namespace popart

