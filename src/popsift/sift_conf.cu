/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "sift_conf.h"
#include "common/debug_macros.h"

namespace popsift
{

Config::Config( )
    : _upscale_factor( 1.0f )
    , octaves( -1 )
    , levels( 3 )
    , sigma( 1.6f )
    , _edge_limit( 10.0f )
    , _threshold( 0.04 ) // ( 10.0f / 256.0f )
    , _gauss_mode( Config::VLFeat_Compute )
    , _sift_mode( Config::PopSift )
    , _log_mode( Config::None )
    , scaling_mode( Config::ScaleDefault )
    , verbose( false )
    , gauss_group_size( 1 )
    , _max_extrema( 10000 )
    , _assume_initial_blur( true )
    , _initial_blur( 0.5f )
    , _use_root_sift( false )
    , _normalization_multiplier( 0 )
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

void Config::setGaussMode( Config::GaussMode m )
{
    _gauss_mode = m;
}

void Config::setGaussMode( const std::string& m )
{
    if( m == "vlfeat" )
        setGaussMode( Config::VLFeat_Compute );
    else if( m == "opencv" )
        setGaussMode( Config::OpenCV_Compute );
    else if( m == "fixed4" )
        setGaussMode( Config::Fixed4 );
    else if( m == "fixed8" )
        setGaussMode( Config::Fixed8 );
    else
        POP_FATAL( "specified Gauss mode must be one of vlfeat, opencv, fixed4 or fixed8" );
}

void Config::setVerbose( bool on )
{
    verbose = on;
}

void Config::setLogMode( LogMode mode )
{
    _log_mode = mode;
}

Config::LogMode Config::getLogMode( ) const
{
    return _log_mode;
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
void Config::setUseRootSift( bool on ) { _use_root_sift = on; }
void Config::setNormalizationMultiplier( int mul ) { _normalization_multiplier = mul; }

void Config::setInitialBlur( float blur )
{
    if( blur == 0.0f ) {
        _assume_initial_blur = false;
        _initial_blur        = blur;
    } else {
        _assume_initial_blur = true;
        _initial_blur        = blur;
    }
}

Config::GaussMode Config::getGaussMode( ) const
{
    return _gauss_mode;
}

Config::SiftMode Config::getSiftMode() const
{
    return _sift_mode;
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


}; // namespace popsift

