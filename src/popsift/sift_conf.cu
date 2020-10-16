/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/debug_macros.h"
#include "sift_conf.h"

#include <iostream>
#include <algorithm>

using namespace std;

static bool stringIsame( string l, string r )
{
    std::for_each( l.begin(), l.end(), [](char& c) { c = ::tolower(c); });
    std::for_each( r.begin(), r.end(), [](char& c) { c = ::tolower(c); });
    return l == r;
}

namespace popsift
{

Config::Config( )
    : _upscale_factor( 1.0f )
    , octaves( -1 )
    , levels( 3 )
    , sigma( 1.6f )
    , _edge_limit( 10.0f )
    , _threshold( 0.04 ) // ( 10.0f / 256.0f )
    , _gauss_mode( getGaussModeDefault() )
    , _sift_mode( Config::PopSift )
    , _log_mode( Config::None )
    , _scaling_mode( Config::ScaleDefault )
    , _desc_mode( Config::Loop )
    , _grid_filter_mode( Config::RandomScale )
    , verbose( false )
    // , _max_extrema( 20000 )
    , _max_extrema( 100000 )
    , _filter_max_extrema( -1 )
    , _filter_grid_size( 2 )
    , _assume_initial_blur( true )
    , _initial_blur( 0.5f )
    , _normalization_mode( getNormModeDefault() )
    , _normalization_multiplier( 0 )
    , _print_gauss_tables( false )
{
    int            currentDev;
    cudaDeviceProp currentProp;
    cudaError_t    err;

    err = cudaGetDevice( &currentDev );
    POP_CUDA_FATAL_TEST( err, "Could not get current device ID" );

    err = cudaGetDeviceProperties( &currentProp, currentDev );
    POP_CUDA_FATAL_TEST( err, "Could not get current device properties" );
}

void Config::setMode( Config::SiftMode m )
{
    _sift_mode = m;
}

void Config::setGaussMode( Config::GaussMode m )
{
    _gauss_mode = m;
}

void Config::setDescMode( const std::string& text )
{
    if( text == "loop" )
        setDescMode( Config::Loop );
    else if( text == "iloop" )
        setDescMode( Config::ILoop );
    else if( text == "grid" )
        setDescMode( Config::Grid );
    else if( text == "igrid" )
        setDescMode( Config::IGrid );
    else if( text == "notile" )
        setDescMode( Config::NoTile );
    else if( text == "vlfeat" )
        setDescMode( Config::VLFeat_Desc );
    else
        POP_FATAL( "specified descriptor extraction mode must be one of loop, grid or igrid" );
}

void Config::setDescMode( Config::DescMode m )
{
    _desc_mode = m;
}

const char* Config::getDescModeUsage( )
{
    return "Choice of descriptor extraction modes:\n"
           "loop, iloop, grid, igrid, notile, vlfeat\n"
	       "Default is loop\n"
           "loop is OpenCV-like horizontal scanning, sampling every pixel in a radius around the "
           "centers or the 16 tiles arond the keypoint. Each sampled point contributes to two "
           "histogram bins."
           "iloop is like loop but samples all constant 1-pixel distances from the keypoint, "
           "using the CUDA texture engine for interpolation. "
           "grid is like loop but works on rotated, normalized tiles, relying on CUDA 2D cache "
           "to replace the manual data aligment idea of loop. "
           "igrid iloop and grid. "
           "notile is like igrid but handles all 16 tiles at once.\n"
           "vlfeat is VLFeat-like horizontal scanning, sampling every pixel in a radius around "
           "keypoint itself, using the 16 tile centers only for weighting. Every sampled point "
           "contributes to up to eight historgram bins.";
}

void Config::setGaussMode( const std::string& m )
{
    if( m == "vlfeat" )
        setGaussMode( Config::VLFeat_Compute );
    else if( m == "vlfeat-hw-interpolated" )
        setGaussMode( Config::VLFeat_Relative );
    else if( m == "relative" )
        setGaussMode( Config::VLFeat_Relative );
    else if( m == "vlfeat-direct" )
        setGaussMode( Config::VLFeat_Relative_All );
    else if( m == "opencv" )
        setGaussMode( Config::OpenCV_Compute );
    else if( m == "fixed9" )
        setGaussMode( Config::Fixed9 );
    else if( m == "fixed15" )
        setGaussMode( Config::Fixed15 );
    else
        POP_FATAL( string("Bad Gauss mode.\n") + getGaussModeUsage() );
}

Config::GaussMode Config::getGaussModeDefault( )
{
    return Config::VLFeat_Compute;
}

const char* Config::getGaussModeUsage( )
{
    return
        "Choice of Gauss filter method. "
        "Options are: "
        "vlfeat (default), "
        "vlfeat-hw-interpolated, "
        "vlfeat-direct, "
        "opencv, "
        "fixed9, "
        "fixed15, "
        "relative (synonym for vlfeat-hw-interpolated)";
}

bool Config::getCanFilterExtrema() const
{
#if __CUDACC_VER_MAJOR__ >= 8
    return true;
#else
    return false;
#endif
}

void Config::setFilterSorting( const std::string& text )
{
    if( text == "up" )
        _grid_filter_mode = Config::SmallestScaleFirst;
    else if( text == "down" )
        _grid_filter_mode = Config::LargestScaleFirst;
    else if( text == "random" )
        _grid_filter_mode = Config::RandomScale;
    else
        POP_FATAL( "filter sorting mode must be one of up, down or random" );
}

void Config::setFilterSorting( Config::GridFilterMode m )
{
    _grid_filter_mode = m;
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
    _scaling_mode = mode;
}

/**
 * Normalization mode
 * Should the descriptor normalization use L2-like classic normalization
 * of the typically better L1-like RootSift normalization?
 */
void Config::setUseRootSift( bool on )
{
    if( on )
        _normalization_mode = RootSift;
    else
        _normalization_mode = Classic;
}

bool Config::getUseRootSift( ) const
{
    return ( _normalization_mode == RootSift );
}

Config::NormMode Config::getNormMode( NormMode m ) const 
{
    return _normalization_mode;
}

void Config::setNormMode( Config::NormMode m )
{
    _normalization_mode = m;
}

void Config::setNormMode( const std::string& m )
{
    if( stringIsame( m, "RootSift" ) )
    {
        setNormMode( Config::RootSift );
    }
    else if( stringIsame( m, "L2" ) )
    {
        setNormMode( Config::Classic );
    }
    else if( stringIsame( m, "Classic" ) )
    {
        setNormMode( Config::Classic );
    }
    else
        POP_FATAL( string("Bad Normalization mode.\n") + getGaussModeUsage() );
}

Config::NormMode Config::getNormModeDefault( )
{
    return Config::NormDefault;
}

const char* Config::getNormModeUsage( )
{
    return
        "Choice of descriptor normalization modes. "
        "Options are: "
        "RootSift (L1-like, default), "
        "Classic (L2-like)";
}

/**
 * Normalization multiplier
 * A power of 2 multiplied with the normalized descriptor. Required
 * for the construction of 1-byte integer desciptors.
 * Usual choice is 2^8 or 2^9.
 */
void Config::setNormalizationMultiplier( int mul )
{
    _normalization_multiplier = mul;
}

int Config::getNormalizationMultiplier( ) const
{
    return _normalization_multiplier;
}

void  Config::setDownsampling( float v ) { _upscale_factor = -v; }
float Config::getDownsampling( ) const   { return -_upscale_factor; }

void Config::setOctaves( int v ) { octaves = v; }
int  Config::getOctaves( ) const { return octaves; }

void Config::setLevels( int v ) { levels = v; }
int  Config::getLevels( ) const { return levels; }

void  Config::setSigma( float v ) { sigma = v; }
float Config::getSigma( ) const { return sigma; }

void  Config::setEdgeLimit( float v ) { _edge_limit = v; }
float Config::getEdgeLimit( ) const   { return _edge_limit; }

void  Config::setThreshold( float v ) { _threshold = v; }
float Config::getThreshold( ) const   { return _threshold; }

void Config::setPrintGaussTables() { _print_gauss_tables = true; }
void Config::setFilterMaxExtrema( int ext ) { _filter_max_extrema = ext; }
void Config::setFilterGridSize( int sz ) { _filter_grid_size = sz; }

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
bool Config::hasInitialBlur( ) const
{
    return _assume_initial_blur;
}
float Config::getInitialBlur( ) const
{
    return _initial_blur;
}


Config::GaussMode Config::getGaussMode( ) const
{
    return _gauss_mode;
}

Config::SiftMode Config::getSiftMode() const
{
    return _sift_mode;
}

float Config::getPeakThreshold() const
{
    return ( _threshold * 0.5f * 255.0f / levels );
}

bool Config::ifPrintGaussTables() const
{
    return _print_gauss_tables;
}

bool Config::equal( const Config& other ) const
{
    #define COMPARE(a) ( this->a != other.a )
    if( COMPARE( octaves ) ||
        COMPARE( levels ) ||
        COMPARE( sigma ) ||
        COMPARE( _edge_limit ) ||
        COMPARE( _threshold ) ||
        COMPARE( _upscale_factor ) ||
        COMPARE( _scaling_mode ) ||
        COMPARE( _max_extrema ) ||
        COMPARE( _gauss_mode ) ||
        COMPARE( _sift_mode ) ||
        COMPARE( _assume_initial_blur ) ||
        COMPARE( _initial_blur ) ||
        COMPARE( _normalization_mode ) ||
        COMPARE( _normalization_multiplier ) ) return false;
    return true;
}

}; // namespace popsift
