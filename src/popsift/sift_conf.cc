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

using namespace std;

namespace popsift
{

Config::Config( )
    : _upscale_factor( 1.0f )
    , octaves( -1 )
    , levels( 3 )
    , sigma( 1.6f )
    , _edge_limit( 10.0f )
    , _threshold( 0.04 ) // ( 10.0f / 256.0f )
    , _sift_mode( Config::RefineInOctave )
    , _log_mode( Config::None )
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
}

void Config::setMode( Config::SiftMode m )
{
    _sift_mode = m;
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
        POP_FATAL( string("filter sorting mode must be one of up, down or random") );
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
    if( m == "RootSift" ) setNormMode( Config::RootSift );
    else if( m == "classic" ) setNormMode( Config::Classic );
    else
        POP_FATAL( string("Bad Normalization mode.\n") );
}

Config::NormMode Config::getNormModeDefault( )
{
    return Config::RootSift;
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

void Config::setDownsampling( float v ) { _upscale_factor = -v; }
void Config::setOctaves( int v ) { octaves = v; }
void Config::setLevels( int v ) { levels = v; }
void Config::setSigma( float v ) { sigma = v; }
void Config::setEdgeLimit( float v ) { _edge_limit = v; }
void Config::setThreshold( float v ) { _threshold = v; }
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

Config::SiftMode Config::getSiftMode() const
{
    return _sift_mode;
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

bool Config::equal( const Config& other ) const
{
    #define COMPARE(a) ( this->a != other.a )
    if( COMPARE( octaves ) ||
        COMPARE( levels ) ||
        COMPARE( sigma ) ||
        COMPARE( _edge_limit ) ||
        COMPARE( _threshold ) ||
        COMPARE( _upscale_factor ) ||
        COMPARE( _max_extrema ) ||
        COMPARE( _sift_mode ) ||
        COMPARE( _assume_initial_blur ) ||
        COMPARE( _initial_blur ) ||
        COMPARE( _normalization_mode ) ||
        COMPARE( _normalization_multiplier ) ) return false;
    return true;
}

}; // namespace popsift

