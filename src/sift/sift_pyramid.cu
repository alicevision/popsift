/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <sys/stat.h>

#include "sift_pyramid.h"
#include "sift_extremum.h"
#include "debug_macros.h"

#define PYRAMID_PRINT_DEBUG 0

using namespace std;

namespace popart {

__global__
void py_print_corner_float( float* img, uint32_t pitch, uint32_t height, uint32_t level )
{
    const int xbase = 0;
    const int ybase = level * height + 0;
    for( int i=0; i<10; i++ ) {
        for( int j=0; j<10; j++ ) {
            printf("%3.3f ", img[(ybase+i)*pitch+xbase+j] );
        }
        printf("\n");
    }
    printf("\n");
}

__global__
void py_print_corner_float_transposed( float* img, uint32_t pitch, uint32_t height, uint32_t level )
{
    const int xbase = 0;
    const int ybase = level * height + 0;
    for( int i=0; i<10; i++ ) {
        for( int j=0; j<10; j++ ) {
            printf("%3.3f ", img[(ybase+j)*pitch+xbase+i] );
        }
        printf("\n");
    }
    printf("\n");
}

void Pyramid::download_and_save_array( const char* basename, uint32_t octave, uint32_t level )
{
    if( octave < _num_octaves ) {
        _octaves[octave].download_and_save_array( basename, octave, level );
    } else {
        cerr << "Octave " << octave << " does not exist" << endl;
        return;
    }
}

void Pyramid::download_descriptors( const Config& conf, uint32_t octave )
{
    _octaves[octave].downloadDescriptor( conf );
}

void Pyramid::save_descriptors( const Config& conf, const char* basename, uint32_t octave )
{
    struct stat st = {0};
    if (stat("dir-desc", &st) == -1) {
        mkdir("dir-desc", 0700);
    }
    ostringstream ostr;
    ostr << "dir-desc/desc-" << basename << "-o-" << octave << ".txt";
    ofstream of( ostr.str().c_str() );
    _octaves[octave].writeDescriptor( conf, of, true );

    if (stat("dir-fpt", &st) == -1) {
        mkdir("dir-fpt", 0700);
    }
    ostringstream ostr2;
    ostr2 << "dir-fpt/desc-" << basename << "-o-" << octave << ".txt";
    ofstream of2( ostr2.str().c_str() );
    _octaves[octave].writeDescriptor( conf, of2, false );
}

Pyramid::Pyramid( Config& config,
                  Image* base,
                  int width,
                  int height )
    : _num_octaves( config.octaves )
    , _levels( config.levels + 3 )
    , _scaling_mode( config.scaling_mode )
    , _gauss_group( config.gauss_group_size )
    , _assume_initial_blur( config.hasInitialBlur() )
    , _initial_blur( config.getInitialBlur() )
{
    // cerr << "Entering " << __FUNCTION__ << endl;

    _octaves = new Octave[_num_octaves];

    int w = width;
    int h = height;

    cout << "Size of the first octave's images: " << w << "X" << h << endl;

    for( int o=0; o<_num_octaves; o++ ) {
        _octaves[o].debugSetOctave( o );
        _octaves[o].alloc( w, h, _levels, _gauss_group );
        w = ceilf( w / 2.0f );
        h = ceilf( h / 2.0f );
    }
}

Pyramid::~Pyramid( )
{
    delete [] _octaves;
}

void Pyramid::find_extrema( const Config&                conf,
                            Image*                       base,
                            vector<vector<Extremum> >*   extrema,
                            vector<vector<Descriptor> >* descs )
{
    reset_extrema_mgmt( );

    build_pyramid( conf, base );

    find_extrema( conf );

    orientation( conf );

    descriptors( conf );

    if( extrema && descs ) {
        extrema->resize( _num_octaves * _levels );
        descs  ->resize( _num_octaves * _levels );
        for( int o=0; o<_num_octaves; o++ ) {
            for( int l=0; l<_levels; l++ ) {
                _octaves[o].downloadToVector( l, (*extrema)[o*_levels+l], (*descs)[o*_levels+l] );
            }
        }
    }
}

void Pyramid::reset_extrema_mgmt( )
{
    for( int o=0; o<_num_octaves; o++ ) {
        _octaves[o].reset_extrema_mgmt( );
    }
}

} // namespace popart

