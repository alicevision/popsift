#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <sys/stat.h>

#include "sift_pyramid.h"
#include "debug_macros.h"

#define PYRAMID_PRINT_DEBUG 0

using namespace std;

namespace popart {

/*************************************************************
 * CUDA device functions for printing debug information
 *************************************************************/

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

/*************************************************************
 * Callers for CUDA device functions that print debug information
 *************************************************************/

void Pyramid::debug_out_floats( float* data, uint32_t pitch, uint32_t height )
{
    py_print_corner_float
        <<<1,1>>>
        ( data,
          pitch,
          height,
          0 );

    test_last_error( __LINE__ );
}

void Pyramid::debug_out_floats_t( float* data, uint32_t pitch, uint32_t height )
{
    py_print_corner_float_transposed
        <<<1,1>>>
        ( data,
          pitch,
          height,
          0 );

    test_last_error( __LINE__ );
}

/*************************************************************
 * Host-sided debug function
 *************************************************************/

void Pyramid::test_last_error( int line )
{
    cudaError_t err;
    cudaDeviceSynchronize( );
    err = cudaGetLastError();
    if( err != cudaSuccess ) {
        printf("A problem in line %d, %s\n", line, cudaGetErrorString(err) );
        exit( -__LINE__ );
    }
}

/*************************************************************
 * Debug output: write an octave/level to disk as PGM
 *************************************************************/

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

/*************************************************************
 * Pyramid constructor
 *************************************************************/

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
    , _bemap_orientation_mode( config.getBemapOrientation() )
{
    // cerr << "Entering " << __FUNCTION__ << endl;

    _octaves = new Octave[_num_octaves];

    int w = width;
    int h = height;

    cout << "Size of the first octave's images: " << w << "X" << h << endl;

    for( int o=0; o<_num_octaves; o++ ) {
#if (PYRAMID_PRINT_DEBUG==1)
        printf("Allocating octave %u with width %u and height %u (%u levels)\n", o, w, h, _levels );
#endif // (PYRAMID_PRINT_DEBUG==1)
        _octaves[o].debugSetOctave( o );
        _octaves[o].alloc( w, h, _levels, _gauss_group );
        w = ceilf( w / 2.0f );
        h = ceilf( h / 2.0f );
    }
}

/*************************************************************
 * Pyramid destructor
 *************************************************************/

Pyramid::~Pyramid( )
{
    delete [] _octaves;
}

/*************************************************************
 * Build the pyramid in all levels, one octave
 *************************************************************/

void Pyramid::find_extrema( const Config& conf, Image* base )
{
    reset_extrema_mgmt( );

    build_v11( conf, base );

    find_extrema_v6( conf );

    orientation_v1();

    descriptors_v1( );
}

void Pyramid::reset_extrema_mgmt( )
{
    for( int o=0; o<_num_octaves; o++ ) {
        _octaves[o].reset_extrema_mgmt( );
    }
}

} // namespace popart

