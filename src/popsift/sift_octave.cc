/*
* Copyright 2016, Simula Research Laboratory
*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "common/clamp.h"
#include "common/debug_macros.h"
#include "common/write_plane_2d.h"
#include "sift_constants.h"
#include "sift_octave.h"
#include "sift_pyramid.h"

#include <sys/stat.h>

#include <new> // for placement new
#include <sstream>
#ifdef _WIN32
#include <direct.h>
#define stat _stat
#define mkdir(name, mode) _mkdir(name)
#endif

using namespace std;

namespace popsift {

Octave::Octave()
{ }

void Octave::alloc( const Config& conf, int width, int height, int levels, int gauss_group )
{
    _max_w = _w = width;
    _max_h = _h = height;
    _levels = levels;

    _w_grid_divider = float(_w) / conf.getFilterGridSize();
    _h_grid_divider = float(_h) / conf.getFilterGridSize();

    alloc_data_planes();
    alloc_interm_array();
    alloc_dog_array();
}

void Octave::resetDimensions( const Config& conf, int w, int h )
{
    if( w == _w && h == _h ) {
        return;
    }

    _w = w;
    _h = h;

    _w_grid_divider = float(_w) / conf.getFilterGridSize();
    _h_grid_divider = float(_h) / conf.getFilterGridSize();

    if( _w > _max_w || _h > _max_h ) {
        _max_w = max( _w, _max_w );
        _max_h = max( _h, _max_h );
    }

    free_data_planes();
    free_interm_array();
    free_dog_array();

    alloc_data_planes();
    alloc_interm_array();
    alloc_dog_array();
}

void Octave::free()
{
    free_dog_array();
    free_interm_array();
    free_data_planes();
}

/*************************************************************
 * Debug output: write an octave/level to disk as PGM
 *************************************************************/

void Octave::download_and_save_array( const char* basename, int octave )
{
    struct stat st = { 0 };

    int width  = getWidth();
    int height = getHeight();

    if (stat("dir-octave", &st) == -1) {
        mkdir("dir-octave", 0700);
    }

    if (stat("dir-octave-dump", &st) == -1) {
        mkdir("dir-octave-dump", 0700);
    }

    if (stat("dir-dog", &st) == -1) {
        mkdir("dir-dog", 0700);
    }

    if (stat("dir-dog-txt", &st) == -1) {
        mkdir("dir-dog-txt", 0700);
    }

    if (stat("dir-dog-dump", &st) == -1) {
        mkdir("dir-dog-dump", 0700);
    }

    for( int l = 0; l<_levels; l++ ) {
        Plane2D_float p(width, height, &_data[l*width*height], width * sizeof(float));

        ostringstream ostr;
        ostr << "dir-octave/" << basename << "-o-" << octave << "-l-" << l << ".pgm";
        popsift::write_plane2Dunscaled( ostr.str().c_str(), false, p );

        ostringstream ostr2;
        ostr2 << "dir-octave-dump/" << basename << "-o-" << octave << "-l-" << l << ".dump";
        popsift::dump_plane2Dfloat(ostr2.str().c_str(), false, p );
    }

    for (int l = 0; l<_levels - 1; l++) {
        Plane2D_float p(width, height, &_dog_3d[l*width*height], width * sizeof(float));

        ostringstream ostr;
        ostr << "dir-dog/d-" << basename << "-o-" << octave << "-l-" << l << ".pgm";
        popsift::write_plane2D(ostr.str().c_str(), false, p);

        ostringstream pstr;
        pstr << "dir-dog-txt/d-" << basename << "-o-" << octave << "-l-" << l << ".txt";
        popsift::write_plane2Dunscaled(pstr.str().c_str(), false, p, 127);

        ostringstream qstr;
        qstr << "dir-dog-dump/d-" << basename << "-o-" << octave << "-l-" << l << ".dump";
        popsift::dump_plane2Dfloat(qstr.str().c_str(), false, p);
    }
}

void Octave::alloc_data_planes()
{
    _data_ext.width  = _w; // for cudaMalloc3DArray, width in elements
    _data_ext.height = _h;
    _data_ext.depth  = _levels;

    _data = new float[_levels * _h * _w];
}

void Octave::free_data_planes()
{
    delete [] _data;
}

void Octave::alloc_interm_array()
{
    _intm_ext.width  = _w;
    _intm_ext.height = _h;
    _intm_ext.depth  = _levels;

    _intm = new float[_levels * _h * _w];
}

void Octave::free_interm_array()
{
    delete [] _intm;
}

void Octave::alloc_dog_array()
{
    _dog_3d_ext.width = _w; // for cudaMalloc3DArray, width in elements
    _dog_3d_ext.height = _h;
    _dog_3d_ext.depth = _levels - 1;

    _dog_3d = new float[ (_levels-1) * _h * _w ];
}

void Octave::free_dog_array()
{
    delete [] _dog_3d;
}

} // namespace popsift
