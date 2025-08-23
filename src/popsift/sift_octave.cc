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

void Octave::alloc( const Config& conf, int width, int height, int levels )
{
    _w = width;
    _h = height;
    _levels = levels;

    _w_grid_divider = float(_w) / conf.getFilterGridSize();
    _h_grid_divider = float(_h) / conf.getFilterGridSize();

    _data  .alloc( width, height, levels );
    _intm  .alloc( width, height, levels );
    _dog_3d.alloc( width, height, levels-1 );
}

void Octave::resetDimensions( const Config& conf, int width, int height )
{
    if( width == _w && height == _h ) {
        return;
    }

    _w = width;
    _h = height;

    _w_grid_divider = float(_w) / conf.getFilterGridSize();
    _h_grid_divider = float(_h) / conf.getFilterGridSize();

    _data  .resetDimensions( width, height, _levels );
    _intm  .resetDimensions( width, height, _levels );
    _dog_3d.resetDimensions( width, height, _levels-1 );
}

// void Octave::free()
// {
//     _data  .dealloc();
//     _intm  .dealloc();
//     _dog_3d.dealloc();
// }


void Octave::free()
{
    _data = PlaneD<float>();    
    _intm = PlaneD<float>();
    _dog_3d = PlaneD<float>();
    
}

/*************************************************************
 * Debug output: write an octave/level to disk as PGM
 *************************************************************/

void Octave::download_and_save_array( const Config& conf, const char* basename, int octave )
{
    POP_INFO2( conf.silent(), " enter " << __PRETTY_FUNCTION__ );

    struct stat st = { 0 };

    // int width  = getWidth();
    // int height = getHeight();

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
        Plane2D_float p;
        p.copyFromPlane( _data, l );

        ostringstream ostr;
        ostr << "dir-octave/" << basename << "-o-" << octave << "-l-" << l << ".pgm";
        popsift::write_plane2Dunscaled( ostr.str().c_str(), false, p );

        ostringstream ostr2;
        ostr2 << "dir-octave-dump/" << basename << "-o-" << octave << "-l-" << l << ".dump";
        popsift::dump_plane2Dfloat(ostr2.str().c_str(), false, p );
    }

    for (int l = 0; l<_levels - 1; l++) {
        Plane2D_float p;
        p.copyFromPlane( _dog_3d, l );

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

} // namespace popsift

