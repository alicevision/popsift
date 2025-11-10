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
#include <sycl/sycl.hpp>  

#ifdef _WIN32
#include <direct.h>
#define stat _stat
#define mkdir(name, mode) _mkdir(name)
#endif

using namespace std;

namespace popsift {

Octave::Octave()
{ }

void Octave::alloc( const Config& conf, int width, int height, int levels, sycl::queue& shared_queue )
{
    _w = width;
    _h = height;
    _levels = levels;

    _w_grid_divider = float(_w) / conf.getFilterGridSize();
    _h_grid_divider = float(_h) / conf.getFilterGridSize();

    // Initialize SYCL queue first
    //initQueue();

     // Use shared queue instead
    _queue = &shared_queue;
    
    auto device = _queue->get_device();
    // std::cout << "Octave " << _debug_octave_id 
    //           << " allocating on device: " 
    //           << device.get_info<sycl::info::device::name>() 
    //           << std::endl;

   _data  .alloc( width, height, levels, _queue );
   _intm  .alloc( width, height, levels, _queue );
   _dog_3d.alloc( width, height, levels, _queue );
    
    // Verify allocation succeeded
    if (!_data.getDevicePtr() || !_dog_3d.getDevicePtr()) {
        throw std::runtime_error("Failed to allocate device memory for octave");
    }
    
    // std::cout << "Octave " << _debug_octave_id 
    //           << " allocated: data=" << _data.getDevicePtr()
    //           << ", dog=" << _dog_3d.getDevicePtr()
    //           << std::endl;
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

    // Wait for any pending operations on this queue before freeing
    try {
        _queue->wait();
    }
    catch (sycl::exception const& e) {
        std::cerr << "SYCL exception during queue wait in Octave::free(): " 
                  << e.what() << std::endl;
    }

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

    if (stat("dir-interm", &st) == -1) {
        mkdir("dir-interm", 0700);
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
       // Allocate HOST memory for downloading
       Plane2D_float p( _w, _h );  // This allocates host memory
       
      // Copy from device to host using SYCL
       const float* src_device = static_cast<const float*>(_data.getDevicePtr()) + (l * _h * _data.getPitchElements());
       _queue->memcpy(p.getHostPtr(), src_device, _w * _h * sizeof(float)).wait();

        ostringstream ostr;
        ostr << "dir-octave/" << basename << "-o-" << octave << "-l-" << l << ".pgm";
        popsift::write_plane2Dunscaled( ostr.str().c_str(), false, p );

        ostringstream ostr2;
        ostr2 << "dir-octave-dump/" << basename << "-o-" << octave << "-l-" << l << ".dump";
        popsift::dump_plane2Dfloat(ostr2.str().c_str(), false, p );
    }

    for( int l = 0; l<_levels; l++ ) {
       // Allocate HOST memory for downloading
       Plane2D_float p( _w, _h );
       
       // Copy from device to host using SYCL
       const float* src_device = static_cast<const float*>(_intm.getDevicePtr()) + (l * _h * _intm.getPitchElements());
       _queue->memcpy(p.getHostPtr(), src_device, _w * _h * sizeof(float)).wait();

        ostringstream ostr;
        ostr << "dir-interm/" << basename << "-o-" << octave << "-l-" << l << ".pgm";
        popsift::write_plane2Dunscaled( ostr.str().c_str(), false, p );
    }

    for (int l = 0; l<_levels - 1; l++) {
       // Allocate HOST memory for downloading
       Plane2D_float p( _w, _h );
       
       // Copy from device to host using SYCL
       const float* src_device = static_cast<const float*>(_dog_3d.getDevicePtr()) + (l * _h * _dog_3d.getPitchElements());
       _queue->memcpy(p.getHostPtr(), src_device, _w * _h * sizeof(float)).wait();

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

