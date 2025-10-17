/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "assist.h"
#include "debug_macros.h"
#include "plane_base.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#ifndef _WIN32
#include <unistd.h>
#else
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <malloc.h>
#endif

using namespace std;

namespace popsift {

bool PlaneBase::alloc( int elemSize, int w, int h, int d )
{
    // _e     = elemSize;
    _pitch = w * elemSize;
    _x     = w;
    _y     = h;
    _z     = d;

    int sz = w * h * d * elemSize;

    _plane = malloc( sz );

    if( _plane ) return true;
        
    stringstream ss;
    ss << "Failed to allocate " << sz << " bytes of unaligned host memory." << endl
       << "Cause: " << strerror(errno);
    POP_FATAL(ss.str());
}


// NEW: SYCL-aware alloc implementation (device memory with queue)
bool PlaneBase::alloc( int elemSize, int w, int h, int d, sycl::queue* queue )
{
    if (!queue) {
        POP_FATAL("PlaneBase::alloc() called with null queue pointer");
        return false;
    }

    _pitch = w * elemSize;
    _x     = w;
    _y     = h;
    _z     = d;
    _queue = queue;  // Store queue for later deallocation

    size_t sz = static_cast<size_t>(w) * h * d * elemSize;

    try {
        // Allocate device memory using SYCL
        _plane = sycl::malloc_device(sz, *queue);
        
        if (!_plane) {
            stringstream ss;
            ss << "Failed to allocate " << sz << " bytes of device memory via SYCL.";
            POP_FATAL(ss.str());
            return false;
        }
        
        _mode = MemMode::CUDA1D;  // Mark as device memory
        return true;
    }
    catch (const sycl::exception& e) {
        stringstream ss;
        ss << "SYCL exception during device allocation: " << e.what();
        POP_FATAL(ss.str());
        return false;
    }
}

void PlaneBase::resize( int elemSize, int w, int h, int d )
{
    int new_sz = w * h * d * elemSize;
    int old_sz = _pitch * _y * _z;

    if( old_sz >= new_sz )
    {
        // _e     = elemSize;
        _pitch = w * elemSize;
        _x     = w;
        _y     = h;
        _z     = d;
    }
    else
    {
        dealloc();
        alloc( elemSize, w, h, d );
    }
}

void PlaneBase::adopt( void* ptr, int elemSize, int w, int h, int d )
{
    // _e     = elemSize;
    _pitch = w * elemSize;
    _x     = w;
    _y     = h;
    _z     = d;
    _plane = ptr;
}

void PlaneBase::dealloc( )
{
    if (!_plane) return;

    // Check if this is device memory allocated via SYCL
    if (_mode == MemMode::CUDA1D && _queue) {
        try {
            sycl::free(_plane, *_queue);
        }
        catch (const sycl::exception& e) {
            cerr << "SYCL exception during deallocation: " << e.what() << endl;
        }
    }
    else {
        // Regular host memory
        free(_plane);
    }
    
    _plane = nullptr;
    _queue = nullptr;
    _mode = MemMode::AlignmentUndefined;
}

} // namespace popsift

