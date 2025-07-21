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

    free(_plane);
}

} // namespace popsift

