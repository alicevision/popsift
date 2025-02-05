/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "assist.h"
#include "debug_macros.h"
#include "plane_2d.h"

#include <cuda_runtime.h>

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

void* PlaneBase::allocHost2D( int w, int h, int elemSize, PlaneMapMode m )
{
    int sz = w * h * elemSize;

    if( m == Unaligned )
    {
        void* ptr = malloc( sz );

        if( ptr ) return ptr;
        
        stringstream ss;
        ss << "Failed to allocate " << sz << " bytes of unaligned host memory." << endl
           << "Cause: " << strerror(errno);
        POP_FATAL(ss.str());
    }
    else if(m == PageAligned)
    {
        void* ptr = memalign(getPageSize(), sz);

        if(ptr) return ptr;

        cerr << __FILE__ << ":" << __LINE__ << endl
             << "    Failed to allocate " << sz << " bytes of page-aligned host memory." << endl
             << "    Cause: " << strerror(errno) << endl
             << "    Trying to allocate unaligned instead." << endl;

        return allocHost2D( w, h, elemSize, Unaligned );
    }
    else
    {
        POP_FATAL("Alignment not correctly specified in host plane allocation");
    }
}

__host__
void PlaneBase::freeHost2D( void* data, PlaneMapMode m )
{
    if (!data)
        return;
    else if (m == Unaligned) {
        free(data);
        return;
    }
    else if (m == PageAligned) {
        memalign_free( data );
        return;
    }
    assert(!"Invalid PlaneMapMode");
}

} // namespace popsift

