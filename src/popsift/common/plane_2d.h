/*
 * Copyright 2016, Simula Research Laboratory
 *           2025, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cassert>
#include <cerrno>
#include <cinttypes>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <memory> // for shared pointer

#include "debug_macros.h"
#include "plane_base.h"

namespace popsift {

/*************************************************************
 * PlaneD
 *************************************************************/
template <typename T> class PlaneD
{
    std::shared_ptr<PlaneT<T> > _ptr {};

public:
    PlaneD( ) { }

    PlaneD( int w )
    {
        _ptr = new PlaneT<T>;
        _ptr->alloc( w );
    }

    PlaneD( int w, int h )
    {
        _ptr = new PlaneT<T>;
        _ptr->alloc( w, h );
    }

    PlaneD( int w, int h, int d )
    {
        _ptr = new PlaneT<T>;
        _ptr->alloc( w, h, d );
    }

    PlaneD( const PlaneD<T>& plane )
    {
        _ptr = plane._ptr;
    }

    /** Overwrite the width and height information. Useful if smaller
     *  planes should be loaded into larger preallocated host planes
     *  without actually allocating again, but dangerous.
     *  @warning: pitch is updated (host side)
     */
    inline void resetDimensions( int w = 1, int h = 1, int d = 1 ) {
        ptr->resize( w, h, d );
    }

    inline int getDimX( ) const     { return _ptr->getDimX(); }
    inline int getDimY( ) const     { return _ptr->getDimY(); }
    inline int getDimZ( ) const     { return _ptr->getDimZ(); }
    inline int getByteSize( ) const { return _ptr->getByteSize(); }

    inline void alloc( int w = 1, int h = 1, int d = 1 ) {
        _ptr = new PlaneT<T>;
        _ptr->alloc( w, h, d );
    }

    inline void dealloc( ) {
        _ptr->dealloc();
    }

    inline void copyToPlane( T* src ) {
        memcpy( _ptr->base(), src,  _ptr->getByteSize() );
    }

    inline       T& operator[]( int x )                     { return _ptr->deref( x ); }
    inline const T& operator[]( int x ) const               { return _ptr->deref( x ); }
    inline       T& operator[]( int y, int x )              { return _ptr->deref( y, x ); }
    inline const T& operator[]( int y, int x ) const        { return _ptr->deref( y, x ); }
    inline       T& operator[]( int z, int y, int x )       { return _ptr->deref( z, y, x ); }
    inline const T& operator[]( int z, int y, int x ) const { return _ptr->deref( z, y, x ); }

    inline T operator[]( PlaneMode m, float x ) const                   { return _ptr->get( m, x ); }
    inline T operator[]( PlaneMode m, float y, float x ) const          { return _ptr->get( m, y, x ); }
    inline T operator[]( PlaneMode m, float z, float y, float x ) const { return _ptr->get( m, z, y, x ); }
};

/*************************************************************
 * PlaneD#type
 * Typedefs for various template instances
 *************************************************************/

typedef PlaneD<uint8_t>      Plane2D_uint8;
typedef PlaneD<uint16_t>     Plane2D_uint16;
typedef PlaneD<float>        Plane2D_float;

} // namespace popsift

