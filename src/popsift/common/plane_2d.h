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
#include <cstring>
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
        _ptr.reset( new PlaneT<T> );
        _ptr->alloc( w );
    }

    PlaneD( int w, int h )
    {
        _ptr.reset( new PlaneT<T> );
        _ptr->alloc( w, h );
    }

    PlaneD( int w, int h, int d )
    {
        _ptr.reset( new PlaneT<T> );
        _ptr->alloc( w, h, d );
    }

    PlaneD( PlaneD<T>& plane )
    {
        _ptr = plane._ptr;
    }

    inline bool isNull() const
    {
        return ( _ptr == nullptr );
    }

    /** Overwrite the width and height information. Useful if smaller
     *  planes should be loaded into larger preallocated host planes
     *  without actually allocating again, but dangerous.
     *  @warning: pitch is updated (host side)
     */
    inline void resetDimensions( int w = 1, int h = 1, int d = 1 ) {
        if( isNull() ) alloc( w, h, d );
        _ptr->resize( w, h, d );
    }

    inline int getDimX( ) const     { return _ptr->getDimX(); }
    inline int getDimY( ) const     { return _ptr->getDimY(); }
    inline int getDimZ( ) const     { return _ptr->getDimZ(); }
    inline int getByteSize( ) const { return _ptr->getByteSize(); }

    inline void alloc( int w = 1, int h = 1, int d = 1 ) {
        _ptr.reset( new PlaneT<T> );
        _ptr->alloc( w, h, d );
    }

    inline void dealloc( ) {
        _ptr->dealloc();
    }

    inline void memcpyFromBuffer( const void* ptr ) {
        std::memcpy( _ptr->base(), ptr,  _ptr->getByteSize() );
    }

    inline void copyFrom( const PlaneD<T>& src ) {

        int w = src._ptr->getDimX();
        int h = src._ptr->getDimY();
        int d = src._ptr->getDimZ();
        int p = src._ptr->getPitch();
        resetDimensions( w, h, d );

        if( _ptr->getPitch() != p ) {
            std::cerr << "E     Alignment trouble, different pitches" << std::endl;
        }

        std::memcpy( _ptr->base(), src._ptr->base(),  _ptr->getByteSize() );
    }

    inline void copyFromPlane( const PlaneD<T>& src, int zLevel ) {
        int w = src._ptr->getDimX();
        int h = src._ptr->getDimY();
        int d = 1;
        int p = src._ptr->getPitch();
        resetDimensions( w, h, d );

        if( _ptr->getPitch() != p ) {
            std::cerr << "E     Alignment trouble, different pitches" << std::endl;
        }

        std::memcpy( _ptr->base(), src._ptr->plane(zLevel),  p * h );
    }

    inline       T& get( int x )                     { return _ptr->deref( x ); }
    inline       T& get( int y, int x )              { return _ptr->deref( y, x ); }
    inline       T& get( int z, int y, int x )       { return _ptr->deref( z, y, x ); }
    inline const T& get( int x ) const               { return _ptr->deref( x ); }
    inline const T& get( int y, int x ) const        { return _ptr->deref( y, x ); }
    inline const T& get( int z, int y, int x ) const { return _ptr->deref( z, y, x ); }

    inline void set( int x, const T& v )               { _ptr->deref( x )       = v; }
    inline void set( int y, int x, const T& v )        { _ptr->deref( y, x )    = v; }
    inline void set( int z, int y, int x, const T& v ) { _ptr->deref( z, y, x ) = v; }

    inline T getM( PlaneMode::Mode m, const float& x ) const                                 { return _ptr->getM( m, x ); }
    inline T getM( PlaneMode::Mode m, const float& y, const float& x ) const                 { return _ptr->getM( m, y, x ); }
    inline T getM( PlaneMode::Mode m, const float& z, const float& y, const float& x ) const { return _ptr->getM( m, z, y, x ); }
};

/*************************************************************
 * PlaneD#type
 * Typedefs for various template instances
 *************************************************************/

typedef PlaneD<uint8_t>      Plane2D_uint8;
typedef PlaneD<uint16_t>     Plane2D_uint16;
typedef PlaneD<float>        Plane2D_float;

} // namespace popsift

