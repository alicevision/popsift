/*
 * Copyright 2016, Simula Research Laboratory
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

#include "debug_macros.h"

namespace popsift {

enum PlaneMapMode
{
    AlignmentUndefined = 0,
    Unaligned          = 2,
    PageAligned        = 3
};

/*************************************************************
 * PlaneBase
 * Non-templated base class for plane allocations. Implements
 * CUDA and system calls in a separate C++ file.
 *************************************************************/

struct PlaneBase
{
    void* allocHost2D( int w, int h, int elemSize );

    void freeHost2D( void* data );
};

/*************************************************************
 * PlaneT
 * Templated class containing the correctly typed pointer to
 * allocated data, and exposed the element size.
 *************************************************************/

template <typename T> struct PlaneT : public PlaneBase
{
    typedef T elem_type;

    enum { elem_size = sizeof(elem_type) };

    T* data;

    PlaneT( )      : data(0) { }
    explicit PlaneT( T* d ) : data(d) { }

    inline size_t elemSize() const { return elem_size; }
};

/*************************************************************
 * PitchPlane2D
 * Templated class containing the step size (CUDA terminology:
 * pitch) for a 2D plane. Able to return every rows of the
 * plane as pointer to elements (ie. array in the C sense).
 *************************************************************/

template <typename T> struct PitchPlane2D : public PlaneT<T>
{
    PitchPlane2D( ) : _pitchInBytes(0) { }

    PitchPlane2D( T* d, int s ) : PlaneT<T>(d) , _pitchInBytes(s) { }

    inline const T* ptr( int y ) const {
        return (const T*)( (const char*)this->data + y * _pitchInBytes );
    }
    inline       T* ptr( int y )       {
        return (T*)( (char*)this->data + y * _pitchInBytes );
    }

    inline void allocHost( int w, int h ) {
        this->data = (T*)PlaneBase::allocHost2D( w, h, this->elemSize(), mode );
        this->_pitchInBytes = w * this->elemSize();
    }

    inline void freeHost( PlaneMapMode mode ) {
        PlaneBase::freeHost2D( this->data, mode );
    }
    inline size_t getPitchInBytes( ) const { return _pitchInBytes; }

protected:
    size_t _pitchInBytes; // pitch width in bytes
};

/*************************************************************
 * Plane2D
 * Templated class containing the width and height (cols and
 * rows) of a 2D plane. Width is stored in terms of elements.
 *************************************************************/
template <typename T> class Plane2D : public PitchPlane2D<T>
{
    short _cols;
    short _rows;

public:
    Plane2D( )
        : _cols(0), _rows(0) { }

    Plane2D( int w, int h, T* d, int s )
        : PitchPlane2D<T>(d,s), _cols(w), _rows(h) { }

    Plane2D( int w, int h, const PitchPlane2D<T>& plane )
        : PitchPlane2D<T>(plane)
        , _cols(w)
        , _rows(h) { }

    template <typename U>
    explicit Plane2D( const Plane2D<U>& orig )
        : PitchPlane2D<T>( (T*)orig.data, orig._pitchInBytes )
        , _rows( orig.getRows() )
    {
        // careful computation: cols is a short
        int width = orig.getCols() * orig.elemSize();
        width /= this->elemSize();
        _cols = width;
    }

    /** Overwrite the width and height information. Useful if smaller
     *  planes should be loaded into larger preallocated host planes
     *  without actually allocating again, but dangerous.
     *  @warning: pitch is updated (host side)
     */
    void resetDimensionsHost( int w, int h );

    inline short getCols( ) const { return _cols; }
    inline short getWidth( ) const { return _cols; }
    inline short getRows( ) const { return _rows; }
    inline short getHeight( ) const { return _rows; }
    inline size_t getByteSize( ) const { return this->_pitchInBytes * _rows; }

    inline void allocHost( int w, int h, PlaneMapMode mode ) {
        _cols = w;
        _rows = h;
        PitchPlane2D<T>::allocHost( w, h, mode );
    }
};

/*************************************************************
 * Plane2D - functions
 * member functions for PitchPlane2D that have been extracted
 * for readability.
 *************************************************************/

template <typename T>
__host__
void Plane2D<T>::resetDimensionsHost( int w, int h )
{
    this->_cols = w;
    this->_rows = h;
    // on the host side, memory is contiguous (no padding) => pitch must be updated to match data
    this->_pitchInBytes  = w * this->elemSize();
}

/*************************************************************
 * Plane2D_#type
 * Typedefs for various template instances
 *************************************************************/

typedef PitchPlane2D<uint8_t>  PitchPlane2D_uint8;
typedef PitchPlane2D<uint16_t> PitchPlane2D_uint16;
typedef PitchPlane2D<float>    PitchPlane2D_float;
typedef PitchPlane2D<uchar2>   PitchPlane2D_uchar_2;
typedef PitchPlane2D<float4>   PitchPlane2D_float_4;

typedef Plane2D<uint8_t>      Plane2D_uint8;
typedef Plane2D<uint16_t>     Plane2D_uint16;
typedef Plane2D<float>        Plane2D_float;
typedef Plane2D<uchar2>       Plane2D_uchar_2;
typedef Plane2D<float4>       Plane2D_float_4;

} // namespace popsift

