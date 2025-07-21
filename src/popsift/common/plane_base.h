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
#include <algorithm>

#include "debug_macros.h"

namespace popsift {

/* MemMode allows us to support a variety of allocation styles.
 * From normal malloc to aligned allocation to CUDA allocations
 */
enum MemMode
{
    AlignmentUndefined = 0,
    HostUnaligned      = 2,   // malloc
    HostAligned        = 3,   // memalign
    CUDA1D             = 10,  // cudaMalloc
    CUDA2D             = 11,  // cudaMallocPitch
    CUDA3D             = 12   // cudaMalloc3D
};

/* PlaneMode is used whenever we try to read from the plane with a float
 * index. The mode determines the interpolation mode and allows also
 * normalized access (relative to plane dimensions).
 */
namespace PlaneMode
{
class Point { };
class Linear { };
class NormalPoint { };
class NormalLinear { };
};

/*************************************************************
 * PlaneBase
 * Non-templated base class for plane allocations. Implements
 * CUDA and system calls in a separate C++ file.
 *************************************************************/

struct PlaneBase
{
    ~PlaneBase()
    {
        dealloc();
    }

    bool alloc( int elemSize, int w, int h = 1, int d = 1 );
    /*
     * when this is used for CUDA, add:
     * bool alloc( int elemSize, MemMode m, int w, int h = 1, int d = 1 );
     */

    void resize( int elemSize, int w, int h = 1, int d = 1 );

    void adopt( void* ptr, int elemSize, int w, int h = 1, int d = 1 );

    void dealloc( );

    inline void* base() {
        return _plane;
    }

    inline const void* base() const {
        return _plane;
    }

    inline void* row( const int& y ) {
        return reinterpret_cast<char*>(_plane) + ( y * _pitch );
    }

    inline const void* row( const int& y ) const {
        return reinterpret_cast<char*>(_plane) + ( y * _pitch );
    }

    inline void* row( const int& y, const int& z ) {
        return row( y + z * _y );
    }

    inline const void* row( const int& y, const int& z ) const {
        return row( y + z * _y );
    }

    inline void* plane( const int& z ) {
        return row( z * _y );
    }

    inline const void* plane( const int& z ) const {
        return row( z * _y );
    }

    inline int getDimX()  const { return _x; }
    inline int getDimY()  const { return _y; }
    inline int getDimZ()  const { return _z; }
    inline int getPitch() const { return _pitch; }
    inline int getByteSize() const { return _pitch * _y * _z; }

    inline int capX( const int& x ) const { return std::clamp( x, 0, _x-1 ); }
    inline int capY( const int& y ) const { return std::clamp( y, 0, _y-1 ); }
    inline int capZ( const int& z ) const { return std::clamp( z, 0, _z-1 ); }

private:
    void* _plane; /// the plane
    int   _pitch; /// width in bytes with padding
    int   _x;     /// width in elements
    int   _y;     /// height
    int   _z;     /// depth
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

    PlaneT( ) { }
    // explicit PlaneT( T* d ) : PlaneBase(d) { }

    inline void alloc( int w, int h = 1, int d = 1 )
    {
        PlaneBase::alloc( elemSize(), w, h, d );
    }

    inline void adopt( T* ptr, int w, int h = 1, int d = 1 )
    {
        PlaneBase::adopt( ptr, elemSize(), w, h, d );
    }

    inline void resize( int w, int h = 1, int d = 1 )
    {
        PlaneBase::resize( elemSize(), w, h, d );
    }

    inline size_t elemSize() const { return elem_size; }

    inline T& deref( int x )
    {
        x = capX( x );
        T* ptr = (T*)PlaneBase::base();
        return ptr[x];
    }

    inline T& deref( int y, int x )
    {
        y = capY( y );
        x = capX( x );
        T* ptr = (T*)PlaneBase::row(y);
        return ptr[x];
    }

    inline T& deref( int z, int y, int x )
    {
        z = capZ( z );
        y = capY( y );
        x = capX( x );
        T* ptr = (T*)PlaneBase::row(y,z);
        return ptr[x];
    }

    inline const T& deref( int x ) const
    {
        x = capX( x );
        T* ptr = (T*)PlaneBase::base();
        return ptr[x];
    }

    inline const T& deref( int y, int x ) const
    {
        y = capY( y );
        x = capX( x );
        T* ptr = (T*)PlaneBase::row(y);
        return ptr[x];
    }

    inline const T& deref( int z, int y, int x ) const
    {
        z = capZ( z );
        y = capY( y );
        x = capX( x );
        T* ptr = (T*)PlaneBase::row(y,z);
        return ptr[x];
    }

    inline T interpolate( const float& frac, T a, T b ) const;

    template <class M>
    inline T get( M m, const float& x ) const;

    template <class M>
    inline T get( M m, const float& y, const float& x ) const;

    template <class M>
    inline T get( M m, const float& z, const float& y, const float& x ) const;

    inline T getPoint( const float& x ) const;
    inline T getPoint( const float& y, const float& x ) const;
    inline T getPoint( const float& z, const float& y, const float& x ) const;
    inline T getLinear( const float& x ) const;
    inline T getLinear( const float& y, const float& x ) const;
    inline T getLinear( const float& z, const float& y, const float& x ) const;
    inline T getNormalPoint( const float& x ) const;
    inline T getNormalPoint( const float& y, const float& x ) const;
    inline T getNormalPoint( const float& z, const float& y, const float& x ) const;
    inline T getNormalLinear( const float& x ) const;
    inline T getNormalLinear( const float& y, const float& x ) const;
    inline T getNormalLinear( const float& z, const float& y, const float& x ) const;
};

template <typename T>
template <class M>
inline T PlaneT<T>::get( M m, const float& x ) const
{
    if     ( typeid(m) == typeid(PlaneMode::Point) )        return getPoint( x );
    else if( typeid(m) == typeid(PlaneMode::Linear) )       return getLinear( x );
    else if( typeid(m) == typeid(PlaneMode::NormalPoint) )  return getNormalPoint( x );
    else if( typeid(m) == typeid(PlaneMode::NormalLinear) ) return getNormalLinear( x );
    else return getPoint( x );
} 

template <typename T>
template <class M>
inline T PlaneT<T>::get( M m, const float& y, const float& x ) const
{
    if     ( typeid(m) == typeid(PlaneMode::Point) )        return getPoint( y, x );
    else if( typeid(m) == typeid(PlaneMode::Linear) )       return getLinear( y, x );
    else if( typeid(m) == typeid(PlaneMode::NormalPoint) )  return getNormalPoint( y, x );
    else if( typeid(m) == typeid(PlaneMode::NormalLinear) ) return getNormalLinear( y, x );
    else return getPoint( y, x );
} 

template <typename T>
template <class M>
inline T PlaneT<T>::get( M m, const float& z, const float& y, const float& x ) const
{
    if     ( typeid(m) == typeid(PlaneMode::Point) )        return getPoint( z, y, x );
    else if( typeid(m) == typeid(PlaneMode::Linear) )       return getLinear( z, y, x );
    else if( typeid(m) == typeid(PlaneMode::NormalPoint) )  return getNormalPoint( z, y, x );
    else if( typeid(m) == typeid(PlaneMode::NormalLinear) ) return getNormalLinear( z, y, x );
    else return getPoint( z, y, x );
} 

template <typename T>
inline T PlaneT<T>::interpolate( const float& frac, T a, T b ) const
{
    return frac * b + (1.0f-frac) * a;
}

template <typename T>
inline T PlaneT<T>::getLinear( const float& x ) const
{
    const int   x0 = (int)x; // quick floor computation
    const float xf = x - x0;
    return interpolate( xf, deref( x0   ),
                            deref( x0+1 ) );
}

template <typename T>
inline T PlaneT<T>::getLinear( const float& y, const float& x ) const
{
    const int   x0 = (int)x;
    const int   y0 = (int)y;
    const float xf = x - x0;
    const float yf = y - y0;
#if 1
    return deref( y0,   x0   );
    /*
    return interpolate( xf, deref( y0,   x0   ),
                            deref( y0,   x0+1 ) );
     */
#elif 0
    auto h1 = interpolate( xf, deref( y0,   x0   ),
                               deref( y0,   x0+1 ) );
    auto h2 = interpolate( xf, deref( y0+1, x0   ),
                               deref( y0+1, x0+1 ) );
    return interpolate( yf, h1, h2 );
#else
    return interpolate( yf, interpolate( xf, deref( y0,   x0   ),
                                             deref( y0,   x0+1 ) ),
                            interpolate( xf, deref( y0+1, x0   ),
                                             deref( y0+1, x0+1 ) ) );
#endif
}

template <typename T>
inline T PlaneT<T>::getLinear( const float& z, const float& y, const float& x ) const
{
    const int   x0 = (int)x;
    const int   y0 = (int)y;
    const int   z0 = (int)z;
    const float xf = x - x0;
    const float yf = y - y0;
    const float zf = z - z0;
    return interpolate( zf, interpolate( yf, interpolate( xf, deref( z0,   y0,   x0   ),
                                                              deref( z0,   y0,   x0+1 ) ),
                                             interpolate( xf, deref( z0,   y0+1, x0   ),
                                                              deref( z0,   y0+1, x0+1 ) ) ),
                            interpolate( yf, interpolate( xf, deref( z0+1, y0,   x0   ),
                                                              deref( z0+1, y0,   x0+1 ) ),
                                             interpolate( xf, deref( z0+1, y0+1, x0   ),
                                                              deref( z0+1, y0+1, x0+1 ) ) ) );
}

template <typename T>
inline T PlaneT<T>::getPoint( const float& x ) const
{
    // (int)(x+0.5f) is faster than roundf(x)
    return deref( (int)(x + 0.5f ) );
}

template <typename T>
inline T PlaneT<T>::getPoint( const float& y, const float& x ) const
{
    return deref( (int)(y + 0.5f ),
                  (int)(x + 0.5f ) );
}

template <typename T>
inline T PlaneT<T>::getPoint( const float& z, const float& y, const float& x ) const
{
    return deref( (int)(z + 0.5f ),
                  (int)(y + 0.5f ),
                  (int)(x + 0.5f ) );
}

template <typename T>
inline T PlaneT<T>::getNormalPoint( const float& x ) const
{
    return getPoint( x * getDimX() );
}

template <typename T>
inline T PlaneT<T>::getNormalPoint( const float& y, const float& x ) const
{
    return getPoint( y * getDimY(),
                     x * getDimX() );
}

template <typename T>
inline T PlaneT<T>::getNormalPoint( const float& z, const float& y, const float& x ) const
{
    return getPoint( z * getDimZ(),
                     y * getDimY(),
                     x * getDimX() );
}

template <typename T>
inline T PlaneT<T>::getNormalLinear( const float& x ) const
{
    return getLinear( x * getDimX() );
}

template <typename T>
inline T PlaneT<T>::getNormalLinear( const float& y, const float& x ) const
{
    return getLinear( y * getDimY(),
                      x * getDimX() );
}

template <typename T>
inline T PlaneT<T>::getNormalLinear( const float& z, const float& y, const float& x ) const
{
    return getLinear( z * getDimZ(),
                      y * getDimY(),
                      x * getDimX() );
}

} // namespace popsift

