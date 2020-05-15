/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once
#include <cuda_runtime.h>

#include <cassert>
#include <cerrno>
#include <cinttypes>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>

#define PLANE2D_CUDA_OP_DEBUG

#ifndef NDEBUG
  // DEBUG mode
#ifndef PLANE2D_CUDA_OP_DEBUG
#define PLANE2D_CUDA_OP_DEBUG
#endif
#endif // not NDEBUG

namespace popsift {

enum PlaneMapMode
{
    AlignmentUndefined = 0,
    OnDevice           = 1,
    Unaligned          = 2,
    PageAligned        = 3,
    CudaAllocated      = 4
};

/*************************************************************
 * PlaneBase
 * Non-templated base class for plane allocations. Implements
 * CUDA and system calls in a separate C++ file.
 *************************************************************/

struct PlaneBase
{
    __host__
    void* allocDev2D( size_t& pitch, int w, int h, int elemSize );

    __host__
    void freeDev2D( void* data );

    __host__
    void* allocHost2D( int w, int h, int elemSize, PlaneMapMode m );

    __host__
    void freeHost2D( void* data, PlaneMapMode m );

    __host__
    void memcpyToDevice( void* dst, int dst_pitch, void* src, int src_pitch, short cols, short rows, int elemSize );

    __host__
    void memcpyToDevice( void* dst, int dst_pitch, void* src, int src_pitch, short cols, short rows, int elemSize, cudaStream_t stream );

    __host__
    void memcpyToHost( void* dst, int dst_pitch, void* src, int src_pitch, short cols, short rows, int elemSize );

    __host__
    void memcpyToHost( void* dst, int dst_pitch, void* src, int src_pitch, short cols, short rows, int elemSize, cudaStream_t stream );

#ifdef PLANE2D_CUDA_OP_DEBUG
    __host__
    void waitAndCheck( cudaStream_t stream ) const;
#else // not PLANE2D_CUDA_OP_DEBUG
    __host__
    inline void waitAndCheck( cudaStream_t stream ) const { }
#endif // not PLANE2D_CUDA_OP_DEBUG
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

    __host__ __device__ PlaneT( )      : data(0) { }
    __host__ __device__ explicit PlaneT( T* d ) : data(d) { }

    __host__ __device__ inline size_t elemSize() const { return elem_size; }
};

/*************************************************************
 * PitchPlane2D
 * Templated class containing the step size (CUDA terminology:
 * pitch) for a 2D plane. Able to return every rows of the
 * plane as pointer to elements (ie. array in the C sense).
 *************************************************************/

template <typename T> struct PitchPlane2D : public PlaneT<T>
{
    __host__ __device__
    PitchPlane2D( ) : _pitchInBytes(0) { }

    __host__ __device__
    PitchPlane2D( T* d, int s ) : PlaneT<T>(d) , _pitchInBytes(s) { }

    /** cuda memcpy from this (plane allocated on host) to
     *  parameter (plane allocated on device) */
    __host__ inline void memcpyToDevice( PitchPlane2D<T>& devPlane,
                                         short cols, short rows );
    __host__ inline void memcpyToDevice( PitchPlane2D<T>& devPlane,
                                         short cols, short rows, cudaStream_t stream );

    /** cuda memcpy from parameter (plane allocated on host) to
     *  this (plane allocated on device) */
    __host__ inline void memcpyFromHost( PitchPlane2D<T>& hostPlane,
                                         short cols, short rows );
    __host__ inline void memcpyFromHost( PitchPlane2D<T>& hostPlane,
                                         short cols, short rows, cudaStream_t stream );

    /** cuda memcpy from parameter (plane allocated on device) to
     *  this (plane allocated on host) */
    __host__ inline void memcpyFromDevice( PitchPlane2D<T>& devPlane,
                                           short cols, short rows );
    __host__ inline void memcpyFromDevice( PitchPlane2D<T>& devPlane,
                                           short cols, short rows, cudaStream_t stream );

    /** cuda memcpy from this (plane allocated on device) to
     *  parameter (plane allocated on host) */
    __host__ inline void memcpyToHost( PitchPlane2D<T>& hostPlane,
                                       short cols, short rows );
    __host__ inline void memcpyToHost( PitchPlane2D<T>& hostPlane,
                                       short cols, short rows, cudaStream_t stream );

    __host__ __device__ inline const T* ptr( int y ) const {
        return (const T*)( (const char*)this->data + y * _pitchInBytes );
    }
    __host__ __device__ inline       T* ptr( int y )       {
        return (T*)( (char*)this->data + y * _pitchInBytes );
    }

    __host__ inline void allocDev( int w, int h ) {
        size_t pitch;
        this->data = (T*)PlaneBase::allocDev2D( pitch, w, h, this->elemSize() );
        this->_pitchInBytes = pitch;
    }

    __host__ inline void freeDev( ) {
        assert( this->data );
        PlaneBase::freeDev2D( this->data );
        this->data = 0;
    }

    __host__ inline void allocHost( int w, int h, PlaneMapMode mode ) {
        this->data = (T*)PlaneBase::allocHost2D( w, h, this->elemSize(), mode );
        this->_pitchInBytes = w * this->elemSize();
    }

    __host__ inline void freeHost( PlaneMapMode mode ) {
        PlaneBase::freeHost2D( this->data, mode );
    }
    __host__ __device__
    inline short getPitchInBytes( ) const { return _pitchInBytes; }

protected:
    int _pitchInBytes; // pitch width in bytes
};

/*************************************************************
 * PitchPlane2D - functions
 * member functions for PitchPlane2D that have been extracted
 * for readability.
 *************************************************************/

template <typename T>
__host__
inline void PitchPlane2D<T>::memcpyToDevice( PitchPlane2D<T>& devPlane, short cols, short rows )
{
    PlaneBase::memcpyToDevice( devPlane.data, devPlane._pitchInBytes,
                               this->data, this->_pitchInBytes,
                               cols, rows,
                               sizeof(T) );
}

template <typename T>
__host__
inline void PitchPlane2D<T>::memcpyToDevice( PitchPlane2D<T>& devPlane, short cols, short rows, cudaStream_t stream )
{
    PlaneBase::memcpyToDevice( devPlane.data, devPlane._pitchInBytes,
                               this->data, this->_pitchInBytes,
                               cols, rows,
                               sizeof(T),
                               stream );
}

template <typename T>
__host__
inline void PitchPlane2D<T>::memcpyFromHost( PitchPlane2D<T>& hostPlane, short cols, short rows )
{
    hostPlane.memcpyToDevice( *this, cols, rows );
}

template <typename T>
__host__
inline void PitchPlane2D<T>::memcpyFromHost( PitchPlane2D<T>& hostPlane, short cols, short rows, cudaStream_t stream )
{
    hostPlane.memcpyToDevice( *this, cols, rows, stream );
}

template <typename T>
__host__
inline void PitchPlane2D<T>::memcpyFromDevice( PitchPlane2D<T>& devPlane, short cols, short rows )
{
    PlaneBase::memcpyToHost( this->data, this->_pitchInBytes,
                             devPlane.data, devPlane._pitchInBytes,
                             cols, rows,
                             sizeof(T) );
}

template <typename T>
__host__
inline void PitchPlane2D<T>::memcpyFromDevice( PitchPlane2D<T>& devPlane, short cols, short rows, cudaStream_t stream )
{
    PlaneBase::memcpyToHost( this->data, this->_pitchInBytes,
                             devPlane.data, devPlane._pitchInBytes,
                             cols, rows,
                             sizeof(T),
                             stream );
}

template <typename T>
__host__
inline void PitchPlane2D<T>::memcpyToHost( PitchPlane2D<T>& hostPlane, short cols, short rows )
{
    hostPlane.memcpyFromDevice( *this, cols, rows );
}

template <typename T>
__host__
inline void PitchPlane2D<T>::memcpyToHost( PitchPlane2D<T>& hostPlane, short cols, short rows, cudaStream_t stream )
{
    hostPlane.memcpyFromDevice( *this, cols, rows, stream );
}

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
    __host__ __device__
    Plane2D( )
        : _cols(0), _rows(0) { }

    __host__ __device__
    Plane2D( int w, int h, T* d, int s )
        : PitchPlane2D<T>(d,s), _cols(w), _rows(h) { }

    __host__ __device__
    Plane2D( int w, int h, const PitchPlane2D<T>& plane )
        : PitchPlane2D<T>(plane)
        , _cols(w)
        , _rows(h) { }

    template <typename U>
    __host__ __device__
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
    __host__ void resetDimensionsHost( int w, int h );

    /** Overwrite the width and height information. Useful if smaller
     *  planes should be loaded into larger preallocated device planes
     *  without actually allocating again, but dangerous.
     *  @warning: pitch is not updated (device side)
     */
    __host__ void resetDimensionsDev( int w, int h );

    /** cuda memcpy from this (plane allocated on host) to
     *  parameter (plane allocated on device) */
    __host__ inline void memcpyToDevice( Plane2D<T>& devPlane );
    __host__ inline void memcpyToDevice( PitchPlane2D<T>& devPlane );
    __host__ inline void memcpyToDevice( Plane2D<T>& devPlane, cudaStream_t stream );
    __host__ inline void memcpyToDevice( PitchPlane2D<T>& devPlane, cudaStream_t stream );

    /** cuda memcpy from parameter (plane allocated on host) to
     *  this (plane allocated on device) */
    __host__ inline void memcpyFromHost( Plane2D<T>& hostPlane );
    __host__ inline void memcpyFromHost( Plane2D<T>& hostPlane, cudaStream_t stream );
    __host__ inline void memcpyFromHost( PitchPlane2D<T>& hostPlane );
    __host__ inline void memcpyFromHost( PitchPlane2D<T>& hostPlane, cudaStream_t stream );

    /** cuda memcpy from parameter (plane allocated on device) to
     *  this (plane allocated on host) */
    __host__ inline void memcpyFromDevice( Plane2D<T>& devPlane );
    __host__ inline void memcpyFromDevice( PitchPlane2D<T>& devPlane );
    __host__ inline void memcpyFromDevice( Plane2D<T>& devPlane, cudaStream_t stream );
    __host__ inline void memcpyFromDevice( PitchPlane2D<T>& devPlane, cudaStream_t stream );

    /** cuda memcpy from this (plane allocated on device) to
     *  parameter (plane allocated on host) */
    __host__ inline void memcpyToHost( Plane2D<T>& hostPlane );
    __host__ inline void memcpyToHost( Plane2D<T>& hostPlane, cudaStream_t stream );
    __host__ inline void memcpyToHost( PitchPlane2D<T>& hostPlane );
    __host__ inline void memcpyToHost( PitchPlane2D<T>& hostPlane, cudaStream_t stream );

    __host__ __device__
    inline short getCols( ) const { return _cols; }
    __host__ __device__
    inline short getWidth( ) const { return _cols; }
    __host__ __device__
    inline short getRows( ) const { return _rows; }
    __host__ __device__
    inline short getHeight( ) const { return _rows; }
    __host__ __device__
    inline short getByteSize( ) const { return this->_pitchInBytes*_rows; }

    __host__ inline void allocDev( int w, int h ) {
        _cols = w;
        _rows = h;
        PitchPlane2D<T>::allocDev( w, h );
    }

    __host__ inline void allocHost( int w, int h, PlaneMapMode mode ) {
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

template <typename T>
__host__
void Plane2D<T>::resetDimensionsDev( int w, int h )
{
    // validate pitch
    if( w * this->elemSize() > this->getPitchInBytes() ) {
        std::stringstream err; 
        err << __FILE__ << ":" << __LINE__ << std::endl
        << " Error: trying to reinterpret plane width to " << w << " units a " << sizeof(T) << " bytes, "
        << "only " << this->getPitchInBytes() << " bytes allocated";
        throw std::runtime_error(err.str());
    }
    this->_cols = w;
    this->_rows = h;
    // on the device side, memory is NOT contiguous (CUDA may add padding) => pitch can not be changed without reallocation
}

template <typename T>
__host__
inline void Plane2D<T>::memcpyToDevice( Plane2D<T>& devPlane )
{
    assert( devPlane._cols == this->_cols );
    assert( devPlane._rows == this->_rows );
    PitchPlane2D<T>::memcpyToDevice( devPlane, this->_cols, this->_rows );
}

template <typename T>
__host__
inline void Plane2D<T>::memcpyToDevice( PitchPlane2D<T>& devPlane )
{
    PitchPlane2D<T>::memcpyToDevice( devPlane, this->_cols, this->_rows );
}

template <typename T>
__host__
inline void Plane2D<T>::memcpyToDevice( Plane2D<T>& devPlane, cudaStream_t stream )
{
    if( devPlane._cols != this->_cols ) {
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                  << "    Error: source columns (" << this->_cols << ") and dest columns (" << devPlane._cols << ") must be identical" << std::endl;
        exit( -1 );
    }
    if( devPlane._rows != this->_rows ) {
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                  << "    Error: source rows (" << this->_rows << ") and dest rows (" << devPlane._rows << ") must be identical" << std::endl;
        exit( -1 );
    }
    PitchPlane2D<T>::memcpyToDevice( devPlane, this->_cols, this->_rows, stream );
}

template <typename T>
__host__
inline void Plane2D<T>::memcpyToDevice( PitchPlane2D<T>& devPlane, cudaStream_t stream )
{
    PitchPlane2D<T>::memcpyToDevice( devPlane, this->_cols, this->_rows, stream );
}

template <typename T>
__host__
inline void Plane2D<T>::memcpyFromHost( Plane2D<T>& hostPlane )
{
    hostPlane.memcpyToDevice( *this );
}

template <typename T>
__host__
inline void Plane2D<T>::memcpyFromHost( PitchPlane2D<T>& hostPlane )
{
    hostPlane.memcpyToDevice( *this, this->_cols, this->_rows );
}

template <typename T>
__host__
inline void Plane2D<T>::memcpyFromHost( Plane2D<T>& hostPlane, cudaStream_t stream )
{
    hostPlane.memcpyToDevice( *this, stream );
}

template <typename T>
__host__
inline void Plane2D<T>::memcpyFromHost( PitchPlane2D<T>& hostPlane, cudaStream_t stream )
{
    hostPlane.memcpyToDevice( *this, this->_cols, this->_rows, stream );
}

template <typename T>
__host__
inline void Plane2D<T>::memcpyFromDevice( Plane2D<T>& devPlane )
{
    assert( devPlane._cols == this->_cols );
    assert( devPlane._rows == this->_rows );
    PitchPlane2D<T>::memcpyFromDevice( devPlane, this->_cols, this->_rows );
}

template <typename T>
__host__
inline void Plane2D<T>::memcpyFromDevice( PitchPlane2D<T>& devPlane )
{
    PitchPlane2D<T>::memcpyFromDevice( devPlane, this->_cols, this->_rows );
}

template <typename T>
__host__
inline void Plane2D<T>::memcpyFromDevice( Plane2D<T>& devPlane, cudaStream_t stream )
{
    assert( devPlane._cols == this->_cols );
    assert( devPlane._rows == this->_rows );
    PitchPlane2D<T>::memcpyFromDevice( devPlane, this->_cols, this->_rows, stream );
}

template <typename T>
__host__
inline void Plane2D<T>::memcpyFromDevice( PitchPlane2D<T>& devPlane, cudaStream_t stream )
{
    PitchPlane2D<T>::memcpyFromDevice( devPlane, this->_cols, this->_rows, stream );
}

template <typename T>
__host__
inline void Plane2D<T>::memcpyToHost( Plane2D<T>& hostPlane )
{
    hostPlane.memcpyFromDevice( *this );
}

template <typename T>
__host__
inline void Plane2D<T>::memcpyToHost( PitchPlane2D<T>& hostPlane )
{
    hostPlane.memcpyFromDevice( *this, this->_cols, this->_rows );
}

template <typename T>
__host__
inline void Plane2D<T>::memcpyToHost( Plane2D<T>& hostPlane, cudaStream_t stream )
{
    hostPlane.memcpyFromDevice( *this, stream );
}

template <typename T>
__host__
inline void Plane2D<T>::memcpyToHost( PitchPlane2D<T>& hostPlane, cudaStream_t stream )
{
    hostPlane.memcpyFromDevice( *this, this->_cols, this->_rows, stream );
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

