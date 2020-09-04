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

__host__
void* PlaneBase::allocDev2D( size_t& pitch, int w, int h, int elemSize, PlaneMapMode m )
{
    _mode = m;

    // cerr << "Alloc " << w*h*elemSize << " B" << endl;
    void*       ptr;
    cudaError_t err;

    std::cerr << "Allocating 2D plane, elem size " << elemSize
              << " size " << w << "x" << h
              << " type " << (m==OnDevice?"OnDevice":"ManagedMen")
              << " pitch " << pitch
              << std::endl;

    if( m == OnDevice )
    {
        std::cerr << "    Trying to allocate " << w * elemSize << "x" << h << " bytes" << std::endl;
        err = cudaMallocPitch( &ptr, &pitch, w * elemSize, h );
        POP_CUDA_FATAL_TEST( err, "Cannot allocate pitched CUDA memory: " );
        std::cerr << "    Allocated " << pitch << "x" << h << "=" << pitch*h << " bytes" << std::endl;
        return ptr;
    }
    else if( m == ManagedMem )
    {
        size_t sz = w * elemSize;
        std::cerr << "    Trying to allocate " << w * elemSize << "x" << h << " bytes with alignment " << pitch << std::endl;
        size_t rest = sz % pitch;
        if( rest == 0 )
            pitch = sz;
        else
            pitch = sz + pitch - rest;
        std::cerr << "    Trying to allocate " << pitch << "x" << h << "=" << pitch*h << " bytes" << std::endl;
        sz  = pitch * h;
        err = cudaMallocManaged( &ptr, sz );
        POP_CUDA_FATAL_TEST( err, "Cannot allocate managed CUDA memory: " );
        std::cerr << "    Allocated " << pitch << "x" << h << "=" << pitch*h << " bytes" << std::endl;
        return ptr;
    }
    else
    {
        cerr << __FILE__ << ":" << __LINE__ << endl
             << "    Allocation mode not correct in device plane allocation" << endl;
        exit( -1 );
    }
}

__host__
void* PlaneBase::allocHost2D( int w, int h, int elemSize, PlaneMapMode m )
{
    _mode = m;

    int sz = w * h * elemSize;

    if( m == Unaligned )
    {
        void* ptr = malloc( sz );

        if( ptr != 0 ) return ptr;
        
#ifdef _GNU_SOURCE
        char b[100];
        const char* buf = strerror_r( errno, b, 100 );
#else
        const char *buf = strerror(errno);
#endif
        cerr << __FILE__ << ":" << __LINE__ << endl
             << "    Failed to allocate " << sz << " bytes of unaligned host memory." << endl
             << "    Cause: " << buf << endl;
        exit( -1 );
    }
    else if( m == PageAligned )
    {
        void* ptr = memalign(getPageSize(), sz);
        if(ptr)
            return ptr;

#ifdef _GNU_SOURCE
        char b[100];
        const char* buf = strerror_r( errno, b, 100 );
#else
		const char* buf = strerror(errno);
#endif
        cerr << __FILE__ << ":" << __LINE__ << endl
             << "    Failed to allocate " << sz << " bytes of page-aligned host memory." << endl
             << "    Cause: " << buf << endl
             << "    Trying to allocate unaligned instead." << endl;

        return allocHost2D( w, h, elemSize, Unaligned );
    }
    else if( m == CudaAllocated )
    {
        void* ptr;
        cudaError_t err;
        err = cudaMallocHost( &ptr, sz );
        POP_CUDA_FATAL_TEST( err, "Failed to allocate aligned and pinned host memory: " );
        return ptr;
    }
    else
    {
        cerr << __FILE__ << ":" << __LINE__ << endl
             << "    Allocation mode not correct in host plane allocation" << endl;
        exit( -1 );
    }
}

__host__
void PlaneBase::free2D( void* data )
{
    if (!data)
        return;

    switch( _mode )
    {
    case OnDevice :
    case ManagedMem :
        cudaFree(data);
        return;
    case CudaAllocated :
        cudaFreeHost(data);
        return;
    case Unaligned :
        free(data);
        return;
    case PageAligned :
        memalign_free( data );
        return;
    default :
        assert(!"Invalid PlaneMapMode");
    }
}

__host__
void PlaneBase::memcpyToDevice( void* dst, int dst_pitch,
                                void* src, int src_pitch,
                                short cols, short rows,
                                int elemSize )
{
    assert( dst != 0 );
    assert( dst_pitch != 0 );
    assert( src != 0 );
    assert( src_pitch != 0 );
    assert( cols != 0 );
    assert( rows != 0 );
    cudaError_t err;
    err = cudaMemcpy2D( dst, dst_pitch,
                        src, src_pitch,
                        cols*elemSize, rows,
                        cudaMemcpyHostToDevice );
    POP_CUDA_FATAL_TEST( err, "Failed to copy 2D plane host-to-device: " );
}

__host__
void PlaneBase::memcpyToDevice( void* dst, int dst_pitch,
                                void* src, int src_pitch,
                                short cols, short rows,
                                int elemSize,
                                cudaStream_t stream )
{
    assert( dst != 0 );
    assert( dst_pitch != 0 );
    assert( src != 0 );
    assert( src_pitch != 0 );
    assert( cols != 0 );
    assert( rows != 0 );
    cudaError_t err;
    err = cudaMemcpy2DAsync( dst, dst_pitch,
                             src, src_pitch,
                             cols*elemSize, rows,
                             cudaMemcpyHostToDevice,
                             stream );
    POP_CUDA_FATAL_TEST( err, "Failed to copy 2D plane host-to-device: " );
}

__host__
void PlaneBase::memcpyToHost( void* dst, int dst_pitch,
                              void* src, int src_pitch,
                              short cols, short rows,
                              int elemSize )
{
    assert( dst != 0 );
    assert( dst_pitch != 0 );
    assert( src != 0 );
    assert( src_pitch != 0 );
    assert( cols != 0 );
    assert( rows != 0 );
    cudaError_t err;
    err = cudaMemcpy2D( dst, dst_pitch,
                        src, src_pitch,
                        cols*elemSize, rows,
                        cudaMemcpyDeviceToHost );
    POP_CUDA_FATAL_TEST( err, "Failed to copy 2D plane device-to-host: " );
}

__host__
void PlaneBase::memcpyToHost( void* dst, int dst_pitch,
                              void* src, int src_pitch,
                              short cols, short rows,
                              int elemSize,
                              cudaStream_t stream )
{
    assert( dst != 0 );
    assert( dst_pitch != 0 );
    assert( src != 0 );
    assert( src_pitch != 0 );
    assert( cols != 0 );
    assert( rows != 0 );
    cudaError_t err;
    err = cudaMemcpy2DAsync( dst, dst_pitch,
                             src, src_pitch,
                             cols*elemSize, rows,
                             cudaMemcpyDeviceToHost,
                             stream );
    POP_CUDA_FATAL_TEST( err, "Failed to copy 2D plane device-to-host: " );
}

#ifdef PLANE2D_CUDA_OP_DEBUG
__host__
void PlaneBase::waitAndCheck( cudaStream_t stream ) const
{
    cudaStreamSynchronize( stream );
    cudaError_t err = cudaGetLastError( );
    POP_CUDA_FATAL_TEST( err, "Failed in error check after async 2D plane operation: " );
}
#endif // PLANE2D_CUDA_OP_DEBUG

} // namespace popsift

