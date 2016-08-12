/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "plane_2d.h"
#include "debug_macros.h"

using namespace std;

namespace popart {

__host__
void* PlaneBase::allocDev2D( size_t& pitch, int w, int h, int elemSize )
{
    // cerr << "Alloc " << w*h*elemSize << " B" << endl;
    void*       ptr;
    cudaError_t err;
    err = cudaMallocPitch( &ptr, &pitch, w * elemSize, h );
    POP_CUDA_FATAL_TEST( err, "Cannot allocate pitched CUDA memory: " );
    return ptr;
}

__host__
void PlaneBase::freeDev2D( void* data )
{
    cudaError_t err;
    err = cudaFree( data );
    POP_CUDA_FATAL_TEST( err, "Failed to free CUDA memory: " );
}

__host__
void* PlaneBase::allocHost2D( int w, int h, int elemSize, PlaneMapMode m )
{
    int sz = w * h * elemSize;

    if( m == Unaligned ) {
        void* ptr = malloc( sz );

        if( ptr != 0 ) return ptr;
        
#ifdef _GNU_SOURCE
        char b[100];
        const char* buf = strerror_r( errno, b, 100 );
#else
        char buf[100];
        strerror_r( errno, buf, 100 );
#endif
        cerr << __FILE__ << ":" << __LINE__ << endl
             << "    Failed to allocate " << sz << " bytes of unaligned host memory." << endl
             << "    Cause: " << buf << endl;
        exit( -1 );
    } else if( m == PageAligned ) {
        void* ptr;
        long  pagesize = sysconf(_SC_PAGESIZE);
        int   retval = posix_memalign( &ptr, pagesize, sz );

        if( retval == 0 ) return ptr;

#ifdef _GNU_SOURCE
        char b[100];
        const char* buf = strerror_r( errno, b, 100 );
#else
        char buf[100];
        strerror_r( errno, buf, 100 );
#endif
        cerr << __FILE__ << ":" << __LINE__ << endl
             << "    Failed to allocate " << sz << " bytes of page-aligned host memory." << endl
             << "    Cause: " << buf << endl
             << "    Trying to allocate unaligned instead." << endl;

        return allocHost2D( w, h, elemSize, Unaligned );
    } else if( m == CudaAllocated ) {
        void* ptr;
        cudaError_t err;
        err = cudaMallocHost( &ptr, sz );
        POP_CUDA_FATAL_TEST( err, "Failed to allocate aligned and pinned host memory: " );
        return ptr;
    } else {
        cerr << __FILE__ << ":" << __LINE__ << endl
             << "    Alignment not correctly specified in host plane allocation" << endl;
        exit( -1 );
    }
}

__host__
void PlaneBase::freeHost2D( void* data, PlaneMapMode m )
{
    if( data != 0 ) {
        if( m != CudaAllocated ) {
            free( data );
        } else {
            cudaFreeHost( data );
        }
        data = 0;
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

} // namespace popart

