/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "debug_macros.h"

#include <cassert>

using namespace std;

void pop_sync_check_last_error( const char* file, size_t line )
{
    cudaDeviceSynchronize();
    pop_check_last_error( file, line );
}

void pop_check_last_error( const char* file, size_t line )
{
    cudaError_t err = cudaGetLastError( );
    if( err != cudaSuccess ) {
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                  << "    called from " << file << ":" << line << std::endl
                  << "    cudaGetLastError failed: " << cudaGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
}

namespace popsift { namespace cuda {
void malloc_dev( void** ptr, int sz,
                 const char* file, int line )
{
    cudaError_t err;
    err = cudaMalloc( ptr, sz );
    if( err != cudaSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
#ifdef DEBUG_INIT_DEVICE_ALLOCATIONS
    popsift::cuda::memset_sync( *ptr, 0, sz, file, line );
#endif // NDEBUG
}
} }

namespace popsift { namespace cuda {
void malloc_hst( void** ptr, int sz,
                 const char* file, int line )
{
    cudaError_t err;
    err = cudaMallocHost( ptr, sz );
    if( err != cudaSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    cudaMallocHost failed: " << cudaGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
#ifdef DEBUG_INIT_DEVICE_ALLOCATIONS
    memset( *ptr, 0, sz );
#endif // NDEBUG
}
} }

namespace popsift { namespace cuda {
void memcpy_async( void* dst, const void* src, size_t sz,
                   cudaMemcpyKind type, cudaStream_t stream,
                   const char* file, size_t line )
{
    POP_CHECK_NON_NULL_FL( dst, "Dest ptr in memcpy async is null.", file, line );
    POP_CHECK_NON_NULL_FL( src, "Source ptr in memcpy async is null.", file, line );
    POP_CHECK_NON_NULL_FL( sz, "Size in memcpy async is null.", file, line );

    cudaError_t err;
    err = cudaMemcpyAsync( dst, src, sz, type, stream );
    if( err != cudaSuccess ) {
        cerr << file << ":" << line << endl
             << "    " << "Failed to copy "
             << (type==cudaMemcpyHostToDevice?"host-to-device":"device-to-host")
             << ": ";
        cerr << cudaGetErrorString(err) << endl;
        cerr << "    src ptr=" << hex << (size_t)src << dec << endl
             << "    dst ptr=" << hex << (size_t)dst << dec << endl;
        exit( -__LINE__ );
    }
    POP_CUDA_FATAL_TEST( err, "Failed to copy host-to-device: " );
}

void memcpy_sync( void* dst, const void* src, size_t sz, cudaMemcpyKind type, const char* file, size_t line )
{
    POP_CHECK_NON_NULL( dst, "Dest ptr in memcpy async is null." );
    POP_CHECK_NON_NULL( src, "Source ptr in memcpy async is null." );
    POP_CHECK_NON_NULL( sz, "Size in memcpy async is null." );

    cudaError_t err;
    err = cudaMemcpy( dst, src, sz, type );
    if( err != cudaSuccess ) {
        cerr << "    " << "Failed to copy "
             << (type==cudaMemcpyHostToDevice?"host-to-device":"device-to-host")
             << ": ";
        cerr << cudaGetErrorString(err) << endl;
        cerr << "    src ptr=" << hex << (size_t)src << dec << endl
             << "    dst ptr=" << hex << (size_t)dst << dec << endl;
        exit( -__LINE__ );
    }
    POP_CUDA_FATAL_TEST( err, "Failed to copy host-to-device: " );
}

void memset_async( void* ptr, int value, size_t bytes, cudaStream_t stream, const char* file, size_t line )
{
    cudaError_t err;
    err = cudaMemsetAsync( ptr, value, bytes, stream );
    if( err != cudaSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    cudaMemsetAsync failed: " << cudaGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
}

void memset_sync( void* ptr, int value, size_t bytes, const char* file, size_t line )
{
    cudaError_t err;
    err = cudaMemset( ptr, value, bytes );
    if( err != cudaSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    cudaMemset failed: " << cudaGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
}
} }

namespace popsift { namespace cuda {
cudaStream_t stream_create( const char* file, size_t line )
{
    cudaStream_t stream;
    cudaError_t err;
    err = cudaStreamCreate( &stream );
    if( err != cudaSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    cudaStreamCreate failed: " << cudaGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
    return stream;
}
void stream_destroy( cudaStream_t s, const char* file, size_t line )
{
    cudaError_t err;
    err = cudaStreamDestroy( s );
    if( err != cudaSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    cudaStreamDestroy failed: " << cudaGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
}
cudaEvent_t event_create( const char* file, size_t line )
{
    cudaEvent_t ev;
    cudaError_t err;
    err = cudaEventCreate( &ev );
    if( err != cudaSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    cudaEventCreate failed: " << cudaGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
    return ev;
}
void event_destroy( cudaEvent_t ev, const char* file, size_t line )
{
    cudaError_t err;
    err = cudaEventDestroy( ev );
    if( err != cudaSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    cudaEventDestroy failed: " << cudaGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
}
void event_record( cudaEvent_t ev, cudaStream_t s, const char* file, size_t line )
{
    cudaError_t err;
    err = cudaEventRecord( ev, s );
    if( err != cudaSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    cudaEventRecord failed: " << cudaGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
}
void event_wait( cudaEvent_t ev, cudaStream_t s, const char* file, size_t line )
{
    cudaError_t err;
    err = cudaStreamWaitEvent( s, ev, 0 );
    if( err != cudaSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    cudaStreamWaitEvent failed: " << cudaGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
}

float event_diff( cudaEvent_t from, cudaEvent_t to )
{   
    float ms;
    cudaEventElapsedTime( &ms, from, to );
    return ms;
}

} // namespace cuda
} // namespace popsift

