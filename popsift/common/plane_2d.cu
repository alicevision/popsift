#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "plane_2d.h"

using namespace std;

namespace popart {

__host__
void* PlaneBase::allocDev2D( size_t& pitch, int w, int h, int elemSize )
{
    void*       ptr;
    cudaError_t err;
    err = cudaMallocPitch( &ptr, &pitch, w * elemSize, h );
    if( err != cudaSuccess ) {
        cerr << __FILE__ << ":" << __LINE__ << endl
             << "    Cannot allocate " << w*h*elemSize << " bytes of CUDA memory" << endl
             << "    Cause: " << cudaGetErrorString( err ) << endl;
        exit( -1 );
    }
    return ptr;
}

__host__
void PlaneBase::freeDev2D( void* data )
{
    cudaError_t err;
    err = cudaFree( data );
    if( err != cudaSuccess ) {
        cerr << __FILE__ << ":" << __LINE__ << endl
             << "    Failed to free CUDA memory." << endl
             << "    Cause: " << cudaGetErrorString( err ) << endl;
        exit( -1 );
    }
}

__host__
void* PlaneBase::allocHost2D( int w, int h, int elemSize, PlaneMapMode m )
{
    int sz = w * h * elemSize;

    if( m == Unaligned ) {
        void* ptr = malloc( sz );

        if( ptr != 0 ) return ptr;
        
        char buf[100];
        strerror_r( errno, buf, 100 );
        cerr << __FILE__ << ":" << __LINE__ << endl
             << "    Failed to allocate " << sz << " bytes of unaligned host memory." << endl
             << "    Cause: " << buf << endl;
        exit( -1 );
    } else if( m == PageAligned ) {
        void* ptr;
        long  pagesize = sysconf(_SC_PAGESIZE);
        int   retval = posix_memalign( &ptr, pagesize, sz );

        if( retval == 0 ) return ptr;

        char buf[100];
        strerror_r( errno, buf, 100 );
        cerr << __FILE__ << ":" << __LINE__ << endl
             << "    Failed to allocate " << sz << " bytes of page-aligned host memory." << endl
             << "    Cause: " << buf << endl
             << "    Trying to allocate unaligned instead." << endl;

        return allocHost2D( w, h, elemSize, Unaligned );
    } else {
        cerr << __FILE__ << ":" << __LINE__ << endl
             << "    Alignment not correctly specified in host plane allocation" << endl;
        exit( -1 );
    }
}

__host__
void PlaneBase::freeHost2D( void* data )
{
    if( data != 0 ) free( data );
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
    if( err != cudaSuccess ) {
        cerr << __FILE__ << ":" << __LINE__ << endl
             << "    Failed to copy 2D plane host-to-device." << endl
             << "    Cause: " << cudaGetErrorString( err ) << endl;
        exit( -1 );
    }
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
    if( err != cudaSuccess ) {
        cerr << __FILE__ << ":" << __LINE__ << endl
             << "    Failed to copy 2D plane host-to-device." << endl
             << "    Cause: " << cudaGetErrorString( err ) << endl;
        exit( -1 );
    }
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
    if( err != cudaSuccess ) {
        cerr << __FILE__ << ":" << __LINE__ << endl
             << "    Failed to copy 2D plane device-to-host." << endl
             << "    Cause: " << cudaGetErrorString( err ) << endl;
        exit( -1 );
    }
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
    if( err != cudaSuccess ) {
        cerr << __FILE__ << ":" << __LINE__ << endl
             << "    Failed to copy 2D plane device-to-host." << endl
             << "    Cause: " << cudaGetErrorString( err ) << endl;
        exit( -1 );
    }
}

#ifdef PLANE2D_CUDA_OP_DEBUG
__host__
void PlaneBase::waitAndCheck( cudaStream_t stream ) const
{
    cudaStreamSynchronize( stream );
    cudaError_t err = cudaGetLastError( );
    if( err != cudaSuccess ) {
        cerr << __FILE__ << ":" << __LINE__ << endl
             << "    Failed in error check after async 2D plane operation." << endl
             << "    Cause: " << cudaGetErrorString( err ) << endl;
        exit( -1 );
    }
}
#endif // PLANE2D_CUDA_OP_DEBUG

} // namespace popart

