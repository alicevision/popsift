#include "debug_macros.h"

#include <assert.h>

using namespace std;

void pop_info_gridsize( bool silent, dim3& grid,dim3& block, const string& kernel, const char* file, size_t line )
{
    if( silent ) return;

//    std::cerr << __FILE__ << ":" << __LINE__ << std::endl
//              << "    called from " << file << ":" << line << std::endl;
    std::cerr << "    " << kernel << " started with "
              << grid.x*grid.y*grid.z*block.x*block.y*block.z
              << " threads (";
    if( grid.z == 1 && grid.y == 1 )
        std::cerr << grid.x;
    else if( grid.z == 1 )
        std::cerr << "{" << grid.x << "," << grid.y << ")";
    else
        std::cerr << "{" << grid.x << "," << grid.y << "," << grid.z << ")";
    std::cerr << " blocks a ";
    if( block.z == 1 && block.y == 1 )
        std::cerr << block.x;
    else if( block.z == 1 )
        std::cerr << "{" << block.x << "," << block.y << ")";
    else
        std::cerr << "{" << block.x << "," << block.y << "," << block.z << ")";
    std::cerr << " threads)"
              << endl;
}

void pop_stream_synchronize( cudaStream_t stream, const char* file, size_t line )
{
    cudaError_t err = cudaStreamSynchronize( stream );
    if( err != cudaSuccess ) {
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                  << "    called from " << file << ":" << line << std::endl
                  << "    cudaStreamSynchronize failed: " << cudaGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
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
#if 0
    err = cudaDeviceSynchronize();
    if( err != cudaSuccess ) {
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                  << "    called from " << file << ":" << line << std::endl
                  << "    cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
#endif
}

void pop_cuda_memcpy_async( void* dst, const void* src, size_t sz, cudaMemcpyKind type, cudaStream_t stream, bool silent, const char* file, size_t line )
{
    POP_CHECK_NON_NULL_FL( dst, "Dest ptr in memcpy async is null.", file, line );
    POP_CHECK_NON_NULL_FL( src, "Source ptr in memcpy async is null.", file, line );
    POP_CHECK_NON_NULL_FL( sz, "Size in memcpy async is null.", file, line );
    if( not silent ) {
        cerr << file << ":" << line << endl
             << "    calling cudaMemcpyAsync("
             << hex << (size_t)src << "->"
             << (size_t)dst << dec << ","
             << (type==cudaMemcpyHostToDevice?"host-to-device":"device-to-host")
             << ")"<< endl;
    }
    // pop_stream_synchronize( stream, file, line );

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
    } else {
        if( not silent ) {
            cerr << "success" << endl;
        }
    }
    POP_CUDA_FATAL_TEST( err, "Failed to copy host-to-device: " );
}

void pop_cuda_memcpy( void* dst, const void* src, size_t sz, cudaMemcpyKind type, const char* file, size_t line )
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

void pop_cuda_memset_async( void* ptr, int value, size_t bytes, cudaStream_t stream, const char* file, size_t line )
{
    cudaError_t err;
    err = cudaMemsetAsync( ptr, value, bytes, stream );
    if( err != cudaSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    cudaMemsetAsync failed: " << cudaGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
}

void pop_cuda_memset( void* ptr, int value, size_t bytes, const char* file, size_t line )
{
    cudaError_t err;
    err = cudaMemset( ptr, value, bytes );
    if( err != cudaSuccess ) {
        std::cerr << file << ":" << line << std::endl
                  << "    cudaMemset failed: " << cudaGetErrorString(err) << std::endl;
        exit( -__LINE__ );
    }
}

