#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

void pop_info_gridsize( bool               silent,
                        dim3&              grid,
                        dim3&              block,
                        const std::string& kernel,
                        const char*        file,
                        size_t             line );
#define POP_INFO_GRIDSIZE(silent,grid,block,kernel) \
    pop_info_gridsize(silent,grid,block,kernel,__FILE__,__LINE__)

void pop_stream_synchronize( cudaStream_t stream,
                             const char*  file,
                             size_t       line );
#define POP_SYNC( stream ) pop_stream_synchronize( stream, __FILE__, __LINE__ )

void pop_check_last_error( const char* file,
                           size_t      line );
#define POP_CHK pop_check_last_error( __FILE__, __LINE__ )

void pop_cuda_memcpy_async( void*          dst,
                            const void*    src,
                            size_t         sz,
                            cudaMemcpyKind type,
                            cudaStream_t   stream,
                            bool           silent,
                            const char*    file,
                            size_t         line );
#define POP_CUDA_MEMCPY_ASYNC( dst, src, sz, type, stream, silent ) \
    pop_cuda_memcpy_async( dst, src, sz, type, stream, silent, __FILE__, __LINE__ )

void pop_cuda_memcpy( void*          dst,
                      const void*    src,
                      size_t         sz,
                      cudaMemcpyKind type,
                      const char*    file,
                      size_t         line );
#define POP_CUDA_MEMCPY( dst, src, sz, type ) \
    pop_cuda_memcpy( dst, src, sz, type, __FILE__, __LINE__ )

void pop_cuda_memset_async( void*        ptr,
                            int          value,
                            size_t       bytes,
                            cudaStream_t stream,
                            const char*  file,
                            size_t       line );
#define POP_CUDA_MEMSET_ASYNC( ptr, val, sz, stream ) \
    pop_cuda_memset_async( ptr, val, sz, stream, __FILE__, __LINE__ )

void pop_cuda_memset( void*        ptr,
                      int          value,
                      size_t       bytes,
                      const char*  file,
                      size_t       line );
#define POP_CUDA_MEMSET( ptr, val, sz ) \
    pop_cuda_memset( ptr, val, sz, __FILE__, __LINE__ )

#define POP_FATAL(s) { \
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl << "    " << s << std::endl; \
        exit( -__LINE__ ); \
    }

#define POP_FATAL_FL(s,file,line) { \
        std::cerr << file << ":" << line << std::endl << "    " << s << std::endl; \
        exit( -__LINE__ ); \
    }

#define POP_CHECK_NON_NULL(ptr,s) if( ptr == 0 ) { POP_FATAL(s); }

#define POP_CHECK_NON_NULL_FL(ptr,s,file,line) if( ptr == 0 ) { POP_FATAL_FL(s,file,line); }

#define POP_INFO(s)
// #define POP_INFO(s) cerr << __FILE__ << ":" << __LINE__ << std::endl << "    " << s << endl

#define POP_INFO2(silent,s) \
    if (not silent) { \
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl << "    " << s << std::endl; \
    }

#define POP_CUDA_FATAL(err,s) { \
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl; \
        std::cerr << "    " << s << cudaGetErrorString(err) << std::endl; \
        exit( -__LINE__ ); \
    }
#define POP_CUDA_FATAL_TEST(err,s) if( err != cudaSuccess ) { POP_CUDA_FATAL(err,s); }

#define POP_CUDA_MALLOC( ptr, sz ) { \
        cudaError_t err; \
        err = cudaMalloc( ptr, sz ); \
        POP_CUDA_FATAL_TEST( err, "cudaMalloc failed: " ); \
        err = cudaMemset( *ptr, 0, sz ); \
        POP_CUDA_FATAL_TEST( err, "cudaMemset failed: " ); \
    }

#define POP_CUDA_MALLOC_HOST( ptr, sz ) { \
        cudaError_t err; \
        err = cudaMallocHost( ptr, sz ); \
        POP_CUDA_FATAL_TEST( err, "cudaMallocHost failed: " ); \
    }

#define POP_CUDA_STREAM_CREATE( ptr ) { \
        cudaError_t err; \
        err = cudaStreamCreate( ptr ); \
        POP_CUDA_FATAL_TEST( err, "cudaStreamCreate failed: " ); \
    }

