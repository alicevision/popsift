#include <iostream>

/* CUDA-C includes */
/* Error if not compiled with -I/path/to/cuda/inc */
#include "cuda_runtime_api.h"

#define MAX_NBLOCKS 65536/2

#define CUDA_CHECK_ERROR()                                              \
    {                                                                   \
        cudaError_t ce = cudaGetLastError();                            \
        if(ce != cudaSuccess) {                                         \
            std::cerr << cudaGetErrorString(ce) << " in file '"         \
                      << __FILE__ << "' in line " << __LINE__<< std::endl; \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    }
