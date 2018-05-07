#include <stdlib.h>
#include <iostream>
#include <cublas_v2.h>

#include "cublas_init.h"

void cublas_init( cublasHandle_t* handle, const char* file, int line )
{
    cublasStatus_t stat;
    stat = cublasCreate( handle );
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS initialization filed in " << file << ":" << line << std::endl;
        exit( -1 );
    }
}

void cublas_uninit( cublasHandle_t handle )
{
    cublasDestroy( handle );
}

