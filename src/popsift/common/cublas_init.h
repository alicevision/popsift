#pragma once

#include <cublas_v2.h>

void cublas_init  ( cublasHandle_t* handle, const char* file, int line );
void cublas_uninit( cublasHandle_t  handle );

