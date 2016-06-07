#pragma once

#include "sift_constants.h"

namespace popart {

extern __device__ __constant__ float d_gauss_filter_initial_blur[ GAUSS_ALIGN ];

extern __device__ __constant__ float d_gauss_filter[ GAUSS_ALIGN * GAUSS_LEVELS ];

extern __device__ __constant__ float d_gauss_from_lvl_1[ GAUSS_ALIGN * GAUSS_LEVELS ];

// void init_filter( float sigma0, int levels, bool vlfeat_mode );
void init_filter( float sigma0,
                  int   levels,
                  bool  vlfeat_mode,
                  bool  assume_initial_blur,
                  float initial_blur,
                  float downsampling );

} // namespace popart

