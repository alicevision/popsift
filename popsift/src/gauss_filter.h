#pragma once

#include "sift_constants.h"

namespace popart {

extern __device__ __constant__ float d_gauss_filter[ GAUSS_ALIGN * GAUSS_LEVELS ];

void init_filter( float sigma0, int levels, bool vlfeat_mode );

} // namespace popart

