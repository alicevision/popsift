#pragma once

#define GAUSS_ALIGN  16
#define GAUSS_SPAN   8
#define GAUSS_LEVELS 12

namespace popart {

extern __device__ __constant__ float d_gauss_filter[ GAUSS_ALIGN * GAUSS_LEVELS ];

} // namespace popart

