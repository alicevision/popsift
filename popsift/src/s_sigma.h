#pragma once

namespace popart {

extern __device__ __constant__ float d_sigma0;
extern __device__ __constant__ float d_sigma_k;

void init_sigma(  float sigma, int level );

} // namespace popart
