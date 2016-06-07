#pragma once

namespace popart {

extern __device__ __constant__ float d_sigma0;
extern __device__ __constant__ float d_sigma_k;
extern __device__ __constant__ float d_edge_limit;
extern __device__ __constant__ float d_threshold;

void init_sigma(  float sigma, int level, float threshold, float edge_limit );

} // namespace popart
