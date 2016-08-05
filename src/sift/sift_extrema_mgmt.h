#pragma once

#include <inttypes.h>
#include <iostream>

namespace popart {

struct MaxInfo
{
    int extrema;
    int orientations;
};

extern                         MaxInfo h_max;
extern __device__ __constant__ MaxInfo d_max;

// extern int h_max_extrema;
// extern int h_max_orientations;
// extern __device__ __constant__ int d_max_extrema;
// extern __device__ __constant__ int d_max_orientations;

void init_extrema_limits( int max_extrema );

} // namespace popart
