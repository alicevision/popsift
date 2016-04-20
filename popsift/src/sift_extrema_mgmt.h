#pragma once

#include <inttypes.h>
#include <iostream>

namespace popart {

extern int h_max_extrema;
extern int h_max_orientations;
extern __device__ __constant__ int d_max_extrema;
extern __device__ __constant__ int d_max_orientations;

void init_extrema_limits( int max_extrema );

struct ExtremaMgmt
{
    int _counter;
};

} // namespace popart
