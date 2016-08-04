#pragma once

#include "sift_constants.h"

namespace popart {

struct Extremum
{
    float xpos;
    float ypos;
    float sigma; // scale;

    int   num_ori; // number of this extremum's orientations
    int   idx_ori; // exclusive prefix sum of the layer's orientations
    float orientation[ORIENTATION_MAX_COUNT];
};

struct Descriptor
{
    float features[128];
};

// #define ANGLE_IS_NAN       0x01
// #define ZERO_HISTOGRAM     0x02
// #define SIGMA_NULL         0x04
// #define DESC_WINDOW_EMPTY  0x08
// #define HYPOT_OUT_OF_RANGE 0x10
// #define ATAN_OUT_OF_RANGE  0x20
// #define NAN_SOURCE_UNKNOWN 0x40

} // namespace popart
