#pragma once

#include "sift_constants.h"

namespace popart {

struct Extremum
{
    float xpos;
    float ypos;
    float sigma; // scale;
    float orientation;

#ifdef DEBUG_ORI_GRID
    int grid_x_min;
    int grid_y_min;
    int grid_x_max;
    int grid_y_max;
#endif // DEBUG_ORI_GRID
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
