#pragma once

#include "sift_constants.h"

namespace popart {

/* For performance reasons, it would be appropriate to split
 * the first 3 floats from the rest of this structure. Right
 * now, descriptor computation is a bigger concern.
 */
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

} // namespace popart
