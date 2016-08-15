/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "sift_constants.h"

namespace popsift {

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

} // namespace popsift
