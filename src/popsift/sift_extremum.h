/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <iostream>
#include <vector>

#include "sift_constants.h"

namespace popsift {

/* This is an internal data structure.
 * Separated from the final Extremum data structure to implement
 * grid filtering in a space-efficient manner. In grid filtering,
 * extrema are first found, after that some may be discarded in
 * some spatial regions of the image. Avoid waste of space by
 * allocating Extremum structures only for the remaining ones.
 */
struct InitialExtremum
{
    float xpos;
    float ypos;
    int   lpos;  // extremum refined into this level
    float sigma; // scale;
    int   cell;  // index into the grid for grid-based extrema filtering
    bool  ignore; // true if this extremum has been filtered
    int   write_index; // if any initial extrema are ignored, new index for Extremum
};

/* This is an internal data structure.
 * For performance reasons, it would be appropriate to split
 * the first 4 values from the rest of this structure. Right
 * now, descriptor computation is a bigger concern.
 */
struct Extremum
{
    float xpos;
    float ypos;
    int   lpos;  // extremum refined into this level
    float sigma; // scale;

    int   octave;  // belonging to this octave
    int   num_ori; // number of this extremum's orientations
    int   idx_ori; // exclusive prefix sum of the layer's orientations
    float orientation[ORIENTATION_MAX_COUNT];
};

/* This is a data structure that is returned to a calling program.
 * This is the SIFT descriptor itself.
 */
struct Descriptor
{
    float features[128];
};

} // namespace popsift
