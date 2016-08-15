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
 * For performance reasons, it would be appropriate to split
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

/* This is a data structure that is returned to a calling program.
 * This is the SIFT descriptor itself.
 */
struct Descriptor
{
    float features[128];
};

/* This is a data structure that is returned to a calling program.
 * The xpos/ypos information in feature is scale-adapted.
 * Note that these data structures are filled host-size in a loop
 * because addresses are not shared between CPU and GPU.

 */
struct Feature
{
    float       xpos;
    float       ypos;
    float       sigma; // scale;
    int         num_ori; // number of this extremum's orientations
    Descriptor* desc[ORIENTATION_MAX_COUNT];

};

std::ostream& operator<<( std::ostream& ostr, const Feature& feature );

/* This is a data structure that is returned to a calling program.
 * _desc_buffer is a transparent flat memory holding descriptors
 * that are referenced by the extrema.
 *
 * Note that the current data structures do no allow to match
 * Descriptors in the transparent array with there extrema except
 * for brute force.
 */
class Features
{
    typedef std::vector<Feature>::iterator       F_iterator;
    typedef std::vector<Feature>::const_iterator F_const_iterator;

    std::vector<Feature> _features;
    Descriptor*          _desc_buffer;

public:
    Features( );
    ~Features( );

    inline F_iterator       begin()       { return _features.begin(); }
    inline F_const_iterator begin() const { return _features.begin(); }
    inline F_iterator       end()         { return _features.end(); }
    inline F_const_iterator end() const   { return _features.end(); }

protected:
    friend class Pyramid;
};

std::ostream& operator<<( std::ostream& ostr, const Features& feature );

} // namespace popsift
