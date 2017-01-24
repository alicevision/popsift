/*
* Copyright 2017, Simula Research Laboratory
*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include "sift_conf.h"
#include "sift_extremum.h"
#include "popsift.h"
#include <vector>

class PopSift;

namespace popsift {

// Flatten descriptors from all octaves to a contiguous block on the device.
Descriptor* FlattenDescriptorsAsyncD(PopSift& ps);

// Create a descriptor -> extrema map.
std::vector<unsigned> CreateFeatureToExtremaMap(PopSift& ps);

struct U8Descriptor {
    unsigned char features[128];
};

// Converts float descriptors to unsigned char by multiplying floats by 512.
U8Descriptor* ConvertDescriptorsToU8(Descriptor* d_descriptors, int count);

// Returns vector v indexed by descriptors in fa s.t. v[i] is the best matching descriptor in fb.
// If a descriptor in fa has no match in fb, v[i] == -1.
std::vector<int> Matching_CPU(const Features& fa, const Features& fb);

class Matching
{
public:
    Matching(Config& config);
    ~Matching();

    std::vector<int> Match(popsift::Descriptor* d_desc_a, size_t num_desc_a, 
                  popsift::Descriptor* d_desc_b, size_t num_desc_b);

private:
    const Config& config;
};

}
