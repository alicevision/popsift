/*
 * Copyright 2016-2017, Simula Research Laboratory
 *           2018-2020, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once
#include "sift_octave.h"
#include "sift_pyramid.h"

namespace popsift
{

// Helper struct to manage device memory lifecycle
struct DescriptorDeviceMemory {
    Extremum* d_extrema;
    int* d_feat_to_ext_map;
    Descriptor* d_desc;
    sycl::queue* queue;
    
    DescriptorDeviceMemory(Extremum* ext, int* map, Descriptor* desc, sycl::queue* q)
        : d_extrema(ext), d_feat_to_ext_map(map), d_desc(desc), queue(q) {}
    
    void cleanup();  // Declared here, defined in .cc
};

// Async version - returns event and device memory for cleanup
std::pair<sycl::event, DescriptorDeviceMemory> start_ext_desc_vlfeat_async(
    const int octave, 
    Octave& oct_obj );


}; // namespace popsift

