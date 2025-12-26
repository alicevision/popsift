/*
 * Copyright 2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "sift_extremum.h"
#include <sycl/sycl.hpp>

namespace popsift {

// Forward declarations
class NormalizeRootSift;
class NormalizeL2;

// CPU fallback (original implementation)
template<class T>
void normalize_histogram(Descriptor* descs, const int num_orientations)
{
    for(int i = 0; i < num_orientations; i++)
    {
        Descriptor* desc = &descs[i];
        T::normalize(desc->features);
    }
}

// Forward declare the tag dispatch functions (implemented in norm classes)
sycl::event normalize_descriptors_sycl(
    sycl::queue& q,
    Descriptor* d_descs,
    const int num_orientations,
    const int norm_multi,
    NormalizeRootSift*);

sycl::event normalize_descriptors_sycl(
    sycl::queue& q,
    Descriptor* d_descs,
    const int num_orientations,
    const int norm_multi,
    NormalizeL2*);

// Host wrapper for SYCL normalization
template<class T>
sycl::event normalize_histogram_sycl(
    sycl::queue& q, 
    Descriptor* d_descs, 
    const int num_orientations,
    float norm_multi,
    bool use_sub_group = true)
{
    // Use tag dispatch to call the appropriate kernel in the norm class
    T* tag = nullptr;
    return normalize_descriptors_sycl(q, d_descs, num_orientations, 
                                      static_cast<int>(norm_multi), tag);
}

} // namespace popsift

// Include the implementations AFTER the declarations
#include "s_desc_norm_l2.h"
#include "s_desc_norm_rs.h"