/*
 * Copyright 2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "common/grid.h"
#include "s_desc_norm_l2.h"
#include "s_desc_norm_rs.h"
#include "sift_extremum.h"
#include <sycl/sycl.hpp>

// Host wrapper for SYCL normalization
template<class T>
sycl::event normalize_histogram_sycl(sycl::queue& q, Descriptor* d_descs, const int num_orientations)
{
   // Use tag dispatch to select correct overload
   T* tag = nullptr;
   return normalize_descriptors_sycl(q, d_descs, num_orientations, h_consts.norm_multi, tag);
}

// CPU fallback (original implementation)
template<class T>
void normalize_histogram( )
{
    Descriptor* descs            = dbuf.desc;
    const int   num_orientations = dct.ori_total;

    for( int i=0; i<num_orientations; i++ )
    {
        Descriptor* desc = &descs[i];

        T::normalize( desc->features );
    }
}

