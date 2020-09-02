/*
 * Copyright 2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "s_desc_norm_l2.h"
#include "s_desc_norm_rs.h"
#include "sift_extremum.h"

template<class T>
__global__
void normalize_histogram( )
{
    Descriptor* descs            = dbuf.desc;
    const int   num_orientations = dct.ori_total;

    int offset = blockIdx.x * 32 + threadIdx.y;

    // all of these threads are useless
    if( blockIdx.x * 32 >= num_orientations ) return;

    offset = ( offset < num_orientations ) ? offset
                                           : num_orientations-1;
    Descriptor* desc = &descs[offset];

    bool ignoreme = ( offset >= num_orientations );

    T::normalize( desc->features, ignoreme );
}

