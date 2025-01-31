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

template<class T>
void normalize_histogram( Grid g )
{
    g.resetBlock();
    while( g.nextBlock() )
    {
        Descriptor* descs            = dbuf.desc;
        const int   num_orientations = dct.ori_total;

        int offset = g.blockIdx.x * 32 + g.threadIdx.y;

        // all of these threads are useless
        if( g.blockIdx.x * 32 >= num_orientations ) return;

        offset = ( offset < num_orientations ) ? offset
                                               : num_orientations-1;
        Descriptor* desc = &descs[offset];

        bool ignoreme = ( offset >= num_orientations );

        g.resetThreadYZ();
        while( g.nextThreadYZ() )
        {
            T::normalize( desc->features, ignoreme );
        }
    }
}

