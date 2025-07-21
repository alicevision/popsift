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

