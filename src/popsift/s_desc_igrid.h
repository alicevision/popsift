/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once
#include "common/debug_macros.h"
#include "sift_extremum.h"
#include "sift_octave.h"
#include "sift_pyramid.h"

/*
 * We assume that this is started with
 * block = 16,4,4 or with 32,4,4, depending on macros
 * grid  = nunmber of orientations
 */
__global__ void ext_desc_igrid(int octave, cudaTextureObject_t texLinear);

namespace popsift
{

#define IGRID_NUMLINES 1

inline static bool start_ext_desc_igrid( const int octave, Octave& oct_obj )
{
    dim3 block;
    dim3 grid;
    grid.x = hct.ori_ct[octave];
    grid.y = 1;
    grid.z = 1;

    if( grid.x == 0 ) return false;

    block.x = 16;
    block.y = 16;
    block.z = IGRID_NUMLINES;

    ext_desc_igrid
        <<<grid,block,0,oct_obj.getStream()>>>
        ( octave,
          oct_obj.getDataTexLinear( ).tex );

    POP_SYNC_CHK;

    return true;
}

}; // namespace popsift
