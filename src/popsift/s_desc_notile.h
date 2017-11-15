/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once
#include "sift_octave.h"
#include "sift_extremum.h"
#include "sift_constants.h"
#include "s_gradiant.h"
#include "common/assist.h"
#include "common/vec_macros.h"

#define NUMLINES 2

/*
 * We assume that this is started with
 * block = 16,4,4 or with 32,4,4, depending on macros
 * grid  = nunmber of orientations
 */
namespace popsift
{
__global__
void ext_desc_notile( const int           octave,
                      cudaTextureObject_t texLinear );


inline static bool start_ext_desc_notile( const int octave, Octave& oct_obj )
{
    dim3 block;
    dim3 grid;
    grid.x = grid_divide( hct.ori_ct[octave], NUMLINES );
    grid.y = 1;
    grid.z = 1;

    if( grid.x == 0 ) return false;

    block.x = 32;
    block.y = 1;
    block.z = NUMLINES;

    ext_desc_notile
        <<<grid,block,0,oct_obj.getStream()>>>
        ( octave,
          oct_obj.getDataTexLinear( ).tex );

    return true;
}

}; // namespace popsift
