/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once
#include "sift_pyramid.h"
#include "sift_octave.h"
#include "sift_extremum.h"
#include "common/debug_macros.h"

__global__
void ext_desc_notile( const int           octave,
                     cudaTextureObject_t texLinear );

                  //   1    -> 19.6 on 980 Ti
#define BLOCK_Z_NOTILE 2 // -> 19.5 on 980 Ti
                  //   3    -> 20.3 on 980 Ti
                  //   4    -> 19.6 on 980 Ti
                  //   8    -> 19.7 on 980 Ti

namespace popsift
{

inline static bool start_ext_desc_notile( const int octave, Octave& oct_obj )
{
    dim3 block;
    dim3 grid;

    block.x = 32;
    block.y = 4;
    block.z = BLOCK_Z_NOTILE;

    grid.x = grid_divide( hct.ori_ct[octave], block.z );
    grid.y = 1;
    grid.z = 1;

    if( grid.x == 0 ) return false;

    ext_desc_notile
        <<<grid,block,0,oct_obj.getStream()>>>
        ( octave,
          oct_obj.getDataTexLinear( ).tex );

    POP_SYNC_CHK;

    return true;
}

}; // namespace popsift
