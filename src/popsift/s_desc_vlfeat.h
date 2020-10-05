/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once
#include "common/debug_macros.h"
#include "common/plane_2d.h"
#include "sift_extremum.h"
#include "sift_octave.h"
#include "sift_pyramid.h"

#undef BLOCK_3_DIMS

__global__ void ext_desc_vlfeat_print(int octave, cudaTextureObject_t layer_tex, int width, int height);
__global__ void ext_desc_vlfeat(int octave, cudaTextureObject_t layer_tex, int width, int height);

namespace popsift
{

inline static bool start_ext_desc_vlfeat( const int octave, Octave& oct_obj )
{
    dim3 block;
    dim3 grid;
    grid.x = hct.ori_ct[octave];
    grid.y = 1;
    grid.z = 1;

    if( grid.x == 0 ) return false;

#ifndef BLOCK_3_DIMS
    block.x = 1;
    block.y = 1;
    block.z = 1;
#else
    block.x = 1;
    block.y = 1;
    block.z = 1;
#endif

    std::cout << "Extracting " << grid.x << " descriptors for octave " << octave << std::endl;

    ext_desc_vlfeat_print
        <<<1,1,0,oct_obj.getStream()>>>
        ( octave,
          oct_obj.getDataTexPoint( ),
          oct_obj.getWidth(),
          oct_obj.getHeight() );
    cudaDeviceSynchronize();

    ext_desc_vlfeat
        <<<grid,block,0,oct_obj.getStream()>>>
        ( octave,
          oct_obj.getDataTexPoint( ),
          oct_obj.getWidth(),
          oct_obj.getHeight() );

    POP_SYNC_CHK;

    return true;
}

}; // namespace popsift

