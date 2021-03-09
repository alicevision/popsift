/*
 * Copyright 2021, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#if POPSIFT_IS_DEFINED(POPSIFT_HAVE_COOPERATIVE_GROUPS)

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace popsift
{

/* A version of get_gradiant that works for a (32,1,1) threadblock
 * and pulls data to shared memory before computing. Data is pulled
 * less frequently, meaning that we do not rely on the L1 cache.
 */
__device__ static inline
void get_gradiant32( cg::thread_block_tile<32>& tile,
                     float&              grad,
                     float&              theta,
                     const int           x,
                     const int           y,
                     cudaTextureObject_t layer,
                     const int           level )
{
    const int idx = tile.thread_rank();

    __shared__ float x_array[34];

    for( int i=idx; i<34; i += tile.size() )
    {
        x_array[i] = readTex( layer, x+i-1.0f, y, level );
    }
    tile.sync();

    const float dx = x_array[idx+2]  - x_array[idx];

    const float dy = readTex( layer, x+idx, y+1.0f, level )
                   - readTex( layer, x+idx, y-1.0f, level );

    grad     = hypotf( dx, dy ); // __fsqrt_rz(dx*dx + dy*dy);
    theta    = atan2f(dy, dx);
}

}; // namespace popsift

#endif // POPSIFT_HAVE_COOPERATIVE_GROUPS

