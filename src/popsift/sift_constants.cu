/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/debug_macros.h"
#include "sift_constants.h"

#include <cuda_runtime.h>

#include <iostream>

using namespace std;

namespace popsift {

thread_local            ConstInfo h_consts;
__device__ __constant__ ConstInfo d_consts;

void init_constants( float sigma0, int levels, float threshold, float edge_limit, int max_extrema, int normalization_multiplier )
{
    cudaError_t err;

    h_consts.sigma0           = sigma0;
    h_consts.sigma_k          = powf(2.0f, 1.0f / levels );
    h_consts.edge_limit       = edge_limit;
    h_consts.threshold        = threshold;
    h_consts.max_extrema      = max_extrema;
    h_consts.max_orientations = max_extrema + max_extrema/4;
    h_consts.norm_multi       = normalization_multiplier;

    float dn_step = 1.0f / 8.0f;
    float dn_base = 0.5f * dn_step - 20.0f * dn_step;
    for( int y=0; y<40; y++ ) {
        for( int x=0; x<40; x++ ) {
            float dnx = dn_base + x * dn_step;
            float dny = dn_base + y * dn_step;
            h_consts.desc_gauss[y][x] = expf( -scalbnf(dnx*dnx + dny*dny, -3));
        }
    }

    for( int i=0; i<16; i++ ) {
        const float nx = -1.0f + 1.0f/16.0f + i * 1.0f/8.0f;
        h_consts.desc_tile[i] = 1.0f - fabs(nx);
    }

    err = cudaMemcpyToSymbol( d_consts, &h_consts,
                              sizeof(ConstInfo), 0,
                              cudaMemcpyHostToDevice );
    POP_CUDA_FATAL_TEST( err, "Failed to upload h_consts to device: " );
}

} // namespace popsift

