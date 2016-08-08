/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cuda_runtime.h>

#include "sift_constants.h"
#include "debug_macros.h"

namespace popart {

ConstInfo                         h_consts;
__device__ __constant__ ConstInfo d_consts;

void init_constants( float sigma0, int levels, float threshold, float edge_limit, int max_extrema )
{
    cudaError_t err;

    h_consts.sigma0       = sigma0;
    h_consts.sigma_k      = powf(2.0f, 1.0f / levels );
    h_consts.edge_limit   = edge_limit;
    h_consts.threshold    = threshold;
    h_consts.extrema      = max_extrema;
    h_consts.orientations = max_extrema + max_extrema/4;

    err = cudaMemcpyToSymbol( d_consts, &h_consts,
                              sizeof(ConstInfo), 0,
                              cudaMemcpyHostToDevice );
    POP_CUDA_FATAL_TEST( err, "Failed to upload h_consts to device: " );
}

} // namespace iopart

