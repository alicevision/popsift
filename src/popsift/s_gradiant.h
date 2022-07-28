/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "common/assist.h"
#include "common/plane_2d.h"
#include "sift_constants.h"

#include <cinttypes>
#include <cstdio>

#include "popsift/sift_config.h"

#include "s_gradiant_cuda9plus.h" // functions that work with CUDA SDK 9 and later

namespace popsift
{
/*
 * We are wasting time by computing gradiants on demand several
 * times. We could precompute gradiants for all pixels once, as
 * other code does, but the number of features should be too low
 * to make that feasible. So, we take this performance hit.
 * Especially punishing in the descriptor computation.
 *
 * Also, we are always computing from the closest blur level
 * as Lowe expects us to do. Other implementations compute the
 * gradiant always from the original image, which we think is
 * not in the spirit of the hierarchy is blur levels. That
 * assumption would only hold if we could simply downscale to
 * every first level of every octave ... which is not compatible
 * behaviour.
 */
__device__ static inline
void get_gradiant( float& grad,
                   float& theta,
                   int    x,
                   int    y,
                   popsift::Plane2D_float& layer )
{
    grad  = 0.0f;
    theta = 0.0f;
    if( x > 0 && x < layer.getCols()-1 && y > 0 && y < layer.getRows()-1 ) {
        float dx = layer.ptr(y)[x+1] - layer.ptr(y)[x-1];
        float dy = layer.ptr(y+1)[x] - layer.ptr(y-1)[x];
        grad     = hypotf( dx, dy ); // __fsqrt_rz(dx*dx + dy*dy);
        theta    = atan2f(dy, dx);
    }
}

/* get_gradiant() works for both point texture and linear interpolation
 * textures. The reason is that readTex must add 0.5 for coordinates in
 * both cases to access the expected pixel.
 */
__device__ static inline
void get_gradiant( float&              grad,
                   float&              theta,
                   const int           x,
                   const int           y,
                   cudaTextureObject_t layer,
                   const int           level )
{
    float dx = readTex( layer, x+1.0f, y, level )
             - readTex( layer, x-1.0f, y, level );
    float dy = readTex( layer, x, y+1.0f, level )
             - readTex( layer, x, y-1.0f, level );
    grad     = hypotf( dx, dy ); // __fsqrt_rz(dx*dx + dy*dy);
    theta    = atan2f(dy, dx);
}

#if POPSIFT_IS_UNDEFINED(POPSIFT_HAVE_COOPERATIVE_GROUPS)
/* A version of get_gradiant that works for a (32,1,1) threadblock
 * and pulls data to shared memory before computing. Data is pulled
 * less frequently, meaning that we do not rely on the L1 cache.
 */
__device__ static inline
void get_gradiant32( float&              grad,
                     float&              theta,
                     const int           x,
                     const int           y,
                     cudaTextureObject_t layer,
                     const int           level )
{
    const int idx = threadIdx.x;

    __shared__ float x_array[34];

    for( int i=idx; i<34; i += blockDim.x )
    {
        x_array[i] = readTex( layer, x+i-1.0f, y, level );
    }
    __syncthreads();

    const float dx = x_array[idx+2]  - x_array[idx];

    const float dy = readTex( layer, x+idx, y+1.0f, level )
                   - readTex( layer, x+idx, y-1.0f, level );

    grad     = hypotf( dx, dy ); // __fsqrt_rz(dx*dx + dy*dy);
    theta    = atan2f(dy, dx);
}
#endif

__device__ static inline
void get_gradiant( float&              grad,
                   float&              theta,
                   float               x,
                   float               y,
                   float               cos_t,
                   float               sin_t,
                   cudaTextureObject_t texLinear,
                   int                 level )
{
    float dx = readTex( texLinear, x+cos_t, y+sin_t, level )
             - readTex( texLinear, x-cos_t, y-sin_t, level );
    float dy = readTex( texLinear, x-sin_t, y+cos_t, level )
             - readTex( texLinear, x+sin_t, y-cos_t, level );
    grad     = hypotf( dx, dy );
    theta    = atan2f( dy, dx );
}

}; // namespace popsift

