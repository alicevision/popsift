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
static inline
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

#if 0
/* get_gradiant() works for both point texture and linear interpolation
 * textures. The reason is that readTex must add 0.5 for coordinates in
 * both cases to access the expected pixel.
 */
static inline
void get_gradiant( float&              grad,
                   float&              theta,
                   const int           x,
                   const int           y,
                   cudaTextureObject_t layer,
                   const int           level )
{
    float dx = layer[level].ptr(y)[x+1]
             - layer[level].ptr(y)[x-1];
    float dy = layer[level].ptr(y+1)[x]
             - layer[level].ptr(y-1)[x];
    grad     = hypotf( dx, dy ); // __fsqrt_rz(dx*dx + dy*dy);
    theta    = atan2f(dy, dx);
}
#endif

/* A version of get_gradiant that works for a (32,1,1) threadblock
 * and pulls data to shared memory before computing. Data is pulled
 * less frequently, meaning that we do not rely on the L1 cache.
 */
static inline
void get_gradiant32( Grid& g,
                     float&              grad,
                     float&              theta,
                     const int           x,
                     const int           y,
                     cudaTextureObject_t layer,
                     const int           level )
{
    const int idx = g.threadIdx.x;

    const float dx = layer[level].ptr(y)[x+idx+1]
                   - layer[level].ptr(y)[x+idx-1];

    const float dy = layer[level].ptr(y+1)[x+idx]
                   - layer[level].ptr(y-1)[x+idx];

    grad     = hypotf( dx, dy ); // __fsqrt_rz(dx*dx + dy*dy);
    theta    = atan2f(dy, dx);
}

#if 0
static inline
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
#endif

}; // namespace popsift

