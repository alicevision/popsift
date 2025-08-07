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
#include "common/grid.h"
#include "sift_constants.h"

#include <cinttypes>
#include <cstdio>
#include <cmath>
#include <sstream>

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
void get_gradiant( float&                        grad,
                   float&                        theta,
                   int                           x,
                   int                           y,
                   const popsift::Plane2D_float& layer )
{
    grad  = 0.0f;
    theta = 0.0f;
    if( x > 0 && x < layer.getDimX()-1 && y > 0 && y < layer.getDimY()-1 ) {
        float dx = layer.get(y  ,x+1) - layer.get(y  ,x-1);
        float dy = layer.get(y+1,x  ) - layer.get(y-1,x  );
        grad     = hypotf( dx, dy ); // __fsqrt_rz(dx*dx + dy*dy);
        theta    = atan2f(dy, dx);
    }
}

/* A version of get_gradiant that works for a (32,1,1) threadblock
 * and pulls data to shared memory before computing. Data is pulled
 * less frequently, meaning that we do not rely on the L1 cache.
 */
static inline
void get_gradiant32( float&               grad,
                     float&               theta,
                     const int            x,
                     const int            y,
                     const Plane2D_float& layer,
                     const int            level )
{
    const float dx = layer.get( level, y  , x+1 );
                   - layer.get( level, y  , x-1 );

    const float dy = layer.get( level, y+1, x   );
                   - layer.get( level, y-1, x   );

    grad     = hypotf( dx, dy ); // __fsqrt_rz(dx*dx + dy*dy);
    theta    = atan2f(dy, dx);
}

static inline
void get_gradiant32( float&               grad,
                     float&               theta,
                     const int            x,
                     const int            y,
                     const Plane2D_float& layer,
                     const int            level,
                     std::ostringstream&  debug_ostr )
{
    const float y_xp = layer.get( level, y  , x+1 );
    const float y_xm = layer.get( level, y  , x-1 );
    const float yp_x = layer.get( level, y+1, x   );
    const float ym_x = layer.get( level, y-1, x   );

    const float dx = y_xp - y_xm;
    const float dy = yp_x - ym_x;

    grad     = hypotf( dx, dy ); // __fsqrt_rz(dx*dx + dy*dy);
    theta    = atan2f(dy, dx);

    debug_ostr << std::setprecision(3)
               << "[" << ym_x << "..." << yp_x << ", " << y_xm << "..." << y_xp << "]->"
               << "(" << dy << "," << dx << ")->"
               << theta << " ";
}

}; // namespace popsift

