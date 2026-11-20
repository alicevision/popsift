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

#include <sycl/sycl.hpp>

#include <cinttypes>
#include <cstdio>
#include <cmath>
#include <sstream>

namespace popsift
{
#ifdef PopSift_TEST_WITH_VL_ATAN2
/* VLFeat's vl_fast_atan2_f (vl/mathop.h), a polynomial approximation with
 * an error of up to about 0.35 degrees.
 * Meant for testing and found in TEST compile-time options.
 */
static inline
float vl_fast_atan2_f( float y, float x )
{
    const float c3    = 0.1821f;
    const float c1    = 0.9675f;
    const float abs_y = sycl::fabs( y ) + 1.19209290E-07F; // VL_EPSILON_F
    float angle;
    float r;
    if( x >= 0 ) {
        r     = ( x - abs_y ) / ( x + abs_y );
        angle = 0.785398163f;   // pi/4
    } else {
        r     = ( x + abs_y ) / ( abs_y - x );
        angle = 2.356194490f;   // 3*pi/4
    }
    angle += ( c3 * r * r - c1 ) * r;
    return ( y < 0 ) ? -angle : angle;
}
#endif

/* The gradient angle used for orientation and descriptor computation.
 * With the CMake option PopSift_TEST_WITH_VL_ATAN2 it is VLFeat's fast
 * approximation, which allows a closer comparison with VLFeat; otherwise
 * it is sycl::atan2.
 */
static inline
float popsift_atan2( float y, float x )
{
#ifdef PopSift_TEST_WITH_VL_ATAN2
    return vl_fast_atan2_f( y, x );
#else
    return sycl::atan2( y, x );
#endif
}

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
        theta    = popsift_atan2(dy, dx);
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
    const float dx = layer.get( level, y  , x+1 )
                   - layer.get( level, y  , x-1 );

    const float dy = layer.get( level, y+1, x   )
                   - layer.get( level, y-1, x   );

    grad     = hypotf( dx, dy ); // __fsqrt_rz(dx*dx + dy*dy);
    theta    = popsift_atan2(dy, dx);
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
    theta    = popsift_atan2(dy, dx);

    debug_ostr << std::setprecision(3)
               << "[" << ym_x << "..." << yp_x << ", " << y_xm << "..." << y_xp << "]->"
               << "(" << dy << "," << dx << ")->"
               << theta << " ";
}

}; // namespace popsift

