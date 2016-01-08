#pragma once

#include <inttypes.h>
#include "plane_2d.h"

/*
 * The Bemap code has a massive overhead by computing grad and
 * theta for every pixel at every level at every octave, even
 * though only very few are ever needed.
 * The Celebrandil code computes grad and theta on demand, which
 * is better, but it computes always from the unsmoothed top
 * layer of an octave. That is not in the spirit of the Lowe
 * paper.
 */
__device__
inline void get_gradiant( float&         grad,
                          float&         theta,
                          uint32_t       x,
                          uint32_t       y,
                          popart::Plane2D_float& layer )
{
    grad  = 0.0f;
    theta = 0.0f;
    if( x > 0 && x < layer.getCols()-1 && y > 0 && y < layer.getRows()-1 ) {
        float dx = layer.ptr(y)[x+1] - layer.ptr(y)[x-1];
        float dy = layer.ptr(y-1)[x] - layer.ptr(y+1)[x];
        grad     = __fsqrt_rz(dx*dx + dy*dy);
        theta    = atan2f(dy, dx);
    }
}

