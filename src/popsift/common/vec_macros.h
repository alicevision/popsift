/*
 * Copyright 2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cmath>

__device__ static inline
float2 operator+( float2 l, const float2& r )
{
    l.x += r.x;
    l.y += r.y;
    return l;
}

__device__ static inline
float2 operator-( float2 l, const float2& r )
{
    l.x -= r.x;
    l.y -= r.y;
    return l;
}

__device__ static inline
float2 operator*( float2 l, float r )
{
    l.x *= r;
    l.y *= r;
    return l;
}

__device__ static inline
float2 operator*( float l, float2 r )
{
    r.x *= l;
    r.y *= l;
    return r;
}

__device__ static inline
float2 operator/( float2 l, float r )
{
    l.x /= r;
    l.y /= r;
    return l;
}

__device__ static inline
float2 round( float2 l )
{
    l.x = roundf( l.x );
    l.y = roundf( l.y );
    return l;
}

__device__ static inline
float2 abs( float2 l )
{
    l.x = fabsf( l.x );
    l.y = fabsf( l.y );
    return l;
}

