/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

__device__ __host__
inline int clamp( int val, int maxval )
{
    return min( max( val, 0 ), maxval - 1 );
}

__device__ __host__
inline int clamp( int val, int minval, int maxval )
{
    return min( max( val, minval ), maxval - 1 );
}

