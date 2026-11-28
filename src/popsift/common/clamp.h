/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cstdint>

template<class T>
__device__ __host__
inline T clamp( T val, uint32_t maxval )
{
    return min( max( val, 0 ), maxval - 1 );
}

template<class T>
__device__ __host__
inline T clamp( T val, uint32_t minval, uint32_t maxval )
{
    return min( max( val, minval ), maxval - 1 );
}

