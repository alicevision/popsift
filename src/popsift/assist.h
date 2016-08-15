/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cuda_runtime.h>
#include <iostream>

using namespace std;

std::ostream& operator<<( std::ostream& ostr, const dim3& p );

namespace popsift
{

/* This computation is needed very frequently when a dim3 grid block is
 * initialized. It ensure that the tail is not forgotten.
 */
__device__ __host__
inline int grid_divide( int size, int divider )
{
    return size / divider + ( size % divider != 0 ? 1 : 0 );
}

}; // namespace popsift

