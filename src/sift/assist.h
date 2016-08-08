#pragma once

#include <cuda_runtime.h>
#include <iostream>

using namespace std;

std::ostream& operator<<( std::ostream& ostr, const dim3& p );

namespace popart
{

/* This computation is needed very frequently when a dim3 grid block is
 * initialized. It ensure that the tail is not forgotten.
 */
__device__ __host__
inline int grid_divide( int size, int divider )
{
    return size / divider + ( size % divider != 0 ? 1 : 0 );
}

}; // namespace popart

