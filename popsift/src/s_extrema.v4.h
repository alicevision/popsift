#pragma once

#include <inttypes.h>
#include "s_pyramid.hpp"

using namespace popart;

/*************************************************************
 * V5: device side
 *************************************************************/

__global__
void find_extrema_in_dog_v4( float*             dog_upper,
                             float*             dog_here,
                             float*             dog_lower,
                             float              edge_limit,
                             float              threshold,
                             const uint32_t     width,
                             const uint32_t     pitch,
                             const uint32_t     height,
                             const uint32_t     level,
                             const uint32_t     maxlevel,
                             ExtremaMgmt*       mgmt_array,
                             ExtremumCandidate* d_extrema );

__global__
void fix_extrema_count_v4( ExtremaMgmt* mgmt_array, uint32_t mgmt_level );

