#pragma once

#include <inttypes.h>
#include "s_pyramid.hpp"

using namespace popart;

/*
 * Compute the keypoint orientations for each extremum
 * using 16 threads for each of them.
 */
__global__
void compute_keypoint_orientations_v1( ExtremumCandidate* extremum,
                                       ExtremaMgmt*       mgmt_array,
                                       uint32_t           mgmt_level,
                                       const float*       layer,
                                       int                layer_pitch,
                                       int                layer_height );

