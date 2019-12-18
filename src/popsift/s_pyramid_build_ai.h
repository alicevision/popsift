/*
 * Copyright 2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/plane_2d.h"

namespace popsift {
namespace gauss {
namespace absoluteSourceInterpolated {

__global__
void horiz( cudaTextureObject_t src_linear_tex,
            cudaSurfaceObject_t dst_data,
            const int           dst_level );

__global__
void vert( cudaTextureObject_t src_linear_tex,
           cudaSurfaceObject_t dst_data,
           const int           dst_level );

__global__
void vert_abs0( cudaTextureObject_t src_linear_tex,
           cudaSurfaceObject_t dst_data,
           const int           dst_level );

__global__
void vert_all_abs0( cudaTextureObject_t src_linear_tex,
                    cudaSurfaceObject_t dst_data,
                    const int           start_level,
                    const int           max_level );

} // namespace absoluteSourceInterpolated
} // namespace gauss
} // namespace popsift

