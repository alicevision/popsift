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
namespace absoluteSource {

__global__ void horiz(cudaTextureObject_t src_point_texture, cudaSurfaceObject_t dst_data, int dst_level);

__global__ void vert(cudaTextureObject_t src_point_texture, cudaSurfaceObject_t dst_data, int dst_level);

__global__ void vert_abs0(cudaTextureObject_t src_point_texture, cudaSurfaceObject_t dst_data, int dst_level);

__global__ void vert_all_abs0(cudaTextureObject_t src_point_texture,
                              cudaSurfaceObject_t dst_data,
                              int start_level,
                              int max_level);

} // namespace absoluteSource
} // namespace gauss
} // namespace popsift

