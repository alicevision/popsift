/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/plane_2d.h"

namespace popsift {
namespace gauss {
namespace normalizedSource {

__global__ void horiz(cudaTextureObject_t src_data,
                      cudaSurfaceObject_t dst_data,
                      int dst_w,
                      int dst_h,
                      int octave,
                      float shift);

__global__ void horiz_level(cudaTextureObject_t src_linear_tex,
                            cudaSurfaceObject_t dst_data,
                            int dst_w,
                            int dst_h,
                            int /* octave - must be 0 */,
                            int level,
                            float shift);

__global__ void horiz_all(
  cudaTextureObject_t src_linear_tex, cudaSurfaceObject_t dst_data, int dst_w, int dst_h, float shift, int max_level);

} // namespace normalizedSource
} // namespace gauss
} // namespace popsift

