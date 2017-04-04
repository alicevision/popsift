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
namespace relativeSource {

__global__
void horiz( cudaTextureObject_t src_data,
            cudaSurfaceObject_t dst_data,
            const int           dst_w,
            const int           dst_h,
            int                 octave,
            float               shift );

} // namespace relativeSource
} // namespace gauss
} // namespace popsift

