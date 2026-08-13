/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

/* The texture and surface handle types the pyramid hands to the kernels.
 */

#include <cuda_runtime.h>

namespace popsift {

/* A texture object created with linear filtering, so that a fetch through it is
 * distinguishable from a fetch through a point-filtered one at the call site.
 */
struct LinearTexture
{
    cudaTextureObject_t tex;
};

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
/* The read handle for a pyramid array, passed to the readTex overloads in
 * common/assist.h.
 *
 * On HIP the pyramid arrays are not layered. Observed on gfx90a and on gfx1100
 * (ROCm 7.2.1): once a layered array has been written layer by layer through
 * surf2DLayeredwrite, a read in a later kernel launch returns the last written
 * layer for every layer index, through tex2DLayered, through surf2DLayeredread
 * and through hipMemcpy3D alike. Filed as ROCm/clr#275. The defect is on the
 * write side: surf2DLayeredwrite passed the layer index in the mipmap level
 * slot, so every layer landed in the same slot. ROCm/rocm-systems#6683 corrects
 * that, and with it all three read paths return correct per-layer data, but it
 * is not in ROCm 7.2.x. A plain 3D array is coherent across launches, so the
 * arrays are allocated non-layered (sift_octave.cu), the layer index becomes the
 * z coordinate, and the read path samples the surface. That needs the surface
 * and the level dimensions alongside the texture, which is what this type
 * carries.
 *
 * On CUDA the handle is the texture object itself and the read path is
 * unchanged.
 */
struct LayeredReadTex
{
    cudaTextureObject_t tex;
    cudaSurfaceObject_t surf;
    int                 width;
    int                 height;
};

inline LayeredReadTex makeLayeredReadTex( cudaTextureObject_t tex,
                                          cudaSurfaceObject_t surf,
                                          int width, int height )
{
    return LayeredReadTex{ tex, surf, width, height };
}
#else
using LayeredReadTex = cudaTextureObject_t;

inline LayeredReadTex makeLayeredReadTex( cudaTextureObject_t tex,
                                          cudaSurfaceObject_t,
                                          int, int )
{
    return tex;
}
#endif

} // namespace popsift
