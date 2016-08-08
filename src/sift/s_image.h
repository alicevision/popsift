/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <stdint.h>
#include "plane_2d.h"
#include "sift_conf.h"

namespace popart {

struct Image
{
    /** Create a device-sided buffer of the given dimensions */
    Image( size_t w, size_t h );

    ~Image( );

    /* This loading function copies all image data to a local
     * buffer that is pinned in memory. We should offer two
     * other functions: one that take a device-sided buffer
     * if the image is already uploaded, and one that takes
     * an image in pinned memory.
     */
    void load( const Config& conf, const unsigned char* input );

    void debug_out( );

    inline cudaTextureObject_t& getInputTexture() {
        return _input_image_tex;
    }

private:
    int _w;
    int _h;

    /* 2D plane holding input image on host for uploading
     * to device. */
    Plane2D_uint8 _input_image_h;

    /* 2D plane holding input image on device for upscaling */
    Plane2D_uint8 _input_image_d;

    /* Texture information for input image on device */
    cudaTextureObject_t _input_image_tex;
    cudaTextureDesc     _input_image_texDesc;
    cudaResourceDesc    _input_image_resDesc;
};

} // namespace popart
