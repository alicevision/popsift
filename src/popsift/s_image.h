/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <stdint.h>
#include "common/plane_2d.h"
#include "sift_conf.h"

namespace popsift {

/*************************************************************
 * ImageBase
 *************************************************************/

struct ImageBase
{
    ImageBase( );

    /** Create a device-sided buffer of the given dimensions */
    ImageBase( int w, int h );

    virtual ~ImageBase( );

    /** Reallocation that takes care of pitch/step when new dimensions
     *  are smaller and actually reallocation when they are bigger.
     */
    virtual void resetDimensions( int w, int h ) = 0;

    /* This loading function copies all image data to a local
     * buffer that is pinned in memory. We should offer two
     * other functions: one that take a device-sided buffer
     * if the image is already uploaded, and one that takes
     * an image in pinned memory.
     */
    virtual void load( void* input ) = 0;

    inline cudaTextureObject_t& getInputTexture() {
        return _input_image_tex;
    }

    inline int getWidth()  const { return _w; }
    inline int getHeight() const { return _h; }

private:
    virtual void allocate( int w, int h ) = 0;
    virtual void createTexture( ) = 0;
    virtual void destroyTexture( ) = 0;

protected:
    int _w;     // width  of current image
    int _h;     // height of current image
    int _max_w; // allocated width  of image
    int _max_h; // allocated height of image

    /* Texture information for input image on device */
    cudaTextureObject_t _input_image_tex;
    cudaTextureDesc     _input_image_texDesc;
    cudaResourceDesc    _input_image_resDesc;
};

/*************************************************************
 * Image
 *************************************************************/

struct Image : public ImageBase
{
    Image( );

    /** Create a device-sided buffer of the given dimensions */
    Image( int w, int h );

    virtual ~Image( );

    /** Reallocation that takes care of pitch/step when new dimensions
     *  are smaller and actually reallocation when they are bigger.
     */
    virtual void resetDimensions( int w, int h );

    /* This loading function copies all image data to a local
     * buffer that is pinned in memory. We should offer two
     * other functions: one that take a device-sided buffer
     * if the image is already uploaded, and one that takes
     * an image in pinned memory.
     */
    virtual void load( void* input );

private:
    void allocate( int w, int h );
    void createTexture( );
    void destroyTexture( );

private:
    /* 2D plane holding input image on host for uploading
     * to device. */
    Plane2D_uint8 _input_image_h;

    /* 2D plane holding input image on device for upscaling */
    Plane2D_uint8 _input_image_d;
};

/*************************************************************
 * ImageFloat
 *************************************************************/

struct ImageFloat : public ImageBase
{
    ImageFloat( );

    /** Create a device-sided buffer of the given dimensions */
    ImageFloat( int w, int h );

    virtual ~ImageFloat( );

    /** Reallocation that takes care of pitch/step when new dimensions
     *  are smaller and actually reallocation when they are bigger.
     */
    virtual void resetDimensions( int w, int h );

    /* This loading function copies all image data to a local
     * buffer that is pinned in memory. We should offer two
     * other functions: one that take a device-sided buffer
     * if the image is already uploaded, and one that takes
     * an image in pinned memory.
     */
    virtual void load( void* input );

private:
    void allocate( int w, int h );
    void createTexture( );
    void destroyTexture( );

private:
    /* 2D plane holding input image on host for uploading
     * to device. */
    Plane2D_float _input_image_h;

    /* 2D plane holding input image on device for upscaling */
    Plane2D_float _input_image_d;
};

} // namespace popsift
