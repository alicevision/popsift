#pragma once

#include <stdint.h>
#include "c_util_img.h"
#include "plane_2d.h"
#include "sift_conf.h"

namespace popart {

struct Image
{
    /** Create a device-sided buffer of the given dimensions */
    Image( size_t w, size_t h );

    ~Image( );

    /** Load a new image, copy to device and upscale */
    void load( const Config& conf, const imgStream& inp );

    void debug_out( );
    void test_last_error( const char* file, int line );

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
