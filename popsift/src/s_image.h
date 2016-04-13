#pragma once

#include <stdint.h>
#include "c_util_img.h"
#include "plane_2d.h"

namespace popart {

struct Image
{
    /** Create a device-sided buffer of the given dimensions */
    Image( size_t w, size_t h );

    ~Image( );

    /** src image must have type_size uchar,
     *  this image must have type_size float
     *  scalefactor is right now 2
     */
    void upscale( Plane2D_uint8&       src,
                  cudaTextureObject_t& tex,
                  float                scalefactor );

    void debug_out( );
    void test_last_error( const char* file, int line );

    // void download_and_save_array( const char* filename );

    inline Plane2D_float& getUpscaledImage() {
        return _upscaled_image_d;
    }

private:
    void upscale_v5( cudaTextureObject_t & tex );

    /** 2D plane holding upscaled image, allocated on device
     */
    Plane2D_float _upscaled_image_d;
};

} // namespace popart
