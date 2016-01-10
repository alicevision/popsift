#pragma once

#include <stdint.h>
#include "c_util_img.h"
#include "plane_2d.h"

namespace popart {

struct Image
{
    Plane2D_float array;    // 2D plane allocated on device

    /** Create a device-sided buffer of the given dimensions */
    Image( size_t w, size_t h );

    ~Image( );

    /** src image must have type_size uchar,
     *  this image must have type_size float
     *  scalefactor is right now 2
     */
    void upscale( Plane2D_uint8 &       src,
                  cudaTextureObject_t & tex,
                  size_t                scalefactor,
                  cudaStream_t          s );

    void debug_out( );
    void test_last_error( const char* file, int line );

    void download_and_save_array( const char* filename );

private:
    void upscale_v1( Plane2D_uint8 & src, cudaStream_t stream );
    void upscale_v2( Plane2D_uint8 & src, cudaStream_t stream );
    void upscale_v3( Plane2D_uint8 & src, cudaStream_t stream );
    void upscale_v4( Plane2D_uint8 & src, cudaStream_t stream );
    void upscale_v5( cudaTextureObject_t & tex, cudaStream_t stream );
};

} // namespace popart
