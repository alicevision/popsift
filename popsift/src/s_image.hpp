#pragma once

#include <stdint.h>
#include "c_util_img.hpp"
#include "plane_2d.h"

namespace popart {

struct Image_uint8
{
    Plane2D_uint8 array;    // 2D plane allocated on device

    /** Allocate device-side buffer.
     */
    Image_uint8( short width, short height );

    /** Upload the grayscale image to the device.
     */
    void upload( imgStream& gray, cudaStream_t stream );

    /** Deallocate device-side buffer.
     */
    ~Image_uint8( );
};

struct Image
{
    cudaStream_t  stream;
    Plane2D_float array;    // 2D plane allocated on device
    size_t        u_width;   // unaligned width
    size_t        u_height;  // unaligned height

    /** Create a device-sided buffer of the given dimensions */
    Image( size_t w, size_t h, cudaStream_t s );

    ~Image( );

    /** src image must have type_size uchar,
     *  this image must have type_size float
     *  scalefactor is right now 2
     */
    void upscale( Image_uint8& src, size_t scalefactor );

    void debug_out( );
    void test_last_error( const char* file, int line );

    void download_and_save_array( const char* filename );

private:
    void upscale_v1( Image_uint8& src );
    void upscale_v2( Image_uint8& src );
    void upscale_v3( Image_uint8& src );
    void upscale_v4( Image_uint8& src );
};

} // namespace popart
