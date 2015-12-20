#pragma once

#include <stdint.h>
#include "c_util_img.hpp"
#include "keep_time.hpp"

namespace popart {

struct Image
{
    cudaStream_t stream;
    uchar*       array;     // pointer to array in device memory
    size_t       a_width;   // width  aligned to 128 byte
    size_t       a_height;  // height aligned to 128 byte
    size_t       pitch;     // perhaps CUDA has stricter align needs than 128
    size_t       u_width;   // unaligned width
    size_t       u_height;  // unaligned height
    size_t       type_size; // uchar or float

    /** Create a device-sided buffer of the given dimensions */
    Image( size_t w, size_t h, size_t type_size, cudaStream_t s );

    /** Create a device-sided buffer that can hold the given
     *  image and copy image to device.
     */
    Image( imgStream& gray, cudaStream_t s );

    ~Image( );

    /** src image must have type_size uchar,
     *  this image must have type_size float
     *  scalefactor is right now 2
     */
    void upscale( Image& src, size_t scalefactor );

    void debug_out( );
    void test_last_error( const char* file, int line );

    void download_and_save_array( const char* filename );

    void report_times( );

private:
    void upscale_v1( Image& src );
    void upscale_v2( Image& src );
    void upscale_v3( Image& src );
    void upscale_v4( Image& src );
    void upscale_v5( Image& src );

    KeepTime _keep_time_image_v1;
    KeepTime _keep_time_image_v2;
    KeepTime _keep_time_image_v3;
    KeepTime _keep_time_image_v4;
    KeepTime _keep_time_image_v5;
};

} // namespace popart
