/*
 * Copyright 2016-2017, Simula Research Laboratory
 *           2018-2024, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/grid.h"
#include "common/plane_2d.h"
#include "gauss_filter.h"
#include "sift_pyramid.h"
#include "sift_constants.h"

#include "common/write_plane_2d.h" // debug

#include <cmath>

namespace popsift {

void Pyramid::horiz_from_input_image( const Config& conf, std::shared_ptr<ImageBase> base )
{    
    POP_INFO2( conf.silent(), "enter " << __PRETTY_FUNCTION__ );

    Octave& oct_obj = _octaves[0];

    if( base->isNull() )
    {
        std::cerr << __FILE__ << ":" << __LINE__ << ": programming error," << std::endl
                  << "source plane is NULL in " << __FUNCTION__ << std::endl;
        assert( 0 );
    }
    if( oct_obj.getIntm().isNull() )
    {
        std::cerr << __FILE__ << ":" << __LINE__ << ": programming error," << std::endl
                  << "dest plane is NULL in " << __FUNCTION__ << std::endl;
        assert( 0 );
    }

    float shift = 0.5f * std::pow( 2.0f, conf.getUpscaleFactor() );

    // Get source and destination planes
    PlaneD<float>& src = base->getFloatPlane();
    PlaneD<float>& dst = oct_obj.getIntm();

    // Get dimensions
    const int dst_w = dst.getDimX();
    const int dst_h = dst.getDimY();
    const int dst_z = dst.getDimZ();
    const int src_w = src.getDimX();
    const int src_h = src.getDimY();
    
    const int src_pitch = src.getPitch();
    const int dst_pitch = dst.getPitch();

    POP_INFO2( conf.silent(), "  dst: " << dst_w << "x" << dst_h << "x" << dst_z << " layers, pitch=" << dst_pitch );
    POP_INFO2( conf.silent(), "  src: " << src_w << "x" << src_h << ", pitch=" << src_pitch );

    // Get Gaussian filter parameters
    const int span = h_gauss.dd.span[0];
    
    // Copy filter to host memory
    std::vector<float> filter_host(span + 1);
    for(int i = 0; i <= span; i++) {
        filter_host[i] = h_gauss.dd.filter[i];
    }

    // Get SYCL queue from octave
    sycl::queue& queue = oct_obj.getQueue();

    POP_INFO2( conf.silent(), "Submitting horiz_from_input_image SYCL kernel..." );

    try {
        // FIX: src is HOST memory, need to copy to device!
        float* src_host_ptr = src.getHostPtr();  // Get host pointer
        float* dst_device_ptr = dst.getDevicePtr();

        if (!src_host_ptr || !dst_device_ptr) {
            throw std::runtime_error("NULL pointer in horiz_from_input_image");
        }

        const int src_total_floats = src_pitch * src_h;
        const int dst_total_floats = dst_pitch * dst_h * dst_z;

        POP_INFO2( conf.silent(), "  src total floats: " << src_total_floats << " (" << (src_total_floats * 4) << " bytes)" );
        POP_INFO2( conf.silent(), "  dst total floats: " << dst_total_floats << " (" << (dst_total_floats * 4) << " bytes)" );

        // Allocate device memory for src
        POP_INFO2( conf.silent(), "  Allocating device memory for src..." );
        float* src_device_ptr = sycl::malloc_device<float>(src_total_floats, queue);
        
        // Copy src from host to device
        POP_INFO2( conf.silent(), "  Copying src to device..." );
        queue.memcpy(src_device_ptr, src_host_ptr, src_total_floats * sizeof(float)).wait();

        // Allocate filter on device
        float* d_filter = sycl::malloc_device<float>(span + 1, queue);
        queue.memcpy(d_filter, filter_host.data(), (span + 1) * sizeof(float)).wait();

        POP_INFO2( conf.silent(), "  ✓ Data uploaded, submitting kernel..." );

        // Launch kernel
        auto event = queue.submit([&](sycl::handler& cgh) {
            const int c_dst_w = dst_w;
            const int c_dst_h = dst_h;
            const int c_src_w = src_w;
            const int c_src_h = src_h;
            const int c_src_pitch = src_pitch;
            const int c_dst_pitch = dst_pitch;
            const int c_span = span;

            cgh.parallel_for(
                sycl::range<2>(dst_h, dst_w),
                [=](sycl::id<2> idx) {
                    const int write_x = idx[1];
                    const int write_y = idx[0];

                    if (write_x >= c_dst_w || write_y >= c_dst_h) return;

                    const float read_x = float(write_x * c_src_w) / float(c_dst_w);
                    const float read_y = float(write_y * c_src_h) / float(c_dst_h);

                    auto sample_bilinear = [=](float fy, float fx) -> float {
                        fx = sycl::clamp(fx, 0.0f, float(c_src_w - 1));
                        fy = sycl::clamp(fy, 0.0f, float(c_src_h - 1));

                        int x0 = int(fx);
                        int y0 = int(fy);
                        int x1 = sycl::min(x0 + 1, c_src_w - 1);
                        int y1 = sycl::min(y0 + 1, c_src_h - 1);

                        float tx = fx - float(x0);
                        float ty = fy - float(y0);

                        float v00 = src_device_ptr[y0 * c_src_pitch + x0];
                        float v10 = src_device_ptr[y0 * c_src_pitch + x1];
                        float v01 = src_device_ptr[y1 * c_src_pitch + x0];
                        float v11 = src_device_ptr[y1 * c_src_pitch + x1];

                        float v0 = v00 * (1.0f - tx) + v10 * tx;
                        float v1 = v01 * (1.0f - tx) + v11 * tx;

                        return v0 * (1.0f - ty) + v1 * ty;
                    };

                    float out = 0.0f;

                    for(int offset = c_span; offset > 0; offset--) {
                        const float g = d_filter[offset];
                        const float offrel = float(offset * c_src_w) / float(c_dst_w);
                        
                        const float v1 = sample_bilinear(read_y, read_x - offrel);
                        const float v2 = sample_bilinear(read_y, read_x + offrel);
                        
                        out += (v1 + v2) * g;
                    }

                    const float g0 = d_filter[0];
                    const float v3 = sample_bilinear(read_y, read_x);
                    out += v3 * g0;

                    const int dst_idx = write_y * c_dst_pitch + write_x;
                    dst_device_ptr[dst_idx] = out * 255.0f;
                }
            );
        });

        event.wait();

        // Free device memory
        sycl::free(src_device_ptr, queue);
        sycl::free(d_filter, queue);

        POP_INFO2( conf.silent(), "✓ horiz_from_input_image completed successfully" );

    } catch (sycl::exception const& e) {
        std::cerr << "SYCL exception: " << e.what() << std::endl;
        throw;
    }
}

} // namespace popsift

