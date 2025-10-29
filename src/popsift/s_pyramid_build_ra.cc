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

struct HorizFromInputKernelTag { };

namespace popsift {

// simple device-callable bilinear sampler
static inline float bilinear_sample_device(const float *src, int pitch,
                                           int w, int h, float fx, float fy) {
    fx = sycl::clamp(fx, 0.0f, float(w - 1));
    fy = sycl::clamp(fy, 0.0f, float(h - 1));
    int x0 = int(fx);
    int y0 = int(fy);
    int x1 = sycl::min(x0 + 1, w - 1);
    int y1 = sycl::min(y0 + 1, h - 1);
    float tx = fx - float(x0);
    float ty = fy - float(y0);
    float v00 = src[y0 * pitch + x0];
    float v10 = src[y0 * pitch + x1];
    float v01 = src[y1 * pitch + x0];
    float v11 = src[y1 * pitch + x1];
    float v0 = v00 * (1.0f - tx) + v10 * tx;
    float v1 = v01 * (1.0f - tx) + v11 * tx;
    return v0 * (1.0f - ty) + v1 * ty;
}


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

    // Get Gaussian filter parameter of span for input image
    const int span = h_gauss.dd.span[0];
    
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

        // POP_INFO2( conf.silent(), "  src total floats: " << src_total_floats << " (" << (src_total_floats * 4) << " bytes)" );
        // POP_INFO2( conf.silent(), "  dst total floats: " << dst_total_floats << " (" << (dst_total_floats * 4) << " bytes)" );

        // Allocate device memory for src
        // POP_INFO2( conf.silent(), "  Allocating device memory for src..." );
        float* src_device_ptr = sycl::malloc_device<float>(src_total_floats, queue);
        
        // Copy src from host to device
        //POP_INFO2( conf.silent(), "  Copying src to device..." );
        sycl::event copy_src_event = queue.memcpy(src_device_ptr, src_host_ptr, src_total_floats * sizeof(float));

        // Allocate filter on device
        float* d_filter = sycl::malloc_device<float>(span + 1, queue);
        sycl::event copy_filter_event = queue.memcpy(d_filter, h_gauss.dd.filter, (span + 1) * sizeof(float));

        //POP_INFO2( conf.silent(), "  Data uploaded, submitting kernel..." );

        // precompute scale factors for mapping dst -> src
        const float scale_x = float(src_w) / float(dst_w);
        const float scale_y = float(src_h) / float(dst_h);

        // Use a file-scope kernel name and only POD captures to avoid missing kernel images
        auto event = queue.submit([=](sycl::handler& cgh) {

            // Make kernel wait for copies to finish
            cgh.depends_on({ copy_src_event, copy_filter_event });

            cgh.parallel_for<HorizFromInputKernelTag>(sycl::range<2>(dst_h, dst_w),
                [=](sycl::id<2> idx) {
                    const int write_y = idx[0];
                    const int write_x = idx[1];

                    // compute corresponding source coords
                    const float read_x = write_x * scale_x;
                    const float read_y = write_y * scale_y;

                    float out = 0.0f;

                    // accumulate symmetric filter samples
                    for (int offset = span; offset > 0; --offset) {
                        const float w = d_filter[offset];
                        const float offrel = offset * scale_x;

                        // sample left and right using the helper
                        out += (bilinear_sample_device(src_device_ptr, src_pitch, src_w, src_h, read_x - offrel, read_y)
                              + bilinear_sample_device(src_device_ptr, src_pitch, src_w, src_h, read_x + offrel, read_y))
                              * w;
                    }

                    // center sample
                    out += bilinear_sample_device(src_device_ptr, src_pitch, src_w, src_h, read_x, read_y)
                         * d_filter[0];

                    const int dst_idx = write_y * dst_pitch + write_x;
                    dst_device_ptr[dst_idx] = out * 255.0f;
                });
        });

        // wait for kernel and check event for error (some backends report asynchronously)
        event.wait();

        // Free device memory
        sycl::free(src_device_ptr, queue);
        sycl::free(d_filter, queue);

        POP_INFO2( conf.silent(), "  horiz_from_input_image completed successfully" );

    } catch (sycl::exception const& e) {
        // Improved, actionable diagnostics for human reading
        try {
            auto dev = queue.get_device();
            auto plat = dev.get_platform();
            std::cerr << "SYCL exception while running horiz_from_input_image:\n";
            std::cerr << "  what(): " << e.what() << "\n";
            std::cerr << "  code(): " << e.code().value() << " (platform-specific)\n";
            std::cerr << "  Device name: " << dev.get_info<sycl::info::device::name>() << "\n";
            std::cerr << "  Platform: " << plat.get_info<sycl::info::platform::name>() << "\n";
        } catch(...) {
            std::cerr << "SYCL exception: " << e.what() << "\n";
        }

        std::cerr << "\nPossible causes and quick fixes:\n";
        std::cerr << " - Kernel not found on device: ensure device image was emitted at link time\n";
        std::cerr << "   * For CUDA: build with your SYCL compiler flags enabling NVPTX device image\n";
        std::cerr << "     e.g. -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda\n";
        std::cerr << " - Avoid nested lambdas or capturing non-POD objects in the kernel lambda.\n";
        std::cerr << " - To diagnose loader/adapter/device selection, run with:\n";
        std::cerr << "     export SYCL_UR_TRACE=1\n";
        std::cerr << " - If you used a custom libsycl on LD_LIBRARY_PATH, ensure you run the matching binary\n";
        std::cerr << " - If problem persists, run: strings <binary> | egrep -i 'nvptx|cuda|<HorizFromInputKernelTag>'\n";
        std::cerr << std::endl;

        throw; // rethrow so caller can decide to exit/cleanup
    } catch (std::exception const& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        throw;
    }
}

} // namespace popsift

