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
#include "gauss_filter.h"
#include "sift_pyramid.h"
#include "sift_constants.h"

namespace popsift {

void Pyramid::horiz_from_prev_level( int octave, int level )
{
    Octave& oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();
    
    PlaneD<float>& data = oct_obj.getData();  
    PlaneD<float>& intm = oct_obj.getIntm();  
    
    float* src_ptr = data.getDevicePtr();
    float* dst_ptr = intm.getDevicePtr();
    
    const int src_pitch = data.getPitch();
    const int dst_pitch = intm.getPitch();

    const int src_level = level - 1;
    const int src_level_offset = src_level * height * src_pitch;
    const int span = h_gauss.inc.span[level];
    
    // Copy filter to device
    std::vector<float> filter_host(span + 1);
    for(int i = 0; i <= span; i++) {
        filter_host[i] = h_gauss.inc.filter[level * GAUSS_ALIGN + i];
    }
    
    sycl::queue& queue = oct_obj.getQueue();
    float* d_filter = sycl::malloc_device<float>(span + 1, queue);
    queue.memcpy(d_filter, filter_host.data(), (span + 1) * sizeof(float)).wait();
    
    // Launch SYCL kernel for horizontal filtering
    auto event = queue.submit([&](sycl::handler& cgh) {        
        cgh.parallel_for(
            sycl::range<2>(height, width),
            [=](sycl::id<2> idx) {
                const int x = idx[1];
                const int y = idx[0];
                
                if (x >= width || y >= height) return;
                
                float out = 0.0f;
                
                #pragma unroll
                for(int offset = span; offset > 0; offset--) {
                    const float weight = d_filter[offset];
                    
                    // Clamp x coordinates
                    int x_neg = sycl::max(0, x - offset);
                    int x_pos = sycl::min(width - 1, x + offset);
                    
                    // Read from src_level, write to dst_level
                    const int idx_neg = src_level_offset + y * src_pitch + x_neg;
                    const int idx_pos = src_level_offset + y * src_pitch + x_pos;
                    
                    out += src_ptr[idx_neg] * weight;
                    out += src_ptr[idx_pos] * weight;
                }
                
                const float weight0 = d_filter[0];
                const int idx_center = src_level_offset + y * src_pitch + x;
                out += src_ptr[idx_center] * weight0;
                
                // Write to dst_level of intermediate plane
                const int dst_idx = level * height * dst_pitch + y * dst_pitch + x;
                dst_ptr[dst_idx] = out;
            }
        );
    });
    
    event.wait();
    sycl::free(d_filter, queue);
}

void Pyramid::vert_from_interm( int octave, int level )
{
    Octave& oct_obj = _octaves[octave];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();
    
    PlaneD<float>& intm = oct_obj.getIntm();  
    PlaneD<float>& data = oct_obj.getData(); 
    
    float* src_ptr = intm.getDevicePtr();
    float* dst_ptr = data.getDevicePtr();
    
    const int src_pitch = intm.getPitch();
    const int dst_pitch = data.getPitch();

    const int src_level_offset = level * height * src_pitch;
    
    const int span = h_gauss.inc.span[level];
    float* h_filter = h_gauss.inc.filter + level * GAUSS_ALIGN;
    
    sycl::queue& queue = oct_obj.getQueue();
    float* d_filter = sycl::malloc_device<float>(span + 1, queue);
    queue.memcpy(d_filter, h_filter, (span + 1) * sizeof(float)).wait();
    
    // Launch SYCL kernel
    auto event = queue.submit([&](sycl::handler& cgh) {
        
        cgh.parallel_for(
            sycl::range<2>(height, width),
            [=](sycl::id<2> idx) {
                const int x = idx[1];
                const int y = idx[0];
                
                if (x >= width || y >= height) return;
                
                float out = 0.0f;
                
                #pragma unroll
                for(int offset = span; offset > 0; offset--) {
                    const float weight = d_filter[offset];
                    
                    // Clamp coordinates
                    int y_neg = sycl::max(0, y - offset);
                    int y_pos = sycl::min(height - 1, y + offset);
                    
                    const int idx_neg = src_level_offset + y_neg * src_pitch + x;
                    const int idx_pos = src_level_offset + y_pos * src_pitch + x;
                    
                    out += src_ptr[idx_neg] * weight;
                    out += src_ptr[idx_pos] * weight;
                }
                
                const float weight0 = d_filter[0];
                const int idx_center = level * height * src_pitch + y * src_pitch + x;
                out += src_ptr[idx_center] * weight0;
                
                const int dst_idx = level * height * dst_pitch + y * dst_pitch + x;
                dst_ptr[dst_idx] = out;
            }
        );
    });
    
    event.wait();
    sycl::free(d_filter, queue);
}

} // namespace popsift

