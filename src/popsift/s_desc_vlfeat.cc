/*
 * Copyright 2016-2017, Simula Research Laboratory
 *           2018-2020, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "popsift/sift_config.h"

#include "common/assist.h"
#include "common/debug_macros.h"
#include "common/vec_macros.h"
#include "s_desc_vlfeat.h"
#include "s_gradiant.h"
#include "sift_constants.h"
#include "sift_pyramid.h"

#include <cstdio>

using namespace popsift;

namespace popsift
{

// Define cleanup method (declared in header)
void DescriptorDeviceMemory::cleanup() {
    if (d_extrema) sycl::free(d_extrema, *queue);
    if (d_feat_to_ext_map) sycl::free(d_feat_to_ext_map, *queue);
    if (d_desc) sycl::free(d_desc, *queue);
}

// SYCL kernel for descriptor extraction
sycl::event ext_desc_vlfeat_sycl(
    sycl::queue& q,
    const int octave,
    const float* d_layer_data,
    const int layer_pitch,
    const int num_levels,
    const int width,
    const int height,
    const Extremum* d_extrema,
    const int* d_feat_to_ext_map,
    const int num_orientations,
    const int orientation_offset,
    Descriptor* d_descriptors,
    const std::vector<sycl::event>& depends_on)
{
return q.submit([&](sycl::handler& cgh) {
        // Add dependencies if provided
        if (!depends_on.empty()) {
            cgh.depends_on(depends_on);
        }
        cgh.parallel_for(sycl::range<1>(num_orientations), [=](sycl::id<1> idx) {
            const int ori_idx = idx[0];
            const int o_offset = orientation_offset + ori_idx;

            const int o_offset_global = orientation_offset + ori_idx;
            const int ext_idx = d_feat_to_ext_map[o_offset];
            const Extremum ext = d_extrema[ext_idx];
            
            const int ext_base = ext.idx_ori;
            const int ori_num = o_offset - ext_base;
            const float ang = ext.orientation[ori_num];
            
            const float x = ext.xpos;
            const float y = ext.ypos;
            const int level = ext.lpos;
            const float sig = ext.sigma;
            const float SBP = sycl::fabs(DESC_MAGNIFY * sig);
            
            if(SBP == 0.0f) return;
            
            float cos_t = sycl::cos(ang);
            float sin_t = sycl::sin(ang);
            
            const float csbp = cos_t * SBP;
            const float ssbp = sin_t * SBP;
            const float crsbp = cos_t / SBP;
            const float srsbp = sin_t / SBP;
            
            const float maxdist_x = -2.0f;
            const float maxdist_y = -2.0f;
            
            const float ptx = sycl::fabs(csbp * maxdist_x - ssbp * maxdist_y);
            const float pty = sycl::fabs(csbp * maxdist_y + ssbp * maxdist_x);
            
            const float bsz = 2.0f * (sycl::fabs(csbp) + sycl::fabs(ssbp));
            
            const int xmin = sycl::max(1, (int)sycl::floor(x - ptx - bsz));
            const int ymin = sycl::max(1, (int)sycl::floor(y - pty - bsz));
            const int xmax = sycl::min(width - 2, (int)sycl::floor(x + ptx + bsz));
            const int ymax = sycl::min(height - 2, (int)sycl::floor(y + pty + bsz));
            
            float dpt[128];
            for(int i = 0; i < 128; i++) dpt[i] = 0.0f;
            
            // Bounds check for level and ensure we can access neighbors
            if(level < 0 || level >= num_levels) return;
            
            // Sample gradients in the patch
            for(int pix_y = ymin; pix_y <= ymax; pix_y++) {
                for(int pix_x = xmin; pix_x <= xmax; pix_x++) {
                    if(pix_x <= 0 || pix_x >= width - 1 || 
                       pix_y <= 0 || pix_y >= height - 1) {
                        continue;
                    }
                    
                    const int offset = level * (height * layer_pitch) + pix_y * layer_pitch + pix_x;
                    
                    const float pix_x_pls_1 = d_layer_data[offset + 1];
                    const float pix_x_min_1 = d_layer_data[offset - 1];
                    const float pix_y_pls_1 = d_layer_data[offset + layer_pitch];
                    const float pix_y_min_1 = d_layer_data[offset - layer_pitch];
                    
                    const float dx = pix_x_pls_1 - pix_x_min_1;
                    const float dy = pix_y_pls_1 - pix_y_min_1;
                    
                    float mod = sycl::sqrt(dx * dx + dy * dy);
                    float th = sycl::atan2(dy, dx);
                    
                    mod /= 2.0f;
                    
                    th -= ang;
                    while(th > M_PI2) th -= M_PI2;
                    while(th < 0.0f) th += M_PI2;
                    
                    const float d_x = pix_x - x;
                    const float d_y = pix_y - y;
                    
                    const float n_x = crsbp * d_x + srsbp * d_y;
                    const float n_y = crsbp * d_y - srsbp * d_x;
                    
                    const float ww = sycl::exp(-(n_x * n_x + n_y * n_y) / 8.0f);
                    
                    const float nt = 8.0f * th / M_PI2;
                    
                    const int t0_x = (int)sycl::floor(n_x - 0.5f);
                    const int t0_y = (int)sycl::floor(n_y - 0.5f);
                    const int t0_z = (int)nt;
                    
                    const float wgt_x = -(n_x - (t0_x + 0.5f));
                    const float wgt_y = -(n_y - (t0_y + 0.5f));
                    const float wgt_t = -(nt - t0_z);
                    
                    for(int tx = 0; tx < 2; tx++) {
                        for(int ty = 0; ty < 2; ty++) {
                            for(int tt = 0; tt < 2; tt++) {
                                if((t0_y + ty >= -2) && (t0_y + ty < 2) &&
                                   (t0_x + tx >= -2) && (t0_x + tx < 2)) {
                                    
                                    float i_wgt_x = (tx == 0) ? 1.0f + wgt_x : wgt_x;
                                    float i_wgt_y = (ty == 0) ? 1.0f + wgt_y : wgt_y;
                                    float i_wgt_t = (tt == 0) ? 1.0f + wgt_t : wgt_t;
                                    
                                    i_wgt_x = sycl::fabs(i_wgt_x);
                                    i_wgt_y = sycl::fabs(i_wgt_y);
                                    i_wgt_t = sycl::fabs(i_wgt_t);
                                    
                                    const float val = ww * mod * i_wgt_x * i_wgt_y * i_wgt_t;
                                    
                                    const int offset_desc = 80 + (t0_y + ty) * 32 + (t0_x + tx) * 8 + (t0_z + tt) % 8;
                                    
                                    dpt[offset_desc] += val;
                                }
                            }
                        }
                    }
                }
            }
            
            // Write descriptor
            Descriptor* desc = &d_descriptors[ori_idx];
            for(int i = 0; i < 128; i++) {
                desc->features[i] = dpt[i];
            }
        });
    });
}

std::pair<sycl::event, DescriptorDeviceMemory> start_ext_desc_vlfeat_async(
    const int octave, 
    Octave& oct_obj )
{
    sycl::queue& q = oct_obj.getQueue();
    
    if( dct.ori_ct[octave] == 0 ) {
        // Return dummy event for no-op
        auto dummy_event = q.submit([&](sycl::handler& cgh) {
            cgh.single_task([=]() { /* no-op */ });
        });
        return {dummy_event, DescriptorDeviceMemory(nullptr, nullptr, nullptr, &q)};
    }

    // Allocate device memory
    const int num_orientations = dct.ori_ct[octave];
     
    Descriptor* d_desc = sycl::malloc_device<Descriptor>(num_orientations, q);
    
    Extremum* d_extrema = sycl::malloc_device<Extremum>(dbuf.extrema.size(), q);
    int* d_feat_to_ext_map = sycl::malloc_device<int>(dbuf.feat_to_ext_map.size(), q);
    
    auto copy_event1 = q.memcpy(d_extrema, dbuf.extrema.data(), 
                                 dbuf.extrema.size() * sizeof(Extremum));
    auto copy_event2 = q.memcpy(d_feat_to_ext_map, dbuf.feat_to_ext_map.data(), 
                                dbuf.feat_to_ext_map.size() * sizeof(int));

    auto kernel_event = ext_desc_vlfeat_sycl(
                            q,
                            octave,
                            oct_obj.getData().getDevPtr(),
                            oct_obj.getData().getCols(),
                            oct_obj.getLevels(),
                            oct_obj.getWidth(),
                            oct_obj.getHeight(),
                            d_extrema,
                            d_feat_to_ext_map,
                            num_orientations,
                            dct.ori_ps[octave],
                            d_desc,
                            {copy_event1, copy_event2}
                        );
        
    // Copy results back to host asynchronously
    const int host_offset = dct.ori_ps[octave];
    auto copy_back_event = q.memcpy(&dbuf.desc[host_offset], d_desc, 
                                     num_orientations * sizeof(Descriptor),
                                     kernel_event);
                                     
    // Return event and device memory (caller will clean up after event completes)
    DescriptorDeviceMemory dev_mem(d_extrema, d_feat_to_ext_map, d_desc, &q);
    return {copy_back_event, dev_mem};
}

}; // namespace popsift