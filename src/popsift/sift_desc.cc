/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/debug_macros.h"
#include "common/grid.h"
#include "s_desc_vlfeat.h"
#include "s_desc_normalize.h"
#include "s_gradiant.h"
#include "sift_config.h"
#include "sift_constants.h"
#include "sift_pyramid.h"
#include "s_desc_loop.h"

#include <cstdio>
#include <iostream>
#include <vector>

using namespace popsift;
using namespace std;

// Helper function to query kernel sub-group size
template<typename KernelName>
size_t get_kernel_subgroup_size(sycl::queue& q) {
    auto device = q.get_device();
    
    // Query the maximum sub-group size supported by the device
    auto sg_sizes = device.get_info<sycl::info::device::sub_group_sizes>();
    
    if(sg_sizes.empty()) {
        return 1; // Fallback if no sub-group support
    }
    
    // Return the maximum sub-group size
    return *std::max_element(sg_sizes.begin(), sg_sizes.end());
}

/*************************************************************
 * descriptor extraction
 * TODO: We use the level of the octave in which the keypoint
 *       was found to extract the descriptor. This is
 *       not 100% as intended by Lowe. The paper says:
 *       "magnitudes and gradient are sampled around the
 *        keypoint location, using the scale of the keypoint
 *        to select the level of Gaussian blur for the image."
 *       This implies that a keypoint that has changed octave
 *       in subpixelic refinement is going to be sampled from
 *       the wrong level of the octave.
 *       Unfortunately, we cannot implement getDataTexPoint()
 *       as a layered 2D texture to fix this issue, because that
 *       would require to store blur levels in cudaArrays, which
 *       are hard to write. Alternatively, we could keep a
 *       device-side octave structure that contains an array of
 *       levels on the device side.
 *************************************************************/
void Pyramid::descriptors(const Config& conf) {

    auto start = std::chrono::high_resolution_clock::now();

    // === Phase 1: Unified Allocation ===
    Descriptor* d_all_descs = sycl::malloc_device<Descriptor>(dct.ori_total, _shared_queue);
    Extremum* d_extrema = sycl::malloc_device<Extremum>(dbuf.extrema.size(), _shared_queue);
    int* d_feat_to_ext_map = sycl::malloc_device<int>(dbuf.feat_to_ext_map.size(), _shared_queue);
    
    auto copy_extrema = _shared_queue.memcpy(d_extrema, dbuf.extrema.data(), 
                                             dbuf.extrema.size() * sizeof(Extremum));
    auto copy_feat_map = _shared_queue.memcpy(d_feat_to_ext_map, dbuf.feat_to_ext_map.data(), 
                                              dbuf.feat_to_ext_map.size() * sizeof(int));
    
    // === Phase 2: Hardware Detection ===
    auto group_size = get_kernel_subgroup_size<sub_group_desc_loop>(_shared_queue);
    bool use_sub_group = group_size >= 32;
    
    if(!use_sub_group) {
        std::cerr << "Warning: Sub-group size " << group_size 
                  << " < 32, using local memory fallback" << std::endl;
    }
    
    // === Phase 3: Multi-threaded Extraction ===
    std::vector<sycl::event> extraction_events;
    for(int octave = _num_octaves - 1; octave >= 0; octave--) {
        if(dct.ori_ct[octave] == 0) continue;
        
        Octave& oct_obj = _octaves[octave];
        const int num_orientations = dct.ori_ct[octave];
        const int orientation_offset = dct.ori_ps[octave];
        
        sycl::range<3> global{4, 4, static_cast<size_t>(num_orientations * 32)};
        sycl::range<3> local{4, 4, 32};
        
        sycl::event e;
        if(use_sub_group) {
            e = _shared_queue.parallel_for(
                sycl::nd_range{global, local},
                {copy_extrema, copy_feat_map},
                Ext_desc_loop(octave, orientation_offset,
                             oct_obj.getWidth(), oct_obj.getHeight(),
                             d_all_descs + orientation_offset,
                             d_extrema, d_feat_to_ext_map,
                             oct_obj.getData().getDevPtr(),
                             oct_obj.getData().getCols()));
        } else {
            e = _shared_queue.submit([&](sycl::handler& cgh) {
                cgh.depends_on({copy_extrema, copy_feat_map});
                auto sum = sycl::local_accessor<float, 1>((local[2] + 7) * 16, cgh);
                cgh.parallel_for(
                    sycl::nd_range{global, local},
                    Ext_desc_loop_local_mem(sum, octave, orientation_offset,
                                           oct_obj.getWidth(), oct_obj.getHeight(),
                                           d_all_descs + orientation_offset,
                                           d_extrema, d_feat_to_ext_map,
                                           oct_obj.getData().getDevPtr(),
                                           oct_obj.getData().getCols()));
            });
        }
        extraction_events.push_back(e);
    }
    sycl::event::wait(extraction_events);
    
    // === Phase 4: Check for descriptors ===
    if(dct.ori_total == 0) {
        fprintf(stderr, "Warning: no descriptors extracted\n");
        sycl::free(d_all_descs, _shared_queue);
        sycl::free(d_extrema, _shared_queue);
        sycl::free(d_feat_to_ext_map, _shared_queue);
        return;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "descriptor extraction" 
              << " took " << duration.count() << " ms" << std::endl;

    
    auto start_norm = std::chrono::high_resolution_clock::now();

    // === Phase 5: Normalization ===
    sycl::event norm_event;
    if(conf.getUseRootSift()) {
        norm_event = normalize_histogram_sycl<NormalizeRootSift>(
            _shared_queue, 
            d_all_descs, 
            dct.ori_total,
            static_cast<float>(h_consts.norm_multi),
            use_sub_group);
    } else {
        norm_event = normalize_histogram_sycl<NormalizeL2>(
            _shared_queue, 
            d_all_descs, 
            dct.ori_total,
            static_cast<float>(h_consts.norm_multi),
            use_sub_group);
    }

    auto end_norm = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> norm_duration = end_norm - start_norm;
    std::cout << "normalization" 
              << " took " << norm_duration.count() << " ms" << std::endl;
    
    // === Phase 6: Download ===
    auto download_event = _shared_queue.memcpy(dbuf.desc, d_all_descs,
                                               dct.ori_total * sizeof(Descriptor),
                                               norm_event);
    download_event.wait();
    
    // === Cleanup ===
    sycl::free(d_all_descs, _shared_queue);
    sycl::free(d_extrema, _shared_queue);
    sycl::free(d_feat_to_ext_map, _shared_queue);
}