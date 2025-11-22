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

#include <cstdio>
#include <iostream>
#include <vector>

using namespace popsift;
using namespace std;

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
void Pyramid::descriptors( const Config& conf )
{
    auto start = std::chrono::high_resolution_clock::now();

    if( dct.ori_total == 0 )
    {
        cerr << "Warning: no descriptors to extract" << endl;
        return;
    }

    sycl::queue& q = _shared_queue;
    
    // Allocate ALL device memory at once (instead of per-octave)
    Descriptor* d_all_descs = sycl::malloc_device<Descriptor>(dct.ori_total, q);
    Extremum* d_extrema = sycl::malloc_device<Extremum>(dbuf.extrema.size(), q);
    int* d_feat_to_ext_map = sycl::malloc_device<int>(dbuf.feat_to_ext_map.size(), q);
    
    if (!d_all_descs || !d_extrema || !d_feat_to_ext_map) {
        cerr << "Error: Failed to allocate device memory for descriptors" << endl;
        if (d_all_descs) sycl::free(d_all_descs, q);
        if (d_extrema) sycl::free(d_extrema, q);
        if (d_feat_to_ext_map) sycl::free(d_feat_to_ext_map, q);
        return;
    }

    // Copy shared data ONCE (instead of per-octave)
    auto copy_extrema = q.memcpy(d_extrema, dbuf.extrema.data(), 
                                  dbuf.extrema.size() * sizeof(Extremum));
    auto copy_feat_map = q.memcpy(d_feat_to_ext_map, dbuf.feat_to_ext_map.data(), 
                                   dbuf.feat_to_ext_map.size() * sizeof(int));


    std::vector<sycl::event> kernel_events;
    kernel_events.reserve(_num_octaves);
    
    // Launch all octave kernels asynchronously with dependencies
    for( int octave = _num_octaves - 1; octave >= 0; octave-- )
    {
        if( dct.ori_ct[octave] == 0 ) continue;
        
        Octave& oct_obj = _octaves[octave];
        const int num_orientations = dct.ori_ct[octave];
        const int orientation_offset = dct.ori_ps[octave];
        
        // Launch kernel with pointer offsets (no extra allocations!)
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
            orientation_offset,
            d_all_descs + orientation_offset,  // Direct write to final buffer
            {copy_extrema, copy_feat_map}      // Dependencies
        );
        
        kernel_events.push_back(kernel_event);
    }

    for(sycl::event e : kernel_events){
        e.wait();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "descriptor extraction" 
              << " took " << duration.count() << " ms" << std::endl;

    
    auto start_norm = std::chrono::high_resolution_clock::now();

    // Normalize immediately after extraction (don't wait individually)
    sycl::event norm_event;
    if( conf.getUseRootSift() ) {
        norm_event = normalize_histogram_sycl<NormalizeRootSift>(
            q, d_all_descs, dct.ori_total);
    } else {
        norm_event = normalize_histogram_sycl<NormalizeL2>(
            q, d_all_descs, dct.ori_total);
    }

    auto end_norm = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> norm_duration = end_norm - start_norm;
    std::cout << "normalization" 
              << " took " << norm_duration.count() << " ms" << std::endl;

    auto start_copy = std::chrono::high_resolution_clock::now();
    
    // Copy all results back in one shot
    auto copy_to_host = q.memcpy(dbuf.desc, d_all_descs, 
                                  dct.ori_total * sizeof(Descriptor),
                                  norm_event);
    copy_to_host.wait();

    auto end_copy = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> copy_duration = end_copy - start_copy;
    std::cout << "descriptor download" << " took " << copy_duration.count() << " ms" << std::endl;

    // ============================================
    // Cleanup
    // ============================================
    
    sycl::free(d_all_descs, q);
    sycl::free(d_extrema, q);
    sycl::free(d_feat_to_ext_map, q);
}
