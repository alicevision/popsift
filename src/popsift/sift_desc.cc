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
    if( dct.ori_total == 0 )
    {
        cerr << "Warning: no descriptors to extract" << endl;
        return;
    }

    // Store events and device memory for cleanup
    std::vector<sycl::event> descriptor_events;
    std::vector<DescriptorDeviceMemory> device_memory;
    
    descriptor_events.reserve(_num_octaves);
    device_memory.reserve(_num_octaves);
    
    // Launch all octave descriptor extractions asynchronously
    for( int octave=_num_octaves-1; octave>=0; octave-- )
    {
        if( dct.ori_ct[octave] != 0 ) {
            Octave& oct_obj = _octaves[octave];

            // Launch async and store event + device memory
            auto [event, dev_mem] = start_ext_desc_vlfeat_async( octave, oct_obj );
            descriptor_events.push_back(event);
            device_memory.push_back(dev_mem);
        }
    }

    // Wait for ALL descriptor extractions to complete
    for(auto& evt : descriptor_events) {
        evt.wait();
    }
    
   // Allocate device memory for all descriptors
   sycl::queue& q = _octaves[0].getQueue();  // Use shared queue

   Descriptor* d_all_descs = sycl::malloc_device<Descriptor>(dct.ori_total, q);
 
    if (!d_all_descs) {
        cerr << "Error: Failed to allocate device memory for descriptors" << endl;
        // Clean up extraction memory and return
        for(auto& dev_mem : device_memory) {
            dev_mem.cleanup();
        }
        return;
    }   


    // Copy all descriptors to device AND WAIT
    auto copy_to_device = q.memcpy(d_all_descs, dbuf.desc, 
                                    dct.ori_total * sizeof(Descriptor));
    copy_to_device.wait();  // CRITICAL: Wait for copy to complete!

   // Launch normalization kernel
   sycl::event norm_event;
    if( conf.getUseRootSift() ) {
        norm_event = normalize_histogram_sycl<NormalizeRootSift>(q, d_all_descs, dct.ori_total);
    } else {
        norm_event = normalize_histogram_sycl<NormalizeL2>(q, d_all_descs, dct.ori_total);
    }
    
    norm_event.wait();

   // Copy normalized descriptors back to host
    auto copy_to_host = q.memcpy(dbuf.desc, d_all_descs, 
                                  dct.ori_total * sizeof(Descriptor));   
   // Wait for normalization to complete
   copy_to_host.wait();

   // Clean up
   sycl::free(d_all_descs, q);
   
   // Clean up descriptor extraction device memory
   for(auto& dev_mem : device_memory) {
       dev_mem.cleanup();
   }

}

