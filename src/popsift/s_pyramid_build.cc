/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/clamp.h"
#include "common/grid.h"
#include "common/debug_macros.h"
#include "gauss_filter.h"
#include "sift_constants.h"
#include "sift_pyramid.h"

#include <cstdio>
#include <iostream>

#include <sycl/sycl.hpp>


/* It makes no sense whatsoever to change this value */
#define PREV_LEVEL 3

using std::cout;
using std::cerr;
using std::endl;

namespace popsift {

namespace gauss {

static inline
void get_by_2_pick_every_second( sycl::queue&   queue,
                                 const int      src_level,
                                 PlaneD<float>& src,
                                 PlaneD<float>& dst )
{
    const int dst_level = 0; // always writing to the first plane in the destination octave

    const int src_w = src.getDimX();
    const int src_h = src.getDimY();
    const int dst_w = dst.getDimX();
    const int dst_h = dst.getDimY();
    
    float* src_ptr = src.getDevicePtr();
    float* dst_ptr = dst.getDevicePtr();
    
    const int src_pitch = src.getPitch();
    const int dst_pitch = dst.getPitch();

    // Launch SYCL kernel for downsampling
    auto event = queue.submit([&](sycl::handler& cgh) {
        const int c_src_w = src_w;
        const int c_src_h = src_h;
        const int c_dst_w = dst_w;
        const int c_dst_h = dst_h;
        const int c_src_pitch = src_pitch;
        const int c_dst_pitch = dst_pitch;
        const int c_src_level = src_level;
        const int c_dst_level = dst_level;
        
        cgh.parallel_for(
            sycl::range<2>(dst_h, dst_w),
            [=](sycl::id<2> idx) {
                const int write_x = idx[1];
                const int write_y = idx[0];
                
                if (write_x >= c_dst_w || write_y >= c_dst_h) return;
                
                // Read every second pixel from source (2x downsampling)
                const int read_x = sycl::clamp(write_x << 1, 0, c_src_w - 1);
                const int read_y = sycl::clamp(write_y << 1, 0, c_src_h - 1);
                
                // Read from src_level
                const int src_idx = c_src_level * c_src_h * c_src_pitch + read_y * c_src_pitch + read_x;
                const float val = src_ptr[src_idx];
                
                // Write to dst_level (layer 0)
                const int dst_idx = c_dst_level * c_dst_h * c_dst_pitch + write_y * c_dst_pitch + write_x;
                dst_ptr[dst_idx] = val;
            }
        );
    });
    
    event.wait();
}

}; // namespace gauss

void Pyramid::downscale_from_prev_octave( int octave )
{
    Octave&      oct_obj = _octaves[octave];
    Octave& prev_oct_obj = _octaves[octave-1];

    const int width  = oct_obj.getWidth();
    const int height = oct_obj.getHeight();
    
    // Use the current octave's queue
    sycl::queue& queue = oct_obj.getQueue();

    gauss::get_by_2_pick_every_second( queue,
                                       _levels-PREV_LEVEL,
                                       prev_oct_obj.getData( ),
                                       oct_obj.getData( ) );
}

namespace gauss {


static
sycl::event make_dog( sycl::queue&   q,
                      PlaneD<float>& src,
                      PlaneD<float>& dog,
                      const int      w,
                      const int      h,
                      const int      max_level )
{
    // Get device pointers 
    float* d_src = src.getDevicePtr();
    float* d_dog = dog.getDevicePtr();
    
    const int src_pitch = src.getPitch();
    const int dog_pitch = dog.getPitch();
    
    // Calculate grid dimensions (matching CUDA: 1024x1 blocks)
    const int block_x = 1024;
    const int block_y = 1;
    const int grid_x = (w + block_x - 1) / block_x;
    const int grid_y = (h + block_y - 1) / block_y;
    
    sycl::range<2> local_range(block_y, block_x);
    sycl::range<2> global_range(grid_y * block_y, grid_x * block_x);
    
    // Submit kernel and return event (non-blocking)
    auto event = q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<2>(global_range, local_range),
            [=](sycl::nd_item<2> item) {
                const int idx = item.get_global_id(1);
                const int idy = item.get_global_id(0);
                
                if (idx < w && idy < h) {
                    const int pixel_offset = idy * src_pitch + idx;
                    float a = d_src[pixel_offset];
                    
                    for (int level = 0; level < max_level - 1; level++) {
                        const int next_level_offset = (level + 1) * h * src_pitch + pixel_offset;
                        const int dog_level_offset = level * h * dog_pitch + idy * dog_pitch + idx;
                        
                        const float b = d_src[next_level_offset];
                        d_dog[dog_level_offset] = b - a;
                        a = b;
                    }
                }
            });
    });
    
    // Return event without waiting - allows parallel execution
    return event;
}


} // namespace gauss


// Async version - returns event
sycl::event Pyramid::dogs_from_blurred( int octave, int max_level )
{
    Octave& oct_obj = _octaves[octave];
    
    // Use the octave's own queue for parallel execution
    sycl::queue& q = oct_obj.getQueue();
    
    return gauss::make_dog( q,
                            oct_obj.getData(),
                            oct_obj.getDog(),
                            oct_obj.getWidth(),
                            oct_obj.getHeight(),
                            max_level);
}

/*************************************************************
 * V11: host side
 *************************************************************/
void Pyramid::build_pyramid( const Config& conf, std::shared_ptr<ImageBase> base )
{
    POP_INFO2( conf.silent(), "enter " << __PRETTY_FUNCTION__ );
    POP_INFO2( conf.silent(), "is image NULL? " << ( base->isNull() ? "yes" : "no") );

    for( uint32_t octave=0; octave<_num_octaves; octave++ )
    {
        Octave&      oct_obj = _octaves[octave];

        for( int level=0; level<_levels; level++ )
        {
            if( level == 0 )
            {
                if( octave == 0 )
                {
                    POP_INFO2( conf.silent(), "call horiz_from_input_image" );
                    horiz_from_input_image( conf, base );

                    POP_INFO2( conf.silent(), "call vert_from_interm" );
                    vert_from_interm( octave, 0 );
                }
                else
                {
                    Octave& prev_oct_obj = _octaves[octave-1];
                    POP_INFO2( conf.silent(), "call downscale_from_prev_octave" );
                    downscale_from_prev_octave( octave );
                }
            }
            else
            {
                POP_INFO2( conf.silent(), "call horiz_from_prev_level" );
                horiz_from_prev_level( octave, level );

                POP_INFO2( conf.silent(), "call vert_from_interm" );
                vert_from_interm( octave, level );
            }
        }
    }

    // Launch DoG kernels asynchronously on all octaves
    // Each octave uses its own queue, so they execute in parallel
    std::vector<sycl::event> events;
    for( int octave=0; octave<_num_octaves; octave++ )
    {
        POP_INFO2( conf.silent(), "call dogs_from_blurred (async)" );
        
        // Submit kernel on octave's queue (non-blocking)
        sycl::event event = dogs_from_blurred( octave, _levels );
        events.push_back(event);
    }

    // Wait for all DoG computations to complete
    POP_INFO2( conf.silent(), "waiting for all DoG kernels to complete" );
    for (auto& e : events) {
        e.wait();
    }
    
    POP_INFO2( conf.silent(), "DoG computation complete" );
}

    
}// namespace popsift

