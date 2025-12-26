/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/clamp.h"
#include "common/debug_macros.h"
#include "common/grid.h"
#include "s_solve.h"
#include "sift_constants.h"
#include "sift_pyramid.h"

#include <sys/stat.h>

#include <cstdio>
#include <cmath>
#include <numeric>
#include <sstream>
#include <iterator>
#include <map> 

namespace popsift{


// Host-side storage for extrema counts (one per octave)
static std::vector<int> extrema_count_host;


// Break the large kernel into smaller functions
__attribute__((noinline))
static bool check_extremum_26_neighbors(
    const float val,
    const float* dog_ptr_kernel,
    int level, int y, int x,
    int c_width, int c_height, int c_dog_pitch, int c_maxlevel,
    size_t c_dog_total_elems)
{
    // Inline the clamp and dog_get logic here
    auto clamp = [](int v, int lo, int hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    };
    
    auto dog_get = [=](int z, int y, int x) -> float {
        int zz = clamp(z, 0, c_maxlevel + 1);
        int yy = clamp(y, 0, c_height - 1);
        int xx = clamp(x, 0, c_width - 1);
        const std::size_t dog_idx = (std::size_t)zz * (std::size_t)c_height * (std::size_t)c_dog_pitch
                                  + (std::size_t)yy * (std::size_t)c_dog_pitch
                                  + (std::size_t)xx;
        
        if (dog_idx >= c_dog_total_elems) return 0.0f;
        return dog_ptr_kernel[dog_idx];
    };
    
    uint32_t gt = 0, lt = 0;
    
    auto extremum_cmp = [&](float f, uint32_t mask) {
        gt |= ((val > f) ? mask : 0);
        lt |= ((val < f) ? mask : 0);
    };

    // 1st group: TX(0,1,1) and TX(2,1,1)
    extremum_cmp(dog_get(level, y, x - 1), 0x00400000);
    extremum_cmp(dog_get(level, y, x + 1), 0x00040000);
    
    if((gt != 0x00440000) && (lt != 0x00440000)) return false;

    // 2nd group
    extremum_cmp(dog_get(level, y - 1, x), 0x00800000);
    extremum_cmp(dog_get(level, y + 1, x), 0x00200000);
    extremum_cmp(dog_get(level - 1, y - 1, x), 0x80000000);
    extremum_cmp(dog_get(level - 1, y + 1, x), 0x40000000);
    extremum_cmp(dog_get(level - 1, y, x), 0x20000000);
    extremum_cmp(dog_get(level + 1, y - 1, x), 0x00008000);
    extremum_cmp(dog_get(level + 1, y, x), 0x00004000);
    extremum_cmp(dog_get(level + 1, y + 1, x), 0x00002000);

    if((gt != 0xe0e4e000) && (lt != 0xe0e4e000)) return false;

    // 3rd group
    extremum_cmp(dog_get(level, y - 1, x - 1), 0x00010000);
    extremum_cmp(dog_get(level, y - 1, x + 1), 0x00020000);
    extremum_cmp(dog_get(level, y + 1, x - 1), 0x00100000);
    extremum_cmp(dog_get(level, y + 1, x + 1), 0x00080000);

    if((gt != 0xe0ffe000) && (lt != 0xe0ffe000)) return false;

    // 4th group
    extremum_cmp(dog_get(level - 1, y - 1, x - 1), 0x01000000);
    extremum_cmp(dog_get(level - 1, y - 1, x + 1), 0x02000000);
    extremum_cmp(dog_get(level - 1, y, x - 1), 0x00000004);
    extremum_cmp(dog_get(level - 1, y, x + 1), 0x04000000);
    extremum_cmp(dog_get(level - 1, y + 1, x - 1), 0x10000000);
    extremum_cmp(dog_get(level - 1, y + 1, x + 1), 0x08000000);

    if((gt != 0xffffe004) && (lt != 0xffffe004)) return false;

    // 5th group
    extremum_cmp(dog_get(level + 1, y - 1, x - 1), 0x00000100);
    extremum_cmp(dog_get(level + 1, y - 1, x + 1), 0x00000200);
    extremum_cmp(dog_get(level + 1, y, x - 1), 0x00000001);
    extremum_cmp(dog_get(level + 1, y, x + 1), 0x00000400);
    extremum_cmp(dog_get(level + 1, y + 1, x - 1), 0x00001000);
    extremum_cmp(dog_get(level + 1, y + 1, x + 1), 0x00000800);

    return (gt == 0xffffff05) || (lt == 0xffffff05);
}

__attribute__((noinline))
static bool refine_extremum(
    int3& n, float3& d,
    const float* dog_ptr_kernel,
    int c_width, int c_height, int c_maxlevel,
    int c_dog_pitch, float c_threshold,
    float c_edge_limit,
    size_t c_dog_total_elems,
    int initial_x, int initial_y, int initial_level,
    int max_iterations)
{
    // Define dog_get locally - compiler can optimize this
    auto clamp = [](int v, int lo, int hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    };
    
    auto dog_get = [=](int z, int y, int x) -> float {
        int zz = clamp(z, 0, c_maxlevel + 1);
        int yy = clamp(y, 0, c_height - 1);
        int xx = clamp(x, 0, c_width - 1);
        const size_t dog_idx = (size_t)zz * (size_t)c_height * (size_t)c_dog_pitch
                              + (size_t)yy * (size_t)c_dog_pitch
                              + (size_t)xx;
        
        if (dog_idx >= c_dog_total_elems) return 0.0f;
        return dog_ptr_kernel[dog_idx];
    };
    
    // Recompute gradients on demand instead of storing
    auto compute_gradient = [&](const int3& pos) -> float3 {
        float3 grad;
        grad.x = 0.5f * (dog_get(pos.z, pos.y, pos.x + 1) - dog_get(pos.z, pos.y, pos.x - 1));
        grad.y = 0.5f * (dog_get(pos.z, pos.y + 1, pos.x) - dog_get(pos.z, pos.y - 1, pos.x));
        grad.z = 0.5f * (dog_get(pos.z + 1, pos.y, pos.x) - dog_get(pos.z - 1, pos.y, pos.x));
        return grad;
    };
    
    auto compute_hessian_diag = [&](const int3& pos) -> float3 {
        float center = dog_get(pos.z, pos.y, pos.x);
        float3 hess;
        hess.x = dog_get(pos.z, pos.y, pos.x + 1) + dog_get(pos.z, pos.y, pos.x - 1) - 2.0f * center;
        hess.y = dog_get(pos.z, pos.y + 1, pos.x) + dog_get(pos.z, pos.y - 1, pos.x) - 2.0f * center;
        hess.z = dog_get(pos.z + 1, pos.y, pos.x) + dog_get(pos.z - 1, pos.y, pos.x) - 2.0f * center;
        return hess;
    };
    
    auto compute_hessian_offdiag = [&](const int3& pos) -> float3 {
        float3 hess_off;
        // Dxy
        hess_off.x = 0.25f * (dog_get(pos.z, pos.y + 1, pos.x + 1) + 
                              dog_get(pos.z, pos.y - 1, pos.x - 1) - 
                              dog_get(pos.z, pos.y + 1, pos.x - 1) - 
                              dog_get(pos.z, pos.y - 1, pos.x + 1));
        // Dxs
        hess_off.y = 0.25f * (dog_get(pos.z + 1, pos.y, pos.x + 1) + 
                              dog_get(pos.z - 1, pos.y, pos.x - 1) - 
                              dog_get(pos.z + 1, pos.y, pos.x - 1) - 
                              dog_get(pos.z - 1, pos.y, pos.x + 1));
        // Dys
        hess_off.z = 0.25f * (dog_get(pos.z + 1, pos.y + 1, pos.x) + 
                              dog_get(pos.z - 1, pos.y - 1, pos.x) - 
                              dog_get(pos.z + 1, pos.y - 1, pos.x) - 
                              dog_get(pos.z - 1, pos.y + 1, pos.x));
        return hess_off;
    };
    
    // Refinement loop
    n = {initial_x, initial_y, initial_level};
    int iter = 0;
    
    while(iter < max_iterations) {
        iter++;
        
        // Recompute at current position
        float3 D = compute_gradient(n);
        float3 DD = compute_hessian_diag(n);
        float3 DX = compute_hessian_offdiag(n);
        
        // Build matrix A
        float A[3][3];
        A[0][0] = DD.x; A[0][1] = DX.x; A[0][2] = DX.y;
        A[1][0] = DX.x; A[1][1] = DD.y; A[1][2] = DX.z;
        A[2][0] = DX.y; A[2][1] = DX.z; A[2][2] = DD.z;
        
        float3 b = {-D.x, -D.y, -D.z};
        
        // Compute determinant using cofactor expansion
        float det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[1][2])
                  - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                  + A[0][2] * (A[1][0] * A[1][2] - A[1][1] * A[2][0]);
        
        if(sycl::fabs(det) < 1e-10f) {
            d = {0.0f, 0.0f, 0.0f};
            return false;
        }
        
        float rsd = 1.0f / det;
        
        // Compute inverse (only what we need for solution)
        float inv00 = (A[1][1] * A[2][2] - A[1][2] * A[1][2]) * rsd;
        float inv01 = (A[0][2] * A[1][2] - A[0][1] * A[2][2]) * rsd;
        float inv02 = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * rsd;
        float inv11 = (A[0][0] * A[2][2] - A[0][2] * A[0][2]) * rsd;
        float inv12 = (A[0][1] * A[0][2] - A[0][0] * A[1][2]) * rsd;
        float inv22 = (A[0][0] * A[1][1] - A[0][1] * A[0][1]) * rsd;
        
        // Solve: d = inv * b
        d.x = inv00 * b.x + inv01 * b.y + inv02 * b.z;
        d.y = inv01 * b.x + inv11 * b.y + inv12 * b.z;
        d.z = inv02 * b.x + inv12 * b.y + inv22 * b.z;
        
        // Check convergence
        if(iter == max_iterations) break;
        
        // Determine movement
        int3 t = {0, 0, 0};
        t.x = ((d.x >= 0.6f && n.x < c_width - 2) ? 1 : 0) +
              ((d.x <= -0.6f && n.x > 1) ? -1 : 0);
        t.y = ((d.y >= 0.6f && n.y < c_height - 2) ? 1 : 0) +
              ((d.y <= -0.6f && n.y > 1) ? -1 : 0);
        t.z = ((d.z >= 0.6f && n.z < c_maxlevel - 1) ? 1 : 0) +
              ((d.z <= -0.6f && n.z > 1) ? -1 : 0);
        
        if(t.x == 0 && t.y == 0 && t.z == 0) break;
        
        n.x += t.x;
        n.y += t.y;
        n.z += t.z;
    }
    
    // Final validation
    if(sycl::fabs(d.x) >= 1.5f || sycl::fabs(d.y) >= 1.5f || sycl::fabs(d.z) >= 1.5f) {
        return false;
    }
    
    float xn = n.x + d.x;
    float yn = n.y + d.y;
    float sn = n.z + d.z;
    
    if(xn < 0.0f || xn > c_width - 1.0f ||
       yn < 0.0f || yn > c_height - 1.0f || 
       sn < 0.0f || sn > c_maxlevel) {
        return false;
    }
    
    // Final contrast and edge checks - recompute only what we need
    float3 D_final = compute_gradient(n);
    float3 DD_final = compute_hessian_diag(n);
    float center = dog_get(n.z, n.y, n.x);
    
    float contr = center + 0.5f * (D_final.x * d.x + D_final.y * d.y + D_final.z * d.z);
    
    if(sycl::fabs(contr) < 2.0f * c_threshold) return false;
    
    // Edge check - only compute Dxy for this
    float Dxy = 0.25f * (dog_get(n.z, n.y + 1, n.x + 1) + 
                         dog_get(n.z, n.y - 1, n.x - 1) - 
                         dog_get(n.z, n.y + 1, n.x - 1) - 
                         dog_get(n.z, n.y - 1, n.x + 1));
    
    float tr = DD_final.x + DD_final.y;
    float det = DD_final.x * DD_final.y - Dxy * Dxy;
    
    if(det <= 0.0f) return false;
    
    float edgeval = tr * tr / det;
    if(edgeval >= (c_edge_limit + 1.0f) * (c_edge_limit + 1.0f) / c_edge_limit) {
        return false;
    }
    
    return true;
}

template<int sift_mode>
static
sycl::event find_extrema_in_dog( const int3&    g,
                          Plane2D_float& dog,
                          int            octave,
                          int            width,
                          int            height,
                          const uint32_t maxlevel,
                          const float    w_grid_divider,
                          const float    h_grid_divider,
                          const int      grid_width,
                          Pyramid*       pyramid )
{
    const bool no_extrema_reporting = false;

//    POP_INFO2( no_extrema_reporting, "Converting find_extrema to SYCL kernel for octave " << octave );

    Octave& oct_obj = pyramid->getOctave(octave);
    sycl::queue& queue = oct_obj.getQueue();

    float* dog_ptr = (float*)dog.getDevicePtr();
    const int dog_pitch = dog.getPitchElements();

    const int max_extrema = g.x * g.y * g.z;

    InitialExtremum* d_extrema = sycl::malloc_device<InitialExtremum>(max_extrema, queue);
    int* d_count = sycl::malloc_device<int>(1, queue);
    
    queue.memset(d_count, 0, sizeof(int)).wait();

    // Launch SYCL kernel
    auto event = queue.submit([&](sycl::handler& cgh) {
        float* dog_ptr_kernel = dog_ptr;
        const int c_width = width;
        const int c_height = height;
        const int c_maxlevel = maxlevel;
        const int c_dog_pitch = dog_pitch;
        const int c_grid_width = grid_width;
        const float c_w_grid_div = w_grid_divider;
        const float c_h_grid_div = h_grid_divider;
        const float c_threshold = h_consts.threshold;
        const float c_edge_limit = h_consts.edge_limit;
        const float c_sigma0 = h_consts.sigma0;
        const float c_sigma_k = h_consts.sigma_k;
        const std::size_t c_dog_total_elems = (std::size_t)(c_maxlevel + 2) * (std::size_t)c_height * (std::size_t)c_dog_pitch;
        const int c_max_extrema = max_extrema;

    cgh.parallel_for(
        sycl::range<3>(g.z, g.y, g.x),
        [=](sycl::id<3> idx) {
            // Grid coordinates (0-based)
            const int gx = idx[2];
            const int gy = idx[1];
            const int gz = idx[0];
            
            // Actual coordinates for refinement (1-based, with border)
            const int x = gx + 1;
            const int y = gy + 1;
            const int level = gz + 1;

            auto clamp = [](int v, int lo, int hi) {
                return v < lo ? lo : (v > hi ? hi : v);
            };

            auto dog_get = [=](int z, int y, int x) -> float {
                // Clamp to valid range, matching PlaneT<T>::deref() logic
                int zz = clamp(z, 0, c_maxlevel + 1);
                int yy = clamp(y, 0, c_height - 1);
                int xx = clamp(x, 0, c_width - 1);
                const std::size_t dog_idx = (std::size_t)zz * (std::size_t)c_height * (std::size_t)c_dog_pitch
                                          + (std::size_t)yy * (std::size_t)c_dog_pitch
                                          + (std::size_t)xx;
                

               if (dog_idx >= c_dog_total_elems) return 0.0f;

               return dog_ptr_kernel[dog_idx];
            };

            // The value to check is at the actual grid position
            // In CPU: is_extremum(dog, x-1, y-1, level-1) with TX(1,1,1)
            // gives dog.get((level-1)+1, (y-1)+1, (x-1)+1) = dog.get(level, y, x)
            const float val = dog_get(level, y, x);

            // First contrast check
            if (sycl::fabs(val) < 1.6f * c_threshold) return;

            // Now check 26 neighbors around (level, y, x)
            // The CPU TX macro adds to (x-1, y-1, level-1), so:
            // TX(0,1,1) = dog.get((level-1)+1, (y-1)+1, (x-1)+0) = dog.get(level, y, x-1)
            // TX(2,1,1) = dog.get((level-1)+1, (y-1)+1, (x-1)+2) = dog.get(level, y, x+1)
            
            if (!check_extremum_26_neighbors(val, dog_ptr_kernel, 
                                            level, y, x,
                                            c_width, c_height, c_dog_pitch, c_maxlevel,
                                            c_dog_total_elems))
            {
                return;
            }
            
            // Refine extremum
            int3 n;
            float3 d;
            const int MAX_ITER = 5;

            bool refinement_success = refine_extremum(
                n, d,
                dog_ptr_kernel,
                c_width, c_height, c_maxlevel,
                c_dog_pitch, c_threshold,
                c_edge_limit,
                c_dog_total_elems,
                x, y, level,
                MAX_ITER
            );

            if(!refinement_success) return;

            // Compute final position
            float xn = n.x + d.x;
            float yn = n.y + d.y;
            float sn = n.z + d.z;

            // Atomically add extremum
            int write_idx = sycl::atomic_ref<int, sycl::memory_order_acq_rel, 
                                            sycl::memory_scope::device>(d_count[0]).fetch_add(1);
                                            
            if(write_idx < c_max_extrema) {
                InitialExtremum& ec = d_extrema[write_idx];
                ec.xpos = xn;
                ec.ypos = yn;
                ec.lpos = (int)sycl::round(sn);
                ec.sigma = c_sigma0 * sycl::pow(c_sigma_k, sn);
                ec.cell = sycl::floor(yn / c_h_grid_div) * c_grid_width + sycl::floor(xn / c_w_grid_div);
                ec.ignore = false;
                ec.write_index = write_idx;
            }

        });
    });


    // Copy results back to host
    auto copy_event = queue.memcpy(&extrema_count_host[octave], d_count, sizeof(int), event);

    // Store device pointers in pyramid for later cleanup
    pyramid->storeDevicePointers(octave, d_extrema, d_count);

    return copy_event;
}

void Pyramid::find_extrema( const Config& conf )
{
//    POP_INFO2( false, "Enter " << __FUNCTION__ );

    dct.extrema_count_per_octave.resize( MAX_OCTAVES );
    extrema_count_host.resize( MAX_OCTAVES );

    // Store events for async execution
    std::vector<sycl::event> octave_events;
    octave_events.reserve(_num_octaves);
      
    for( int octave=0; octave<_num_octaves; octave++ )
    {
        Octave& oct_obj = _octaves[octave];

        int* extrema_num_blocks = getNumberOfBlocks( octave );

        int cols = oct_obj.getWidth();
        int rows = oct_obj.getHeight();

        //Get the DoG plane that was allocated on THIS octave's queue
        Plane2D_float& dog = oct_obj.getDog();
       
        // Verify the dog pointer is valid before passing to kernel
        float* dog_test_ptr = (float*)dog.getDevicePtr();
        if(dog_test_ptr == nullptr) {
           POP_FATAL("ERROR: DoG pointer is NULL for octave " << octave);
           continue;
        }

        switch( conf.getSiftMode() )
        {
        case Config::RefineInLevel :
               octave_events.push_back(
                   find_extrema_in_dog<Config::RefineInLevel>
                    ( make_int3(cols, rows, _levels-3),
                      dog,
                      octave,
                      cols,
                      rows,
                      _levels-1,
                      oct_obj.getWGridDivider(),
                      oct_obj.getHGridDivider(),
                      conf.getFilterGridSize(),
                      this ));
                break;
        default :
               octave_events.push_back(
                   find_extrema_in_dog<Config::RefineInOctave>
                    ( make_int3(cols, rows, _levels-3),
                      dog,
                      octave,
                      cols,
                      rows,
                      _levels-1,
                      oct_obj.getWGridDivider(),
                      oct_obj.getHGridDivider(),
                      conf.getFilterGridSize(),
                     this));
                break;
        }

   }

   // Wait for ALL octaves to complete
//   POP_INFO2( false, "Waiting for all " << octave_events.size() << " octave kernels to complete..." );
   for(auto& evt : octave_events) {
       evt.wait();
   }
//   POP_INFO2( false, "All octave extrema detection complete" );

   // Now process results for each octave
   for( int octave=0; octave<_num_octaves; octave++ )
   {
        Octave& oct_obj = _octaves[octave];

        std::vector<InitialExtremum>& i_extrema = dct.initial_extrema_in_octave[octave];
      
        int extrema_count = extrema_count_host[octave];
//        POP_INFO2( false, "Found " << extrema_count << " extrema in octave " << octave );
      

       // Get stored device pointers
       auto [d_extrema, d_count] = getDevicePointers(octave);
   

       if(extrema_count > 0) {
          std::vector<InitialExtremum> host_extrema(extrema_count);
          oct_obj.getQueue().memcpy(host_extrema.data(), d_extrema, 
                                    extrema_count * sizeof(InitialExtremum)).wait();
          
          i_extrema.insert(i_extrema.end(), host_extrema.begin(), host_extrema.end());
      }
      
      // Cleanup device memory
      sycl::free(d_extrema, oct_obj.getQueue());
      sycl::free(d_count, oct_obj.getQueue());
      
      dct.extrema_count_per_octave[octave] = i_extrema.size();
      //POP_INFO2( false, "final extrema count in octave " << octave << ": " << dct.extrema_count_per_octave[octave] );


        bool log_to_file = ( conf.getLogMode() == popsift::Config::All );
        if( log_to_file ) {
            struct stat st = { 0 };

            if (stat("dir-extrema", &st) == -1) {
                mkdir("dir-extrema", 0700);
            }

            std::vector<int2> red_pixel_list;
            for( auto it : dct.initial_extrema_in_octave[octave] )
            {
                red_pixel_list.emplace_back( make_int2( roundf(it.xpos), roundf(it.ypos) ) );
            }

            std::ostringstream ostr;
            ostr << "dir-extrema/" << "pyramid" << "-o-" << octave << "-red" << ".ppm";
            popsift::write_plane2Dppm( ostr.str().c_str(), oct_obj.getData(), red_pixel_list );
        }
    }

//    POP_INFO2( false, "found extrema in all octaves" );
 
    /* Copy the extreme count for every octave from the (already initialized)
     * array extrema_count_per_octave to the (uninitialized) array extrema_count_prefix_sum. */
    dct.extrema_count_prefix_sum.resize( dct.extrema_count_per_octave.size() + 1 );

    auto it = dct.extrema_count_prefix_sum.begin();
    *it = 0;
    it++;

    /* Compute the exclusive prefix sum on the array extrema_count_prefix_sum, but add
     * the total sum in the last element. Easier to achieve with an inclusive_scan. */
    std::inclusive_scan( dct.extrema_count_per_octave.begin(),
                         dct.extrema_count_per_octave.end(),
                         it );

    /* Store the total number of orientations and the total number of
     * extrema as well. */
    dct.extrema_count_total = dct.extrema_count_prefix_sum.back();

    // std::ostringstream debug_ostr;
    // debug_ostr << "Extrema per octave:" << std::endl;
    // std::copy( dct.extrema_count_per_octave.begin(),
    //            dct.extrema_count_per_octave.end(),
    //            std::ostream_iterator<int>(debug_ostr, " ") );
    // debug_ostr << std::endl
    //       << "Extrema prefix sum per octave:" << std::endl;
    // std::copy( dct.extrema_count_prefix_sum.begin(),
    //            dct.extrema_count_prefix_sum.end(),
    //            std::ostream_iterator<int>(debug_ostr, " ") );
    // debug_ostr << std::endl
    //       << "Extrema prefix sum per octave: "
    //       << dct.extrema_count_total
    //       << std::endl;
    // POP_INFO2( false, debug_ostr.str() );
}

} // namespace popsift
