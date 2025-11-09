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


template<int sift_mode>
static
void find_extrema_in_dog( const int3&    g,
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

    std::vector<InitialExtremum>& i_extrema = dct.initial_extrema_in_octave[octave];

    POP_INFO2( no_extrema_reporting, "Converting find_extrema to SYCL kernel for octave " << octave );

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
            
            uint32_t gt = 0;
            uint32_t lt = 0;

            auto extremum_cmp = [&](float f, uint32_t mask) {
                gt |= ((val > f) ? mask : 0);
                lt |= ((val < f) ? mask : 0);
            };

            // 1st group: TX(0,1,1) and TX(2,1,1)
            extremum_cmp(dog_get(level, y, x - 1), 0x00400000);
            extremum_cmp(dog_get(level, y, x + 1), 0x00040000);
            
            if((gt != 0x00440000) && (lt != 0x00440000)) return;

            // 2nd group: TX(1,0,1), TX(1,2,1), TX(1,0,0), TX(1,2,0), TX(1,1,0), TX(1,0,2), TX(1,1,2), TX(1,2,2)
            extremum_cmp(dog_get(level, y - 1, x), 0x00800000);
            extremum_cmp(dog_get(level, y + 1, x), 0x00200000);
            extremum_cmp(dog_get(level - 1, y - 1, x), 0x80000000);
            extremum_cmp(dog_get(level - 1, y + 1, x), 0x40000000);
            extremum_cmp(dog_get(level - 1, y, x), 0x20000000);
            extremum_cmp(dog_get(level + 1, y - 1, x), 0x00008000);
            extremum_cmp(dog_get(level + 1, y, x), 0x00004000);
            extremum_cmp(dog_get(level + 1, y + 1, x), 0x00002000);

            if((gt != 0xe0e4e000) && (lt != 0xe0e4e000)) return;

            // 3rd group: TX(0,0,1), TX(2,0,1), TX(0,2,1), TX(2,2,1)
            extremum_cmp(dog_get(level, y - 1, x - 1), 0x00010000);
            extremum_cmp(dog_get(level, y - 1, x + 1), 0x00020000);
            extremum_cmp(dog_get(level, y + 1, x - 1), 0x00100000);
            extremum_cmp(dog_get(level, y + 1, x + 1), 0x00080000);

            if((gt != 0xe0ffe000) && (lt != 0xe0ffe000)) return;

            // 4th group: TX(0,0,0), TX(2,0,0), TX(0,1,0), TX(2,1,0), TX(0,2,0), TX(2,2,0)
            extremum_cmp(dog_get(level - 1, y - 1, x - 1), 0x01000000);
            extremum_cmp(dog_get(level - 1, y - 1, x + 1), 0x02000000);
            extremum_cmp(dog_get(level - 1, y, x - 1), 0x00000004);
            extremum_cmp(dog_get(level - 1, y, x + 1), 0x04000000);
            extremum_cmp(dog_get(level - 1, y + 1, x - 1), 0x10000000);
            extremum_cmp(dog_get(level - 1, y + 1, x + 1), 0x08000000);

            if((gt != 0xffffe004) && (lt != 0xffffe004)) return;

            // 5th group: TX(0,0,2), TX(2,0,2), TX(0,1,2), TX(2,1,2), TX(0,2,2), TX(2,2,2)
            extremum_cmp(dog_get(level + 1, y - 1, x - 1), 0x00000100);
            extremum_cmp(dog_get(level + 1, y - 1, x + 1), 0x00000200);
            extremum_cmp(dog_get(level + 1, y, x - 1), 0x00000001);
            extremum_cmp(dog_get(level + 1, y, x + 1), 0x00000400);
            extremum_cmp(dog_get(level + 1, y + 1, x - 1), 0x00001000);
            extremum_cmp(dog_get(level + 1, y + 1, x + 1), 0x00000800);

            if((gt != 0xffffff05) && (lt != 0xffffff05)) return;

            // NOW the extremum check passed, use the SAME value v for refinement
            const float v = val;  // This is already dog_get(level, y, x)


            // Refinement loop - use coordinates (x, y, level)
            float3 D; // Dx Dy Ds
            float3 DD; // Dxx Dyy Dss
            float3 DX; // Dxy Dxs Dys
            float3 d; // dx dy ds
            int3 n = {x, y, level};
            int iter = 0;
            const int MAX_ITER = 5;
            

            while(iter < MAX_ITER) {
                iter++;

                // ... gradient and Hessian computation stays the same ...
                
                const float x2y1z1 = dog_get(n.z, n.y, n.x + 1);
                const float x0y1z1 = dog_get(n.z, n.y, n.x - 1);
                const float x1y2z1 = dog_get(n.z, n.y + 1, n.x);
                const float x1y0z1 = dog_get(n.z, n.y - 1, n.x);
                const float x1y1z2 = dog_get(n.z + 1, n.y, n.x);
                const float x1y1z0 = dog_get(n.z - 1, n.y, n.x);

                D.x = 0.5f * (x2y1z1 - x0y1z1);
                D.y = 0.5f * (x1y2z1 - x1y0z1);
                D.z = 0.5f * (x1y1z2 - x1y1z0);

                const float x1y1z1 = dog_get(n.z, n.y, n.x);

                DD.x = x2y1z1 + x0y1z1 - 2.0f * x1y1z1;
                DD.y = x1y2z1 + x1y0z1 - 2.0f * x1y1z1;
                DD.z = x1y1z2 + x1y1z0 - 2.0f * x1y1z1;

                const float x0y0z1 = dog_get(n.z, n.y - 1, n.x - 1);
                const float x0y2z1 = dog_get(n.z, n.y + 1, n.x - 1);
                const float x2y0z1 = dog_get(n.z, n.y - 1, n.x + 1);
                const float x2y2z1 = dog_get(n.z, n.y + 1, n.x + 1);
                const float x0y1z0 = dog_get(n.z - 1, n.y, n.x - 1);
                const float x0y1z2 = dog_get(n.z + 1, n.y, n.x - 1);
                const float x2y1z0 = dog_get(n.z - 1, n.y, n.x + 1);
                const float x2y1z2 = dog_get(n.z + 1, n.y, n.x + 1);
                const float x1y0z0 = dog_get(n.z - 1, n.y - 1, n.x);
                const float x1y0z2 = dog_get(n.z + 1, n.y - 1, n.x);
                const float x1y2z0 = dog_get(n.z - 1, n.y + 1, n.x);
                const float x1y2z2 = dog_get(n.z + 1, n.y + 1, n.x);

                DX.x = 0.25f * (x2y2z1 + x0y0z1 - x0y2z1 - x2y0z1);
                DX.y = 0.25f * (x2y1z2 + x0y1z0 - x0y1z2 - x2y1z0);
                DX.z = 0.25f * (x1y2z2 + x1y0z0 - x1y2z0 - x1y0z2);

                float A[3][3];
                A[0][0] = DD.x; A[0][1] = DX.x; A[0][2] = DX.y;
                A[1][0] = DX.x; A[1][1] = DD.y; A[1][2] = DX.z;
                A[2][0] = DX.y; A[2][1] = DX.z; A[2][2] = DD.z;

                float3 b = {-D.x, -D.y, -D.z};

                // Compute determinants for matrix inversion
                float det0b = -A[1][2] * A[1][2];
                float det0a = A[1][1] * A[2][2];
                float det0 = det0b + det0a;

                float det1b = -A[0][1] * A[2][2];
                float det1a = A[1][2] * A[0][2];
                float det1 = det1b + det1a;

                float det2b = -A[1][1] * A[0][2];
                float det2a = A[0][1] * A[1][2];
                float det2 = det2b + det2a;

                float det3b = -A[0][2] * A[0][2];
                float det3a = A[0][0] * A[2][2];
                float det3 = det3b + det3a;

                float det4b = -A[0][0] * A[1][2];
                float det4a = A[0][1] * A[0][2];
                float det4 = det4b + det4a;

                float det5b = -A[0][1] * A[0][1];
                float det5a = A[0][0] * A[1][1];
                float det5 = det5b + det5a;

                float det = (A[0][0] * det0) + (A[0][1] * det1) + (A[0][2] * det2);

                if(sycl::fabs(det) < 1e-10f) {
                    d.x = 0.0f;
                    d.y = 0.0f;
                    d.z = 0.0f;
                    break;
                }

                float rsd = 1.0f / det;

                // Compute inverse matrix
                float inv[3][3];
                inv[0][0] = det0 * rsd;
                inv[1][0] = det1 * rsd;
                inv[2][0] = det2 * rsd;
                inv[1][1] = det3 * rsd;
                inv[1][2] = det4 * rsd;
                inv[2][2] = det5 * rsd;
                inv[0][1] = inv[1][0];
                inv[0][2] = inv[2][0];
                inv[2][1] = inv[1][2];

                // Multiply inv * b to get solution
                d.x = inv[0][0] * b.x + inv[0][1] * b.y + inv[0][2] * b.z;
                d.y = inv[1][0] * b.x + inv[1][1] * b.y + inv[1][2] * b.z;
                d.z = inv[2][0] * b.x + inv[2][1] * b.y + inv[2][2] * b.z;

                // Match CPU refine logic: on last iteration, don't check for movement
                const bool last_it = (iter == MAX_ITER);
                if(last_it) break;  // CPU returns 0 (continue), but loop ends anyway
                
                int3 t = {0, 0, 0};
                
                // Launch (cols-2)×(rows-2), threads at [1, cols-2]×[1, rows-2]
                // Allow movement to reach [1, cols-1]×[1, rows-1], but ensure neighbors stay valid
                // Max position where we can read n+1 is width-2, so before moving must be < width-2
                t.x = ((d.x >= 0.6f && n.x < c_width - 2) ? 1 : 0) +
                      ((d.x <= -0.6f && n.x > 1) ? -1 : 0);
                t.y = ((d.y >= 0.6f && n.y < c_height - 2) ? 1 : 0) +
                      ((d.y <= -0.6f && n.y > 1) ? -1 : 0);
                
                if constexpr (sift_mode == Config::RefineInOctave) {
                    t.z = ((d.z >= 0.6f && n.z < c_maxlevel - 1) ? 1 : 0) +
                          ((d.z <= -0.6f && n.z > 1) ? -1 : 0);
                }
                
                if(t.x == 0 && t.y == 0 && t.z == 0) break;  // No movement, converged
                
                n.x += t.x;
                n.y += t.y;
                n.z += t.z;
            }

            // Final validation
            if(d.x >= 1.5f || d.y >= 1.5f || d.z >= 1.5f) return;

            const float xn = n.x + d.x;
            const float yn = n.y + d.y;
            const float sn = n.z + d.z;


            if(xn < 0.0f || xn > c_width - 1.0f ||   // Match CPU: allows positions up to width-1
               yn < 0.0f || yn > c_height - 1.0f || 
               sn < -0.0f || sn > c_maxlevel) return;
                

            const float x2y1z1_f = dog_get(n.z, n.y, n.x + 1);
            const float x0y1z1_f = dog_get(n.z, n.y, n.x - 1);
            const float x1y2z1_f = dog_get(n.z, n.y + 1, n.x);
            const float x1y0z1_f = dog_get(n.z, n.y - 1, n.x);
            const float x1y1z2_f = dog_get(n.z + 1, n.y, n.x);
            const float x1y1z0_f = dog_get(n.z - 1, n.y, n.x);
            const float x1y1z1_f = dog_get(n.z, n.y, n.x);

            float3 D_f;
            D_f.x = 0.5f * (x2y1z1_f - x0y1z1_f);
            D_f.y = 0.5f * (x1y2z1_f - x1y0z1_f);
            D_f.z = 0.5f * (x1y1z2_f - x1y1z0_f);

            float3 DD_f;
            DD_f.x = x2y1z1_f + x0y1z1_f - 2.0f * x1y1z1_f;
            DD_f.y = x1y2z1_f + x1y0z1_f - 2.0f * x1y1z1_f;
            DD_f.z = x1y1z2_f + x1y1z0_f - 2.0f * x1y1z1_f;

            // Compute DX_f at final position
            const float x0y0z1_f = dog_get(n.z, n.y - 1, n.x - 1);
            const float x0y2z1_f = dog_get(n.z, n.y + 1, n.x - 1);
            const float x2y0z1_f = dog_get(n.z, n.y - 1, n.x + 1);
            const float x2y2z1_f = dog_get(n.z, n.y + 1, n.x + 1);
           
            float DX_f_x = 0.25f * (x2y2z1_f + x0y0z1_f - x0y2z1_f - x2y0z1_f);

            const float contr = x1y1z1_f + 0.5f * (D_f.x * d.x + D_f.y * d.y + D_f.z * d.z);
            const float tr      = DD_f.x + DD_f.y;
            const float det     = DD_f.x * DD_f.y - DX_f_x * DX_f_x;
            const float edgeval = tr * tr / det;
            
            if(sycl::fabs(contr) < 2.0f * c_threshold) return;
            if(det <= 0.0f) return;

            if(edgeval >= (c_edge_limit + 1.0f) * (c_edge_limit + 1.0f) / c_edge_limit) return;

            // Atomically add extremum
            int write_idx = sycl::atomic_ref<int, sycl::memory_order_acq_rel, sycl::memory_scope::device>(d_count[0]).fetch_add(1);            
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

    event.wait();

    // Copy results back to host
    int extrema_count = 0;
    queue.memcpy(&extrema_count, d_count, sizeof(int)).wait();

    POP_INFO2( no_extrema_reporting, "Found " << extrema_count << " extrema on device" );

    if(extrema_count > 0) {
        std::vector<InitialExtremum> host_extrema(extrema_count);
        queue.memcpy(host_extrema.data(), d_extrema, extrema_count * sizeof(InitialExtremum)).wait();
                
        std::vector<InitialExtremum> unique_extrema;
        for(const auto& ex : host_extrema) {
            unique_extrema.push_back(ex);
        }
                
        // DEBUG: Print extrema positions grouped by level (like CPU version)
        POP_INFO2( false, "Extrema in octave " << octave << " by level:" );
                
        // Group by level (lpos)
        std::map<int, std::vector<InitialExtremum>> by_level;
        for(const auto& ex : unique_extrema) {
            by_level[ex.lpos].push_back(ex);
        }
                
        i_extrema.insert(i_extrema.end(), unique_extrema.begin(), unique_extrema.end());
    }

    // Cleanup
    sycl::free(d_extrema, queue);
    sycl::free(d_count, queue);

    dct.extrema_count_per_octave[octave] = i_extrema.size();

    POP_INFO2( no_extrema_reporting, "final extrema count in octave " << octave << ": " << dct.extrema_count_per_octave[octave] );
}

void Pyramid::find_extrema( const Config& conf )
{
    POP_INFO2( false, "Enter " << __FUNCTION__ );

    dct.extrema_count_per_octave.resize( MAX_OCTAVES );

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
                find_extrema_in_dog<Config::RefineInLevel>
                    ( int3(cols, rows, _levels-3),
                      dog,
                      octave,
                      cols,
                      rows,
                      _levels-1,
                      oct_obj.getWGridDivider(),
                      oct_obj.getHGridDivider(),
                      conf.getFilterGridSize(),
                      this );
                break;
        default :
                find_extrema_in_dog<Config::RefineInOctave>
                    ( int3(cols, rows, _levels-3),
                      dog,
                      octave,
                      cols,
                      rows,
                      _levels-1,
                      oct_obj.getWGridDivider(),
                      oct_obj.getHGridDivider(),
                      conf.getFilterGridSize(),
                      this ); 
                break;
        }

        bool log_to_file = ( conf.getLogMode() == popsift::Config::All );
        if( log_to_file ) {
            struct stat st = { 0 };

            if (stat("dir-extrema", &st) == -1) {
                mkdir("dir-extrema", 0700);
            }

            std::vector<int2> red_pixel_list;
            for( auto it : dct.initial_extrema_in_octave[octave] )
            {
                red_pixel_list.emplace_back( int2( roundf(it.xpos), roundf(it.ypos) ) );
            }

            std::ostringstream ostr;
            ostr << "dir-extrema/" << "pyramid" << "-o-" << octave << "-red" << ".ppm";
            popsift::write_plane2Dppm( ostr.str().c_str(), oct_obj.getData(), red_pixel_list );
        }
    }

    POP_INFO2( false, "found extrema in all octaves" );
 
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

    std::ostringstream debug_ostr;
    debug_ostr << "Extrema per octave:" << std::endl;
    std::copy( dct.extrema_count_per_octave.begin(),
               dct.extrema_count_per_octave.end(),
               std::ostream_iterator<int>(debug_ostr, " ") );
    debug_ostr << std::endl
          << "Extrema prefix sum per octave:" << std::endl;
    std::copy( dct.extrema_count_prefix_sum.begin(),
               dct.extrema_count_prefix_sum.end(),
               std::ostream_iterator<int>(debug_ostr, " ") );
    debug_ostr << std::endl
          << "Extrema prefix sum per octave: "
          << dct.extrema_count_total
          << std::endl;
    POP_INFO2( false, debug_ostr.str() );
}

} // namespace popsift
