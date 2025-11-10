/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/debug_macros.h"
#include "common/excl_blk_prefix_sum.h"
#include "common/warp_bitonic_sort.h"
#include "s_gradiant.h"
#include "sift_config.h"
#include "sift_constants.h"
#include "sift_pyramid.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <iterator>

using namespace popsift;
using namespace std;

/* Smoothing like VLFeat is the default mode.
 * If you choose to undefine it, you get the smoothing approach taken by OpenCV
 */
#define WITH_VLFEAT_SMOOTHING

namespace popsift
{

inline float compute_angle( int bin, float hc, float hn, float hp )
{
    /* interpolate */
    float di = bin + 0.5f * (hn - hp) / (hc+hc-hn-hp);

    /* clamp */
    di = (di < 0) ? 
            (di + ORI_NBINS) : 
            ((di >= ORI_NBINS) ? (di - ORI_NBINS) : (di));

    // float th = __fdividef( M_PI2 * di, ORI_NBINS ) - M_PI;
    float th = ((M_PI2 * di) / ORI_NBINS) - M_PI;
    return th;
}

/*
 * Histogram smoothing helper
 */
inline static
float smoothe( const std::vector<float>& src, const int bin )
{
    const int prev = (bin <= 0) ? ORI_NBINS-1 : bin-1;
    const int next = (bin >= ORI_NBINS-1) ? 0 : bin+1;

    const float f  = ( src.at(prev) + src.at(bin) + src.at(next) ) / 3.0f;

    return f;
}


static sycl::event compute_orientations_sycl(
    sycl::queue& q,
    const int octave,
    const float* d_layer_data,  // Device pointer to layer data
    const int layer_pitch,      // Pitch in elements, not bytes
    const int num_levels,
    const int width,
    const int height,
    const InitialExtremum* d_initial_extrema,
    const int extrema_count,
    Extremum* d_output_extrema,
    int* d_extrema_counter,
    sycl::event depends_on = {})  // Optional dependency event
{
    const int ORI_NBINS_LOCAL = ORI_NBINS;
    const float ORI_WINFACTOR_LOCAL = ORI_WINFACTOR;
    
    auto event = q.submit([&](sycl::handler& cgh) {
        // If there's a dependency, wait for it
        if(depends_on != sycl::event{}) {
            cgh.depends_on(depends_on);
        }
  
        cgh.parallel_for(sycl::range<1>(extrema_count), [=](sycl::id<1> idx) {
            const int extremum_index = idx[0];
            
            const InitialExtremum iext = d_initial_extrema[extremum_index];
            
            float hist[36];  // ORI_NBINS = 36
            for(int i = 0; i < 36; i++) hist[i] = 0.0f;
            
            const float x = iext.xpos;
            const float y = iext.ypos;
            const int level = iext.lpos;
            const float sig = iext.sigma;
            
            const float sigw = ORI_WINFACTOR_LOCAL * sig;
            const int rad = (int)sycl::round(3.0f * sigw);
            const float factor = -0.5f / (sigw * sigw);
            const int sq_thres = rad * rad;
            
            int xmin = sycl::max(1, (int)sycl::round(x) - rad);
            int xmax = sycl::min(width - 2, (int)sycl::round(x) + rad);
            int ymin = sycl::max(1, (int)sycl::round(y) - rad);
            int ymax = sycl::min(height - 2, (int)sycl::round(y) + rad);
            
            // Build histogram
            for(int yy = ymin; yy <= ymax; yy++) {
                for(int xx = xmin; xx <= xmax; xx++) {
                    const float dx = xx - x;
                    const float dy = yy - y;
                    const int sq_dist = (int)(dx * dx + dy * dy);
                    
                    if(sq_dist <= sq_thres) {
                        // Data layout: level * (height * pitch) + y * pitch + x
                        const int offset = level * height * layer_pitch + yy * layer_pitch + xx;
                        
                        const float Dx = d_layer_data[offset + 1] - d_layer_data[offset - 1];
                        const float Dy = d_layer_data[offset + layer_pitch] - d_layer_data[offset - layer_pitch];
                        
                        const float grad = sycl::sqrt(Dx * Dx + Dy * Dy);
                        const float theta = sycl::atan2(Dy, Dx);
                        
                        const float weight = grad * sycl::exp((float)sq_dist * factor);
                        
                        int bidx = (int)sycl::round(36.0f * (theta + M_PI) / M_PI2);
                        bidx = (bidx == 36) ? 0 : bidx;
                        bidx = sycl::clamp(bidx, 0, 35);
                        
                        hist[bidx] += weight;
                    }
                }
            }
            
            // Smooth histogram (VLFeat: 6 passes = 3 iterations of double-smoothing)
            float sm_hist[36];
            float temp_hist[36];
            
            for(int i = 0; i < 36; i++) temp_hist[i] = hist[i];
            
            for(int iter = 0; iter < 6; iter++) {
                for(int bin = 0; bin < 36; bin++) {
                    const int prev = (bin == 0) ? 35 : bin - 1;
                    const int next = (bin == 35) ? 0 : bin + 1;
                    sm_hist[bin] = (temp_hist[prev] + temp_hist[bin] + temp_hist[next]) / 3.0f;
                }
                
                for(int bin = 0; bin < 36; bin++) {
                    const int prev = (bin == 0) ? 35 : bin - 1;
                    const int next = (bin == 35) ? 0 : bin + 1;
                    temp_hist[bin] = (sm_hist[prev] + sm_hist[bin] + sm_hist[next]) / 3.0f;
                }
            }
            
            for(int i = 0; i < 36; i++) sm_hist[i] = temp_hist[i];
            
            // Sub-cell refinement
            float yval[36];
            float refined_angle[36];
            
            for(int bin = 0; bin < 36; bin++) {
                const int prev = (bin == 0) ? 35 : bin - 1;
                const int next = (bin == 35) ? 0 : bin + 1;
                
                bool predicate = (sm_hist[bin] > sycl::max(sm_hist[prev], sm_hist[next]));
                
                const float num = predicate ? 3.0f * sm_hist[prev] - 4.0f * sm_hist[bin] + 1.0f * sm_hist[next] : 0.0f;
                const float denB = predicate ? 2.0f * (sm_hist[prev] - 2.0f * sm_hist[bin] + sm_hist[next]) : 1.0f;
               
               // CRITICAL: Invalidate predicate if denB is near zero (flat histogram)
               predicate = predicate && (sycl::fabs(denB) > 1e-6f);
               
               const float newbin = predicate ? (num / denB) : 0.0f;
                
                predicate = (predicate && newbin >= 0.0f && newbin <= 2.0f);
                
                refined_angle[bin] = predicate ? (float)prev + newbin : -1.0f;
               yval[bin] = predicate ? -(num * num) / (4.0f * denB) + sm_hist[prev] : -INFINITY;
            }
            
            // Find max value for threshold
            float max_yval = yval[0];
            for(int i = 1; i < 36; i++) {
                if(yval[i] > max_yval) max_yval = yval[i];
            }
            const float acceptance_threshold = 0.8f * max_yval;
            
            // Find and sort peaks
            Extremum ext;
            ext.xpos = iext.xpos;
            ext.ypos = iext.ypos;
            ext.lpos = iext.lpos;
            ext.sigma = iext.sigma;
            ext.octave = octave;
            
            int angles = 0;
            for(int i = 0; i < 36 && angles < ORIENTATION_MAX_COUNT; i++) {
                // Find max
                int max_idx = -1;
                float max_val = -INFINITY;
                for(int j = 0; j < 36; j++) {
                    if(yval[j] > max_val) {
                        max_val = yval[j];
                        max_idx = j;
                    }
                }
                
                if(max_val >= acceptance_threshold && max_idx >= 0) {
                    float chosen_bin = refined_angle[max_idx];
                    if(chosen_bin >= 36.0f) chosen_bin -= 36.0f;
                    float th = M_PI2 * chosen_bin / 36.0f - M_PI;
                    ext.orientation[angles] = th;
                    angles++;
                    yval[max_idx] = -INFINITY;
                } else {
                    break;
                }
            }
            
            ext.num_ori = angles;
            
            // Atomic increment and write
            sycl::atomic_ref<int, 
                sycl::memory_order::relaxed,
                sycl::memory_scope::device,
                sycl::access::address_space::global_space> atomic_counter(*d_extrema_counter);
            int output_idx = atomic_counter.fetch_add(1);
            
            d_output_extrema[output_idx] = ext;
        });
    });
    
    return event;  // Return event for async execution
}


}; // namespace popsift

class ExtremaRead
{
    const Extremum* const _oris;
public:
    inline 
    explicit ExtremaRead( const Extremum* const d_oris ) : _oris( d_oris ) { }

    inline 
    int get( int n ) const { return _oris[n].num_ori; }
};

class ExtremaWrt
{
    Extremum* _oris;
public:
    inline 
    explicit ExtremaWrt( Extremum* d_oris ) : _oris( d_oris ) { }

    inline 
    void set( int n, int value ) { _oris[n].idx_ori = value; }
};

class ExtremaTot
{
    int& _extrema_counter;
public:
    inline 
    explicit ExtremaTot( int& extrema_counter ) : _extrema_counter( extrema_counter ) { }

    inline 
    void set( int value ) { _extrema_counter = value; }
};

class ExtremaWrtMap
{
    int* _featvec_to_extrema_mapper;
    int  _max_feat;
public:
    inline 
    ExtremaWrtMap( int* featvec_to_extrema_mapper, int max_feat )
        : _featvec_to_extrema_mapper( featvec_to_extrema_mapper )
        , _max_feat( max_feat )
    { }

    inline 
    void set( int base, int num, int value )
    {
        int* baseptr = &_featvec_to_extrema_mapper[base];
        do {
            num--;
            if( base + num < _max_feat ) {
                baseptr[num] = value;
            }
        } while( num > 0 );
    }
};

void ori_prefix_sum( const int num_octaves )
{
    if( dct.extrema_count_total < 1 )
    {
        POP_FATAL("Calling " << __FUNCTION__ << " with " << dct.extrema_count_total << " found extrema");
    }

    std::vector<Extremum>& all_extrema = dbuf.extrema;

    assert( all_extrema.size() == dct.extrema_count_total );

    vector<int> ori_count;

    for( auto ext : all_extrema )
    {
        ori_count.push_back( ext.num_ori );
    }

    if( ori_count.size() != dct.extrema_count_total )
    {
        POP_FATAL( "Number of counted extrema: " << dct.extrema_count_total << ", orientation counters pushed to ori_count: " << ori_count.size() );
    }

    /* set ori_offset[0] to 0,
     * then compute an inclusive prefix sum for the values in ori_count into
     * the target array ori_offset, but starting at offset 1 instead of 0.
     * The last entry of ori_offset is the total number of orientations, which
     * we also want to keep.
     */
    vector<int> ori_offset( dct.extrema_count_total+1 );
    ori_offset[0] = 0;
    std::inclusive_scan( ori_count.begin(), ori_count.end(), &ori_offset[1] );
    const int total_ori = ori_offset[dct.extrema_count_total];

    POP_INFO2( false, "Total number of orientations: " << total_ori );

    for( int i=0; i<dct.extrema_count_total; i++ )
    {
        all_extrema[i].idx_ori = ori_offset[i];
    }

    /* For every orientation (there are total_ori of them), store the extremum
     * to which they belong in the array feat_to_ext_map. */
    std::vector<int>& feat_to_ext_map = dbuf.feat_to_ext_map;

    if( dbuf.feat_to_ext_map.size() != 0 )
    {
        POP_FATAL( "Programming error in " << __FILE__ << ":" << __LINE__ << ": reverse map size should be 0 but it is " << dbuf.feat_to_ext_map.size() );
    }

    for( int extr=0; extr<dct.extrema_count_total; extr++ )
    {
        for( int ori=0; ori<ori_count[extr]; ori++ )
        {
            feat_to_ext_map.push_back( extr );
        }
    }


    /* Fill the array ori_ct with the number of orientations that belong
     * to the octave given by the index. */
    for( int o=0; o<MAX_OCTAVES; o++ ) {
        if( dct.extrema_count_per_octave[o] == 0 ) {
            dct.ori_ct[o] = 0;
        } else {
            int fe = dct.extrema_count_prefix_sum[o  ];   /* first extremum for this octave */
            int le = dct.extrema_count_prefix_sum[o+1]-1; /* last  extremum for this octave */
            int lo_ori_index = dbuf.extrema[fe].idx_ori;
            int num_ori      = dbuf.extrema[le].num_ori;
            int hi_ori_index = dbuf.extrema[le].idx_ori + num_ori;
            dct.ori_ct[o] = hi_ori_index - lo_ori_index;
        }
    }

    /* Like above, compute the exclusive prefix sum of all orientations
     * in ori_ps. */
#if 1
    std::cerr << "Orientations by octave (dct.ori_ct): ";
    for( int i=0; i<MAX_OCTAVES; i++ ) std::cerr << dct.ori_ct[i] << " ";
    std::cerr << std::endl;
#endif

    std::copy( &dct.ori_ct[0],
               &dct.ori_ct[MAX_OCTAVES],
               &dct.ori_ps[0] );
#if 1
    std::cerr << "Orientations by octave (dct.ori_ps): ";
    for( int i=0; i<MAX_OCTAVES; i++ ) std::cerr << dct.ori_ps[i] << " ";
    std::cerr << std::endl;
#endif
    std::exclusive_scan( &dct.ori_ps[0],
                         &dct.ori_ps[MAX_OCTAVES],
                         &dct.ori_ps[0],
                         0 );
#if 1
    std::cerr << "Exclusive prefix sum of orientations (dct.ori_ps): ";
    for( int i=0; i<MAX_OCTAVES; i++ ) std::cerr << dct.ori_ps[i] << " ";
    std::cerr << std::endl;
#endif

    /* Store the total number of orientations and the total number of
     * extrema as well. */
    dct.ori_total = dct.ori_ps[MAX_OCTAVES-1] + dct.ori_ct[MAX_OCTAVES-1];
}

void Pyramid::orientation( const Config& conf )
{
    POP_INFO2( conf.silent(), "enter " << __PRETTY_FUNCTION__ );

    int debug_octave = 0;
    int ext_total = 0;
    for( int n : dct.extrema_count_per_octave )
    {
        POP_INFO2( conf.silent(), "octave " << debug_octave << " has " << n << " extrema" );
        if( n > 0 )
        {
            ext_total += n;
        }
        debug_octave++;
    }
    POP_INFO2( conf.silent(), "total number of extrema is " << ext_total );

    if(ext_total == 0) {
        POP_INFO2( conf.silent(), "No extrema found, skipping orientation" );
        dct.ori_total = 0;
        return;
    }

    if( conf.getFilterMaxExtrema() > 0 && int(conf.getFilterMaxExtrema()*1.1) < ext_total )
    {
        ext_total = extrema_filter_grid( conf, ext_total );
    }

    reallocExtrema( ext_total );

    sycl::queue& q = _octaves[0].getQueue();
    int* d_counter = sycl::malloc_device<int>(1, q);
    Extremum* d_output_extrema = sycl::malloc_device<Extremum>(ext_total * ORIENTATION_MAX_COUNT, q);
    
    q.memset(d_counter, 0, sizeof(int)).wait();

   // Store events and device pointers for async execution
   std::vector<sycl::event> orientation_events;
   std::vector<InitialExtremum*> d_extrema_ptrs(_num_octaves, nullptr);
   orientation_events.reserve(_num_octaves);

   // Launch all octave orientation kernels asynchronously
   for( int octave=0; octave<_num_octaves; octave++ )
    {
        Octave& oct_obj = _octaves[octave];
        int extrema_count = dct.extrema_count_per_octave[octave];

        if( extrema_count > 0 )
        {
            POP_INFO2( conf.silent(), "Computing orientations for octave " << octave << " with " << extrema_count << " extrema" );
            
            const std::vector<InitialExtremum>& h_extrema = dct.initial_extrema_in_octave[octave];
           d_extrema_ptrs[octave] = sycl::malloc_device<InitialExtremum>(extrema_count, q);
           
           // Copy extrema to device (async)
           auto copy_event = q.memcpy(d_extrema_ptrs[octave], h_extrema.data(), extrema_count * sizeof(InitialExtremum));
            
           // Launch orientation kernel (depends on copy_event)
           auto kernel_event = compute_orientations_sycl(
                oct_obj.getQueue(),
                octave,
                oct_obj.getData().getDevPtr(),
                oct_obj.getData().getCols(),
                oct_obj.getLevels(),
                oct_obj.getWidth(),
                oct_obj.getHeight(),
                d_extrema_ptrs[octave],
                extrema_count,
                d_output_extrema,
                d_counter,
                copy_event);  // Pass dependency
            
            orientation_events.push_back(kernel_event);
        }
    }

    // Wait for ALL orientation kernels to complete
   POP_INFO2( conf.silent(), "Waiting for all orientation kernels to complete..." );
   for(auto& evt : orientation_events) {
       evt.wait();
   }
   POP_INFO2( conf.silent(), "All orientation computation complete" );
   
   // Free device memory for extrema inputs
   for(int octave=0; octave<_num_octaves; octave++) {
       if(d_extrema_ptrs[octave] != nullptr) {
           sycl::free(d_extrema_ptrs[octave], q);
       }
   }

    int h_counter;
    q.memcpy(&h_counter, d_counter, sizeof(int)).wait();
        
    dbuf.extrema.resize(h_counter);
    q.memcpy(dbuf.extrema.data(), d_output_extrema, h_counter * sizeof(Extremum)).wait();
    
    sycl::free(d_counter, q);
    sycl::free(d_output_extrema, q);

    ori_prefix_sum( _num_octaves );
    
}