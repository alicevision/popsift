#pragma once

#include "sift_extremum.h"
#include "sift_constants.h"
#include "s_gradiant.h"
#include <sycl/sycl.hpp>

namespace popsift {

// Forward declaration for sub-group size query
class sub_group_desc_loop;

// Core descriptor extraction function
template<bool UseLocalAccessor = false, typename... Args>
static inline void ext_desc_loop_sub(
    const float ang,
    const Extremum* ext,
    float* features,
    const float* layer_data,
    const int layer_pitch,
    const int width,
    const int height,
    sycl::nd_item<3> it,
    Args&&... args)
{
    const int ix = it.get_local_id(1);
    const int iy = it.get_local_id(0);
    const int tile = (((iy << 2) + ix) << 3); // Base of 8 floats per sub-block
    
    const float x = ext->xpos;
    const float y = ext->ypos;
    const int level = ext->lpos;
    const float sig = ext->sigma;
    const float SBP = sycl::fabs(DESC_MAGNIFY * sig);
    
    if(SBP == 0.0f) return;
    
    float cos_t = sycl::cos(ang);
    float sin_t = sycl::sin(ang);
    
    const float csbp = cos_t * SBP;
    const float ssbp = sin_t * SBP;
    const float crsbp = cos_t / SBP;
    const float srsbp = sin_t / SBP;
    
    const sycl::vec<float, 2> offsetpt(ix - 1.5f, iy - 1.5f);
    
    const float ptx = sycl::fma(csbp, offsetpt.x(), sycl::fma(-ssbp, offsetpt.y(), x));
    const float pty = sycl::fma(csbp, offsetpt.y(), sycl::fma(ssbp, offsetpt.x(), y));
    
    const float bsz = sycl::fabs(csbp) + sycl::fabs(ssbp);
    const int xmin = sycl::max(1, (int)sycl::floor(ptx - bsz));
    const int ymin = sycl::max(1, (int)sycl::floor(pty - bsz));
    const int xmax = sycl::min(width - 2, (int)sycl::floor(ptx + bsz));
    const int ymax = sycl::min(height - 2, (int)sycl::floor(pty + bsz));
    
    const int wx = xmax - xmin + 1;
    const int hy = ymax - ymin + 1;
    const int loops = wx * hy;
    
    float dpt[9] = {0.0f};
    
    // Sample gradients
    for(int i = it.get_local_id(2); sycl::any_of_group(it.get_sub_group(), i < loops); i += it.get_local_range(2)) {
        if(i >= loops) continue;
        
        const int ii = i / wx + ymin;
        const int jj = i % wx + xmin;
        
        const sycl::vec<float, 2> d(jj - ptx, ii - pty);
        const sycl::vec<float, 2> n(sycl::fma(crsbp, d.x(), srsbp * d.y()), 
                                     sycl::fma(crsbp, d.y(), -srsbp * d.x()));
        const sycl::vec<float, 2> nn = sycl::fabs(n);
        
        if(nn.x() < 1.0f && nn.y() < 1.0f) {
            // Get gradient at this position
            const int offset = level * (height * layer_pitch) + ii * layer_pitch + jj;
            
            const float pix_x_pls = layer_data[offset + 1];
            const float pix_x_min = layer_data[offset - 1];
            const float pix_y_pls = layer_data[offset + layer_pitch];
            const float pix_y_min = layer_data[offset - layer_pitch];
            
            const float dx = (pix_x_pls - pix_x_min) * 0.5f;
            const float dy = (pix_y_pls - pix_y_min) * 0.5f;
            
            float mod = sycl::sqrt(dx * dx + dy * dy);
            float th = sycl::atan2(dy, dx);
            
            const sycl::vec<float, 2> dn = n + offsetpt;
            const float ww = sycl::exp(-sycl::ldexp(sycl::dot(dn, dn), -3));
            const sycl::vec<float, 2> w(1.0f - nn.x(), 1.0f - nn.y());
            const float wgt = ww * w.x() * w.y() * mod;
            
            th -= ang;
            th += (th < 0.0f ? M_PI2 : 0.0f);
            th -= (th >= M_PI2 ? M_PI2 : 0.0f);
            
            const float tth = th * M_4RPI;
            const int fo0 = static_cast<int>(sycl::floor(tth));
            const float do0 = tth - fo0;
            const float wgt1 = 1.0f - do0;
            const float wgt2 = do0;
            
            int fo = fo0 % DESC_BINS;
            
            dpt[fo] = sycl::fma(wgt1, wgt, dpt[fo]);
            dpt[fo + 1] = sycl::fma(wgt2, wgt, dpt[fo + 1]);
        }
    }
    
    sycl::group_barrier(it.get_group());
    
    dpt[0] += dpt[8];
    
    if constexpr(!UseLocalAccessor) {
        // Sub-group reduction path
        for(int i = 0; i < 8; i++) {
            dpt[i] = sycl::reduce_over_group(it.get_sub_group(), dpt[i], sycl::plus<float>());
        }
        
        if(it.get_local_id(2) < 8) {
            features[tile + it.get_local_id(2)] = dpt[it.get_local_id(2)];
        }
    } else {
        // Local memory reduction path
        auto& sum = std::get<0>(std::forward_as_tuple(args...));
        const int base = (it.get_local_linear_id() >> 5) * 39;
        
        for(int i = 0; i < 8; i++) {
            sum[base + it.get_local_id(2) + i] = dpt[i];
            sycl::group_barrier(it.get_group());
            
            for(int stride = it.get_local_range(2) / 2; stride > 0; stride >>= 1) {
                if(it.get_local_id(2) < stride)
                    sum[base + it.get_local_id(2) + i] += sum[base + it.get_local_id(2) + i + stride];
                sycl::group_barrier(it.get_group());
            }
        }
        
        if(it.get_local_id(2) < 8) {
            features[tile + it.get_local_id(2)] = sum[base + it.get_local_id(2)];
        }
    }
}

// Kernel functor for sub-group path
class Ext_desc_loop {
private:
    const int octave;
    const int orientation_offset;
    const int width;
    const int height;
    const int layer_pitch;
    Descriptor* d_descs;
    const Extremum* d_extrema;
    const int* d_feat_to_ext_map;
    const float* d_layer_data;

public:
    Ext_desc_loop(int octave, int orientation_offset, int width, int height,
                  Descriptor* d_descs, const Extremum* d_extrema, 
                  const int* d_feat_to_ext_map, const float* d_layer_data,
                  int layer_pitch)
        : octave(octave), orientation_offset(orientation_offset)
        , width(width), height(height), layer_pitch(layer_pitch)
        , d_descs(d_descs), d_extrema(d_extrema)
        , d_feat_to_ext_map(d_feat_to_ext_map), d_layer_data(d_layer_data)
    {}

    inline void operator()(sycl::nd_item<3> it) const {
        const int ori_idx = it.get_group(2);
        const int o_offset = orientation_offset + ori_idx;
        
        Descriptor* desc = &d_descs[ori_idx];
        const int ext_idx = d_feat_to_ext_map[o_offset];
        const Extremum* ext = &d_extrema[ext_idx];
        
        const int ext_base = ext->idx_ori;
        const int ori_num = o_offset - ext_base;
        const float ang = ext->orientation[ori_num];
        
        ext_desc_loop_sub(ang, ext, desc->features, d_layer_data, layer_pitch,
                         width, height, it);
    }
};

// Kernel functor for local memory path
class Ext_desc_loop_local_mem {
private:
    sycl::local_accessor<float, 1> sum;
    const int octave;
    const int orientation_offset;
    const int width;
    const int height;
    const int layer_pitch;
    Descriptor* d_descs;
    const Extremum* d_extrema;
    const int* d_feat_to_ext_map;
    const float* d_layer_data;

public:
    Ext_desc_loop_local_mem(sycl::local_accessor<float, 1> sum,
                           int octave, int orientation_offset, int width, int height,
                           Descriptor* d_descs, const Extremum* d_extrema,
                           const int* d_feat_to_ext_map, const float* d_layer_data,
                           int layer_pitch)
        : sum(sum), octave(octave), orientation_offset(orientation_offset)
        , width(width), height(height), layer_pitch(layer_pitch)
        , d_descs(d_descs), d_extrema(d_extrema)
        , d_feat_to_ext_map(d_feat_to_ext_map), d_layer_data(d_layer_data)
    {}

    inline void operator()(sycl::nd_item<3> it) const {
        const int ori_idx = it.get_group(2);
        const int o_offset = orientation_offset + ori_idx;
        
        Descriptor* desc = &d_descs[ori_idx];
        const int ext_idx = d_feat_to_ext_map[o_offset];
        const Extremum* ext = &d_extrema[ext_idx];
        
        const int ext_base = ext->idx_ori;
        const int ori_num = o_offset - ext_base;
        const float ang = ext->orientation[ori_num];
        
        ext_desc_loop_sub<true>(ang, ext, desc->features, d_layer_data, layer_pitch,
                               width, height, it, sum);
    }
};


} // namespace popsift