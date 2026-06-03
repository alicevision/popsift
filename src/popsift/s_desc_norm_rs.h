/*
 * Copyright 2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once
#include "common/assist.h"
#include "s_desc_normalize.h"

using namespace popsift;
using namespace std;

class NormalizeRootSift
{
public:
    __device__ static inline
    void normalize( float* features, bool ignoreme );

    __device__ static inline
    void normalize_restrict( const float* __restrict__ src_desc,
                             float* __restrict__       dest_desc );

    __device__ static inline void normalize(const float* src_desc, float* dest_desc, bool ignoreme);
};

__device__ inline
void NormalizeRootSift::normalize( float* features, bool ignoreme )
{
    normalize( features, features, ignoreme );
}

__device__ inline
void NormalizeRootSift::normalize_restrict( const float* __restrict__ src_desc,
                                            float* __restrict__       dst_desc )
{
    normalize( src_desc, dst_desc, false );
}

__device__ inline
void NormalizeRootSift::normalize( const float* src_desc, float* dst_desc, bool ignoreme )
{
    const float4* ptr4 = (const float4*)src_desc;

    float4 descr;
    descr = ptr4[threadIdx.x];

    float sum = descr.x + descr.y + descr.z + descr.w;

    // 32-lane reduction over threadIdx.x; confine to a width-32 sub-group on a
    // 64-lane wavefront (normalize block is (32,32), one descriptor per row).
    // CUDA unchanged.
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
    sum += popsift::shuffle_down( sum, 16, 32 );
    sum += popsift::shuffle_down( sum,  8, 32 );
    sum += popsift::shuffle_down( sum,  4, 32 );
    sum += popsift::shuffle_down( sum,  2, 32 );
    sum += popsift::shuffle_down( sum,  1, 32 );

    sum = popsift::shuffle( sum,  0, 32 );
#else
    sum += popsift::shuffle_down( sum, 16 );
    sum += popsift::shuffle_down( sum,  8 );
    sum += popsift::shuffle_down( sum,  4 );
    sum += popsift::shuffle_down( sum,  2 );
    sum += popsift::shuffle_down( sum,  1 );

    sum = popsift::shuffle( sum,  0 );
#endif

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
    // RootSift takes sqrt(bin/sum). The fmaxf(.,0) below clamps a bin that came out
    // slightly negative from the descriptor accumulation's round-toward-+inf
    // intrinsics (mapped to round-to-nearest in cuda_to_hip.h; a negative bin is
    // unphysical). The divisor is gated at a small threshold so a degenerate
    // near-zero sum is treated as an all-zero descriptor: an all-flat window
    // (sum==0) normalizes to 0, and a tiny subnormal sum can no longer make 1/sum
    // overflow to +inf (which would normalize a positive bin to +inf). CUDA, where
    // the directed-rounding intrinsics exist and bins stay non-negative, is unchanged.
    const float inv = ( sum > 1e-20f ) ? __fdividef( 1.0f, sum ) : 0.0f;
    float val;
    val = scalbnf( __fsqrt_rn( fmaxf( descr.x * inv, 0.0f ) ), d_consts.norm_multi );
    descr.x = val;
    val = scalbnf( __fsqrt_rn( fmaxf( descr.y * inv, 0.0f ) ), d_consts.norm_multi );
    descr.y = val;
    val = scalbnf( __fsqrt_rn( fmaxf( descr.z * inv, 0.0f ) ), d_consts.norm_multi );
    descr.z = val;
    val = scalbnf( __fsqrt_rn( fmaxf( descr.w * inv, 0.0f ) ), d_consts.norm_multi );
    descr.w = val;
#else
    float val;
    val = scalbnf( __fsqrt_rn( __fdividef( descr.x, sum ) ),
                   d_consts.norm_multi );
    descr.x = val;
    val = scalbnf( __fsqrt_rn( __fdividef( descr.y, sum ) ),
                   d_consts.norm_multi );
    descr.y = val;
    val = scalbnf( __fsqrt_rn( __fdividef( descr.z, sum ) ),
                   d_consts.norm_multi );
    descr.z = val;
    val = scalbnf( __fsqrt_rn( __fdividef( descr.w, sum ) ),
                   d_consts.norm_multi );
    descr.w = val;
#endif

    if( ! ignoreme ) {
        float4* out4 = (float4*)dst_desc;
        out4[threadIdx.x] = descr;
    }
}

