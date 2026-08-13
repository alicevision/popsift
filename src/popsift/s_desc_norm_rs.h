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

    // The normalize block is (32,32) and each threadIdx.y row holds one
    // descriptor. The shuffle width is that row width, not the hardware warp
    // size: on a 64-lane wavefront two rows share a wavefront, and an
    // unrestricted reduction and lane-0 broadcast would mix the two descriptors.
    sum += popsift::shuffle_down( sum, 16, 32 );
    sum += popsift::shuffle_down( sum,  8, 32 );
    sum += popsift::shuffle_down( sum,  4, 32 );
    sum += popsift::shuffle_down( sum,  2, 32 );
    sum += popsift::shuffle_down( sum,  1, 32 );

    sum = popsift::shuffle( sum,  0, 32 );

    // RootSift takes sqrt(bin/sum). A descriptor bin cannot be negative, so a
    // non-positive sum means an all-zero descriptor, which stays all-zero
    // instead of dividing. The per-bin test keeps a bin that came out slightly
    // negative (round-to-nearest weight accumulation on platforms without the
    // round-toward-+inf intrinsics) out of the square root.
    const float inv = ( sum > 0.0f ) ? __fdividef( 1.0f, sum ) : 0.0f;

    if( inv <= 0.0f )
    {
        descr.x = descr.y = descr.z = descr.w = 0.0f;
    }
    else
    {
        descr.x = descr.x <= 0.0f ? 0.0f : scalbnf( __fsqrt_rn( descr.x * inv ), d_consts.norm_multi );
        descr.y = descr.y <= 0.0f ? 0.0f : scalbnf( __fsqrt_rn( descr.y * inv ), d_consts.norm_multi );
        descr.z = descr.z <= 0.0f ? 0.0f : scalbnf( __fsqrt_rn( descr.z * inv ), d_consts.norm_multi );
        descr.w = descr.w <= 0.0f ? 0.0f : scalbnf( __fsqrt_rn( descr.w * inv ), d_consts.norm_multi );
    }

    if( ! ignoreme ) {
        float4* out4 = (float4*)dst_desc;
        out4[threadIdx.x] = descr;
    }
}

