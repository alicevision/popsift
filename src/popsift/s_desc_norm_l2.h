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
#include "sift_config.h"

using namespace popsift;
using namespace std;

class NormalizeL2
{
public:
    __device__ static inline
    void normalize( float* features, const bool ignoreme );

    __device__ static inline
    void normalize_restrict( const float* __restrict__ src_desc,
                             float* __restrict__       dest_desc );

    __device__ static inline
    void normalize( const float* src_desc,
                    float*       dest_desc,
                    const  bool  ignoreme );
};

__device__ inline
void NormalizeL2::normalize( float* features, const bool ignoreme )
{
    normalize( features, features, ignoreme );
}

__device__ inline
void NormalizeL2::normalize_restrict( const float* __restrict__ src_desc,
                                      float* __restrict__       dst_desc )
{
    normalize( src_desc, dst_desc, false );
}

__device__ inline
void NormalizeL2::normalize( const float* src_desc, float* dst_desc, const bool ignoreme )
{
    const float4* ptr4 = (const float4*)src_desc;

    float4 descr;
    descr = ptr4[threadIdx.x];

#if POPSIFT_IS_DEFINED(POPSIFT_HAVE_NORMF)
    // normf() is an elegant function: sqrt(sum_0^127{v^2})
    // It exists from CUDA 7.5 but the trouble with CUB on the GTX 980 Ti forces
    // us to with CUDA 7.0 right now

    float norm;

    if( threadIdx.x == 0 ) {
        norm = rnormf( 128, src_desc ); // 1/sqrt(sum of squares)
    }
    __syncthreads();
    norm = popsift::shuffle( norm, 0 );

    descr.x = min( descr.x*norm, 0.2f );
    descr.y = min( descr.y*norm, 0.2f );
    descr.z = min( descr.z*norm, 0.2f );
    descr.w = min( descr.w*norm, 0.2f );

    norm = descr.x * descr.x
         + descr.y * descr.y
         + descr.z * descr.z
         + descr.w * descr.w;
    norm += popsift::shuffle_down( norm, 16 );
    norm += popsift::shuffle_down( norm,  8 );
    norm += popsift::shuffle_down( norm,  4 );
    norm += popsift::shuffle_down( norm,  2 );
    norm += popsift::shuffle_down( norm,  1 );
    if( threadIdx.x == 0 ) {
        // norm = __fsqrt_rn( norm );
        // norm = __fdividef( 512.0f, norm );
        norm = __frsqrt_rn( norm ); // inverse square root
        norm = scalbnf( norm, d_consts.norm_multi );
    }
#else // not HAVE_NORMF
    float norm;

    norm = descr.x * descr.x
         + descr.y * descr.y
         + descr.z * descr.z
         + descr.w * descr.w;
    norm += popsift::shuffle_down( norm, 16 );
    norm += popsift::shuffle_down( norm,  8 );
    norm += popsift::shuffle_down( norm,  4 );
    norm += popsift::shuffle_down( norm,  2 );
    norm += popsift::shuffle_down( norm,  1 );
    if( threadIdx.x == 0 ) {
        norm = __frsqrt_rn( norm );
    }
    norm = popsift::shuffle( norm,  0 );

    descr.x = min( descr.x*norm, 0.2f );
    descr.y = min( descr.y*norm, 0.2f );
    descr.z = min( descr.z*norm, 0.2f );
    descr.w = min( descr.w*norm, 0.2f );

    norm = descr.x * descr.x
         + descr.y * descr.y
         + descr.z * descr.z
         + descr.w * descr.w;
    norm += popsift::shuffle_down( norm, 16 );
    norm += popsift::shuffle_down( norm,  8 );
    norm += popsift::shuffle_down( norm,  4 );
    norm += popsift::shuffle_down( norm,  2 );
    norm += popsift::shuffle_down( norm,  1 );
    if( threadIdx.x == 0 ) {
        // norm = __fsqrt_rn( norm );
        // norm = __fdividef( 512.0f, norm );
        norm = __frsqrt_rn( norm ); // inverse square root
        norm = scalbnf( norm, d_consts.norm_multi );
    }
#endif // HAVE_NORMF
    norm = popsift::shuffle( norm,  0 );

    descr.x = descr.x * norm;
    descr.y = descr.y * norm;
    descr.z = descr.z * norm;
    descr.w = descr.w * norm;

    if( ! ignoreme ) {
        float4* out4 = (float4*)dst_desc;
        out4[threadIdx.x] = descr;
    }
}

