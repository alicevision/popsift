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
    static inline
    void normalize( Grid& g, float* features, const bool ignoreme );

    static inline
    void normalize( Grid& g, 
                    const float* src_desc,
                    float*       dest_desc,
                    const  bool  ignoreme );
};

inline
void NormalizeL2::normalize( Grid& g, float* features, const bool ignoreme )
{
    normalize( g, features, features, ignoreme );
}

inline
void NormalizeL2::normalize( Grid& g, const float* src_desc, float* dst_desc, const bool ignoreme )
{
  g.resetThreadX();
  while( g.nextThreadX() )
  {
    const float4* ptr4 = (const float4*)src_desc;

    float4 descr;
    descr = ptr4[g.threadIdx.x];

    float norm[32];

    // 32 threads compute 4 squares each, then shuffle to performing an addition by
    // reduction for the sum of 128 squares, result in thread 0
    norm[g.threadIdx.x] = descr.x * descr.x
                        + descr.y * descr.y
                        + descr.z * descr.z
                        + descr.w * descr.w;
  }

  g.resetThreadX();
  g.nextThreadX();
  while( g.nextThreadX() )
  {
    norm[0] += norm[g.threadIdx.x];
  }

  // compute 1 / sqrt(sum) in round-to-nearest even mode in thread 0
  norm[0] = frsqrtf( norm[0] );

  // spread the inverted norm from thread 0 to all threads in the warp
  g.resetThreadX();
  g.nextThreadX();
  while( g.nextThreadX() )
  {
    norm[threadIdx.x] = norm[0];
  }

  g.resetThreadX();
  while( g.nextThreadX() )
  {
    // quasi-normalize all 128 floats
    descr.x = min( descr.x*norm[g.threadIdx.x], 0.2f );
    descr.y = min( descr.y*norm[g.threadIdx.x], 0.2f );
    descr.z = min( descr.z*norm[g.threadIdx.x], 0.2f );
    descr.w = min( descr.w*norm[g.threadIdx.x], 0.2f );

    // Repeat the procedure, but also add a multiplier. E.g., if the user wants to
    // descriptors as bytes rather than floats, multiply by 256 - or even by 512
    // for better accuracy, which is OK because a point cannot be a keypoint if more
    // than half of its gradient is in a single direction.
    norm[g.threadIdx.x] = descr.x * descr.x
                        + descr.y * descr.y
                        + descr.z * descr.z
                        + descr.w * descr.w;
  }

  g.resetThreadX();
  g.nextThreadX();
  while( g.nextThreadX() )
  {
    norm[0] += norm[g.threadIdx.x];
  }

  norm[0] = frsqrtf( norm[0] ); // inverse square root
  norm[0] = scalbnf( norm[0], d_consts.norm_multi );

  g.resetThreadX();
  g.nextThreadX();
  while( g.nextThreadX() )
  {
    norm[threadIdx.x] = norm[0];
  }

  g.resetThreadX();
  while( g.nextThreadX() )
  {
    descr.x = descr.x * norm[g.threadIdx.x];
    descr.y = descr.y * norm[g.threadIdx.x];
    descr.z = descr.z * norm[g.threadIdx.x];
    descr.w = descr.w * norm[g.threadIdx.x];

    if( ! ignoreme ) {
        float4* out4 = (float4*)dst_desc;
        out4[threadIdx.x] = descr;
    }
  }
}

