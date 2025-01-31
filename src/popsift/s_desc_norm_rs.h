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
    static inline
    void normalize( Grid& g, float* features, bool ignoreme );

    static inline
    void normalize( Grid& g, const float* src_desc, float* dest_desc, bool ignoreme);
};

inline
void NormalizeRootSift::normalize( Grid& g, float* features, bool ignoreme )
{
    normalize( g, features, features, ignoreme );
}

inline
void NormalizeRootSift::normalize( Grid& g, const float* src_desc, float* dst_desc, bool ignoreme )
{
  float sum = 0;

  g.resetThreadX();
  while( g.nextThreadX() )
  {
    const float4* ptr4 = (const float4*)src_desc;

    float4 descr;
    descr = ptr4[g.threadIdx.x];

    sum += descr.x + descr.y + descr.z + descr.w;
  }

  g.resetThreadX();
  while( g.nextThreadX() )
  {
    float val;
    val = scalbnf( sqrtf( descr.x / sum ),
                   d_consts.norm_multi );
    descr.x = val;
    val = scalbnf( sqrtf( descr.y / sum ),
                   d_consts.norm_multi );
    descr.y = val;
    val = scalbnf( sqrtf( descr.z / sum ),
                   d_consts.norm_multi );
    descr.z = val;
    val = scalbnf( sqrtf( descr.w / sum ),
                   d_consts.norm_multi );
    descr.w = val;

    if( ! ignoreme ) {
        float4* out4 = (float4*)dst_desc;
        out4[g.threadIdx.x] = descr;
    }
  }
}

