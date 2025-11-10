/*
 * Copyright 2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once
#include "common/assist.h"
#include "sift_config.h"
#include "sift_extremum.h"
#include <sycl/sycl.hpp>
#include <cmath>

using namespace popsift;
using namespace std;

class NormalizeRootSift
{
public:
    static inline
    void normalize( float* features );

    static inline
    void normalize( const float* src_desc, float* dest_desc );
};

inline
void NormalizeRootSift::normalize( float* features )
{
    normalize( features, features );
}

inline
void NormalizeRootSift::normalize( const float* src_desc, float* dst_desc )
{
    float sum = 0;

    for( int i=0; i<128; i++ )
    {
        sum += src_desc[i];
    }

    for( int i=0; i<128; i++ )
    {

        dst_desc[i] = scalbnf( sqrtf( src_desc[i] / sum ),
                              h_consts.norm_multi);
    }
}

// SYCL kernel for RootSIFT normalization
static sycl::event normalize_descriptors_sycl(
    sycl::queue& q,
    Descriptor* d_descs,
    const int num_orientations,
    const int norm_multi,
    NormalizeRootSift*)  // Tag dispatch
{
    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(num_orientations), [=](sycl::id<1> idx) {
            const int i = idx[0];
            
            Descriptor* desc = &d_descs[i];
            float* features = desc->features;
            
            // L1 normalization (sum of values)
            float sum = 0.0f;
            for(int j = 0; j < 128; j++) {
                sum += features[j];
            }
            
            // Avoid division by zero
            if(sum == 0.0f) sum = 1.0f;
            
            // Normalize, take square root, and scale
            float scale = sycl::ldexp(1.0f, norm_multi);
            
            for(int j = 0; j < 128; j++) {
                features[j] = sycl::sqrt(features[j] / sum) * scale;
            }
        });
    });
}

