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

#include <cmath>

using namespace popsift;
using namespace std;

class NormalizeL2
{
public:
    static inline
    void normalize( float* features );

    static inline
    void normalize( const float* src_desc,
                    float*       dest_desc );
};

inline
void NormalizeL2::normalize( float* features )
{
    normalize( features, features );
}

inline
void NormalizeL2::normalize( const float* src_desc, float* dst_desc )
{
    float norm = 0;

    /* Sum of squares of the descriptor values. */
    for( int i=0; i<128; i++ )
    {
        norm += ( src_desc[i] * src_desc[i] );
    }

    // compute 1 / sqrt(sum)
    // norm = frsqrtf( norm );
    norm = 1.0f / std::sqrt( norm );

    float desc[128];

    // quasi-normalize all 128 floats
    for( int i=0; i<128; i++ )
    {
        desc[i] = min( src_desc[i] * norm, 0.2f );
    }

    norm = 0;

    /* Sum of squares of the descriptor values. */
    for( int i=0; i<128; i++ )
    {
        norm += ( desc[i] * desc[i] );
    }

    // norm = frsqrtf( norm ); // inverse square root
    norm = 1.0f / std::sqrt( norm );

    // scale for the desired output scale (0-1, 0-256 og 0-512)
    norm = scalbnf( norm, h_consts.norm_multi );

    /* Write back normalized and scaled descriptors */
    for( int i=0; i<128; i++ )
    {
        dst_desc[i] = desc[i] * norm;
    }
}

