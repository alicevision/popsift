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
        float val;
        dst_desc[i] = scalbnf( sqrtf( src_desc[i] / sum ),
                               h_consts.norm_multi );
    }
}

