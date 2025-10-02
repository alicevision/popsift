/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/debug_macros.h"
#include "common/grid.h"
#include "s_desc_vlfeat.h"
#include "s_desc_normalize.h"
#include "s_gradiant.h"
#include "sift_config.h"
#include "sift_constants.h"
#include "sift_pyramid.h"

#include <cstdio>
#include <iostream>

using namespace popsift;
using namespace std;

/*************************************************************
 * descriptor extraction
 * TODO: We use the level of the octave in which the keypoint
 *       was found to extract the descriptor. This is
 *       not 100% as intended by Lowe. The paper says:
 *       "magnitudes and gradient are sampled around the
 *        keypoint location, using the scale of the keypoint
 *        to select the level of Gaussian blur for the image."
 *       This implies that a keypoint that has changed octave
 *       in subpixelic refinement is going to be sampled from
 *       the wrong level of the octave.
 *       Unfortunately, we cannot implement getDataTexPoint()
 *       as a layered 2D texture to fix this issue, because that
 *       would require to store blur levels in cudaArrays, which
 *       are hard to write. Alternatively, we could keep a
 *       device-side octave structure that contains an array of
 *       levels on the device side.
 *************************************************************/
void Pyramid::descriptors( const Config& conf )
{
    for( int octave=_num_octaves-1; octave>=0; octave-- )
    {
        if( dct.ori_ct[octave] != 0 ) {
            Octave& oct_obj = _octaves[octave];

            start_ext_desc_vlfeat( octave, oct_obj );
        }
    }

    if( dct.ori_total == 0 )
    {
        cerr << "Warning: no descriptors extracted" << endl;
        return;
    }

    if( conf.getUseRootSift() ) {
        normalize_histogram<NormalizeRootSift>( );
    } else {
        normalize_histogram<NormalizeL2>( );
    }
}

