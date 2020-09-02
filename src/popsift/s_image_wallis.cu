/*
 * Copyright 2020, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "s_image.h"
// #include <iostream>
// #include <fstream>
// #include "common/debug_macros.h"
// #include "common/assist.h"
// #include <stdio.h>
// #include <assert.h>

// #ifdef USE_NVTX
// #include <nvToolsExtCuda.h>
// #else
// #define nvtxRangePushA(a)
// #define nvtxRangePop()
// #endif

using namespace std;

namespace popsift {

/*************************************************************
 * ImageBase::wallisFilter
 *************************************************************/

    // Taken from here: https://se.mathworks.com/matlabcentral/answers/287847-what-is-wallis-filter-i-have-an-essay-on-it-and-i-cannot-understand-of-find-info-on-it

    // function WallisFilter(obj, Md, Dd, Amax, p, W)
    // Md and Dd are mean and contrast to match,
    // Amax and p constrain the change in individual pixels,
void ImageBase::wallisFilter( int filterWidth )
{
    const NppiSize COMPLETE = { .height = _h, .width = _w };
    const NppiPoint NOOFFSET = { .w = 0, .h = 0 };

    if( filterWidth %2 == 0 ) filterWidth++;
    const NppiSize FILTERSIZE = { .height = filterWidth, .width = filterWidht };

    int w = filterWidth >> 1; // floor(W/2)


    Plane<float> M( _w, _h );
    nppiFilterBox_32f_C1R( _input_image_d.getBuffer(), // src ptr
                           _input_image_d.getPitch(),  // src step
                           M.getBuffer(),              // dst ptr
                           M.getPitch(),               // dst step
                           COMPLETE,                   // region
                           FILTERSIZE,                 // filtersize
                           NOOFFSET );                 // shift
    // Plane<float> ipsum( _w, _h );
    // compute the inclusive prefix sum on all horizontals
    // after that compute the inclusive prefix sum on all verticals
    // that creates the basis for a box filter
    // ipsum = initBoxFilter( _input_image_d );
    // compute box filter ( pix(x+filterWidth/2,y+filterWidth/2) - pix(x-filterWidth/2,y-filterWidth/2) ) / filerWidth^2
    // M = runBoxFilter( ipsum, w );

    Plane<float> FminusM( _w, _h );
    // FminusM = _input_image_d - M; // element-wise substract 
    nppiSub_32s_C1R( _input_image_d.getBuffer(),
                     _input_image_d.getPitch(),
                     M.getBuffer(),
                     M.getPitch(),
                     FminusM.getBuffer(),
                     FminusM.getPitch(),
                     COMPLETE );

    Plane<float> Dprep( _w, _h );
    Plane<float> D( _w, _h );
    // compute element-wise: ( _input_image_d[pos] - M[pos] )^2
    // D = FminusM;
    // D.square(); // element-wise square
    nppiSqr_32f_C1R( FminusM.getBuffer(),
                     FminusM.getPitch(),
                     Dprep.getBuffer(),
                     Dprep.getPitch(),
                     COMPLETE );

    // ipsum = initBoxFilter( D );
    // D = runBoxFilter( ipsum, w );
    // D.divide( filterWidth^2 );
    nppiFilterBox_32f_C1R( Dprep.getBuffer(),
                           Dprep.getPitch(),
                           D.getBuffer(),
                           D.getPitch(),
                           COMPLETE,
                           FILTERSIZE,
                           NOOFFSET );

    // D.sqrt();
    nppiSqrt_32f_C1IR( D.getBuffer(),
                       D.getPitch(),
                       COMPLETE );
    // D.multiply( Amax );
    nppiMulC_32f_C1IR( Amax,
                       D.getBuffer(),
                       D.getPitch(),
                       COMPLETE );
    // D.add( Dd );
    nppiAddC_32f_C1IR( Dd,
                       D.getBuffer(),
                       D.getPitch(),
                       COMPLETE );

    Plane<float> G( _w, _h );
    // G = FminusM;
    // G.multiply( Amax * Dd );
    nppiMulC_32f_C1R( D.getBuffer(),
                      D.getPitch(),
                      Amax * Dd,
                      G.getBuffer(),
                      G.getPitch(),
                      COMPLETE );

    // D = G / D; // element-wise division
    nppiDiv_32f_C1IR( G.getBuffer(),
                      G.getPitch(),
                      D.getBuffer(),
                      D.getPitch(),
                      COMPLETE );

    // D.add( p * Md );
    nppiAddC_32f_C1IR( p * Md,
                       D.getBuffer(),
                       D.getPitch(),
                       COMPLETE );
    // M.multiply( 1-p );
    nppiMulC_32f_C1IR( 1.0f-p,
                       M.getBuffer(),
                       M.getPitch(),
                       COMPLETE );
    // D = D + M; // element-wise addition
    nppiAdd_32f_C1IR( M.getBuffer(),
                      M.getPitch(),
                      D.getBuffer(),
                      D.getPitch(),
                      COMPLETE );
    // D.max(0);
    nppiThreshold_LTVal_32f_C1IR( D.getBuffer(),
                                  D.getPitch(),
                                  COMPLETE,
                                  0.0f,   // if less-than this
                                  0.0f ); // set to this value
    // D.min(65534)
    nppiThreshold_GTVal_32f_C1IR( D.getBuffer(),
                                  D.getPitch(),
                                  COMPLETE,
                                  65534.0f,   // if greater-than this
                                  65534.0f ); // set to this value
}

} // namespace popsift

