/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/debug_macros.h"
#include "s_desc_grid.h"
#include "s_desc_igrid.h"
#include "s_desc_iloop.h"
#include "s_desc_loop.h"
#include "s_desc_normalize.h"
#include "s_desc_notile.h"
#include "s_desc_vlfeat.h"
#include "s_gradiant.h"
#include "sift_config.h"
#include "sift_constants.h"
#include "sift_pyramid.h"

#include <cstdio>
#include <iostream>

#if POPSIFT_IS_DEFINED(POPSIFT_USE_NVTX)
#include <nvToolsExtCuda.h>
#else
#define nvtxRangePushA(a)
#define nvtxRangePop()
#endif

using namespace popsift;
using namespace std;

        // start_ext_desc_notile<NormalizeRootSift>( octave, layer_tex );
        // start_ext_desc_notile<NormalizeL2>( octave, layer_tex );

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
__host__
void Pyramid::descriptors( const Config& conf )
{
   nvtxRangePushA("Reading orientation count");

   readDescCountersFromDevice( _octaves[0].getStream() );
   cudaStreamSynchronize( _octaves[0].getStream() );
   nvtxRangePop( );

    for( int octave=_num_octaves-1; octave>=0; octave-- )
    // for( int octave=0; octave<_num_octaves; octave++ )
    {
        if( hct.ori_ct[octave] != 0 ) {
            Octave& oct_obj = _octaves[octave];

            if( conf.getDescMode() == Config::Loop ) {
                start_ext_desc_loop(  octave, oct_obj );
            } else if( conf.getDescMode() == Config::ILoop ) {
                start_ext_desc_iloop( octave, oct_obj );
            } else if( conf.getDescMode() == Config::Grid ) {
                start_ext_desc_grid(  octave, oct_obj );
            } else if( conf.getDescMode() == Config::IGrid ) {
                start_ext_desc_igrid( octave, oct_obj );
            } else if( conf.getDescMode() == Config::NoTile ) {
                start_ext_desc_notile( octave, oct_obj );
            } else if( conf.getDescMode() == Config::VLFeat_Desc ) {
                start_ext_desc_vlfeat( octave, oct_obj );
            } else {
                POP_FATAL( "not yet" );
            }
            cuda::event_record( oct_obj.getEventDescDone(), oct_obj.getStream(), __FILE__, __LINE__ );
            cuda::event_wait(   oct_obj.getEventDescDone(), _download_stream,    __FILE__, __LINE__ );
        }
    }

    if( hct.ori_total == 0 )
    {
        cerr << "Warning: no descriptors extracted" << endl;
        return;
    }

    dim3 block;
    block.x = 32;
    block.y = 32;
    block.z = 1;

    dim3 grid;
    grid.x  = grid_divide( hct.ori_total, block.y );

    if( conf.getUseRootSift() ) {
        normalize_histogram<NormalizeRootSift> <<<grid,block,0,_download_stream>>> ( );
        POP_SYNC_CHK;
    } else {
        normalize_histogram<NormalizeL2> <<<grid,block,0,_download_stream>>> ( );
        POP_SYNC_CHK;
    }

    cudaDeviceSynchronize( );
}

