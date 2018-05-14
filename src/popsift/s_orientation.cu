/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <math.h>
#include <stdio.h>
#include <inttypes.h>

#include "common/assist.h"
#include "sift_pyramid.h"
#include "sift_constants.h"
#include "s_gradiant.h"
#include "common/excl_blk_prefix_sum.h"
#include "common/warp_bitonic_sort.h"
#include "common/debug_macros.h"
#include "common/assist.h"

#ifdef USE_NVTX
#include <nvToolsExtCuda.h>
#else
#define nvtxRangePushA(a)
#define nvtxRangePop()
#endif

using namespace popsift;
using namespace std;

/* Smoothing like VLFeat is the default mode.
 * If you choose to undefine it, you get the smoothing approach taken by OpenCV
 */
#define WITH_VLFEAT_SMOOTHING

namespace popsift
{

__device__
inline float compute_angle( int bin, float hc, float hn, float hp )
{
    /* interpolate */
    float di = bin + 0.5f * (hn - hp) / (hc+hc-hn-hp);

    /* clamp */
    di = (di < 0) ? 
            (di + ORI_NBINS) : 
            ((di >= ORI_NBINS) ? (di - ORI_NBINS) : (di));

    float th = __fdividef( M_PI2 * di, ORI_NBINS ) - M_PI;
    // float th = ((M_PI2 * di) / ORI_NBINS);
    return th;
}

/*
 * Compute the keypoint orientations for each extremum
 * using 16 threads for each of them.
 * direct curve fitting approach
 */
__global__
void ori_par( const int           octave,
              const int           ext_ct_prefix_sum,
              cudaTextureObject_t layer,
              const int           w,
              const int           h )
{
    const int extremum_index  = blockIdx.x * blockDim.y;

    if( extremum_index >= dct.ext_ct[octave] ) return; // a few trailing warps

    const int              iext_off =  dobuf.i_ext_off[octave][extremum_index];
    const InitialExtremum* iext     = &dobuf.i_ext_dat[octave][iext_off];

    __shared__ float hist   [ORI_NBINS];
    __shared__ float sm_hist[ORI_NBINS];

    for( int i = threadIdx.x; i < ORI_NBINS; i += blockDim.x )  hist[i] = 0.0f;

    /* keypoint fractional geometry */
    const float x     = iext->xpos;
    const float y     = iext->ypos;
    const int   level = iext->lpos; // old_level;
    const float sig   = iext->sigma;

    /* orientation histogram radius */
    float  sigw = ORI_WINFACTOR * sig;
    int32_t rad  = (int)roundf((3.0f * sigw));

    float factor = __fdividef( -0.5f, (sigw * sigw) );
    int sq_thres  = rad * rad;

    // int xmin = max(1,     (int)floor(x - rad));
    // int xmax = min(w - 2, (int)floor(x + rad));
    // int ymin = max(1,     (int)floor(y - rad));
    // int ymax = min(h - 2, (int)floor(y + rad));
    int xmin = max(1,     (int)roundf(x) - rad);
    int xmax = min(w - 2, (int)roundf(x) + rad);
    int ymin = max(1,     (int)roundf(y) - rad);
    int ymax = min(h - 2, (int)roundf(y) + rad);

    int wx = xmax - xmin + 1;
    int hy = ymax - ymin + 1;
    int loops = wx * hy;

    for( int i = threadIdx.x; popsift::any(i < loops); i += blockDim.x )
    {
        if( i < loops ) {
            int yy = i / wx + ymin;
            int xx = i % wx + xmin;

            float grad;
            float theta;
            get_gradiant( grad,
                          theta,
                          xx,
                          yy,
                          layer,
                          level );

            float dx = xx - x;
            float dy = yy - y;

            int sq_dist  = dx * dx + dy * dy;
            if (sq_dist <= sq_thres) {
                float weight = grad * expf(sq_dist * factor);

                // int bidx = (int)rintf( __fdividef( ORI_NBINS * (theta + M_PI), M_PI2 ) );
                int bidx = (int)roundf( __fdividef( float(ORI_NBINS) * (theta + M_PI), M_PI2 ) );

                if( bidx > ORI_NBINS ) {
                    printf("Crashing: bin %d theta %f :-)\n", bidx, theta);
                }

                bidx = (bidx == ORI_NBINS) ? 0 : bidx;

                atomicAdd( &hist[bidx], weight );
            }
        }
        __syncthreads();
    }

#ifdef WITH_VLFEAT_SMOOTHING
    for( int i=0; i<3; i++ ) {
        for( int bin = threadIdx.x; bin < ORI_NBINS; bin += blockDim.x ) {
            int prev = bin == 0 ? ORI_NBINS-1 : bin-1;
            int next = bin == ORI_NBINS-1 ? 0 : bin+1;
            sm_hist[bin] = ( hist[prev] + hist[bin] + hist[next] ) / 3.0f;
        }
        __syncthreads();
        for( int bin = threadIdx.x; bin < ORI_NBINS; bin += blockDim.x ) {
            int prev = bin == 0 ? ORI_NBINS-1 : bin-1;
            int next = bin == ORI_NBINS-1 ? 0 : bin+1;
            hist[bin] = ( sm_hist[prev] + sm_hist[bin] + sm_hist[next] ) / 3.0f;
        }
        __syncthreads();
    }
    for( int bin = threadIdx.x; bin < ORI_NBINS; bin += blockDim.x ) {
        sm_hist[bin] = hist[bin];
    }
    __syncthreads();
#else // not WITH_VLFEAT_SMOOTHING
    for( int bin = threadIdx.x; bin < ORI_NBINS; bin += blockDim.x ) {
        int prev2 = bin - 2;
        int prev1 = bin - 1;
        int next1 = bin + 1;
        int next2 = bin + 2;
        if( prev2 < 0 )          prev2 += ORI_NBINS;
        if( prev1 < 0 )          prev1 += ORI_NBINS;
        if( next1 >= ORI_NBINS ) next1 -= ORI_NBINS;
        if( next2 >= ORI_NBINS ) next2 -= ORI_NBINS;
        sm_hist[bin] = (   hist[prev2] + hist[next2]
                         + ( hist[prev1] + hist[next1] ) * 4.0f
                         +   hist[bin] * 6.0f ) / 16.0f;
    }
    __syncthreads();
#endif // not WITH_VLFEAT_SMOOTHING

    // sub-cell refinement of the histogram cell index, yielding the angle
    // not necessary to initialize, every cell is computed
    __shared__ float refined_angle[64];
    __shared__ float yval         [64];

    for( int bin = threadIdx.x; popsift::any( bin < ORI_NBINS ); bin += blockDim.x ) {
        const int prev = bin == 0 ? ORI_NBINS-1 : bin-1;
        const int next = bin == ORI_NBINS-1 ? 0 : bin+1;

        bool predicate = ( bin < ORI_NBINS ) && ( sm_hist[bin] > max( sm_hist[prev], sm_hist[next] ) );

        const float num  = predicate ?   3.0f * sm_hist[prev]
                                       - 4.0f * sm_hist[bin]
                                       + 1.0f * sm_hist[next]
                                     : 0.0f;
        // const float num  = predicate ?   2.0f * sm_hist[prev]
        //                                - 4.0f * sm_hist[bin]
        //                                + 2.0f * sm_hist[next]
        //                              : 0.0f;
        const float denB = predicate ? 2.0f * ( sm_hist[prev] - 2.0f * sm_hist[bin] + sm_hist[next] ) : 1.0f;

        const float newbin = __fdividef( num, denB ); // verified: accuracy OK

        predicate   = ( predicate && newbin >= 0.0f && newbin <= 2.0f );

        refined_angle[bin] = predicate ? prev + newbin : -1;
        yval[bin]          = predicate ?  -(num*num) / (4.0f * denB) + sm_hist[prev] : -INFINITY;
    }

    int2 best_index = make_int2( threadIdx.x, threadIdx.x + 32 );

    BitonicSort::Warp32<float> sorter( yval );
    sorter.sort64( best_index );
    __syncthreads();

    // All threads retrieve the yval of thread 0, the largest
    // of all yvals.
    const float best_val = yval[best_index.x];
    const float yval_ref = 0.8f * popsift::shuffle( best_val, 0 );
    const bool  valid    = ( best_val >= yval_ref );
    bool        written  = false;

    Extremum* ext = &dobuf.extrema[ext_ct_prefix_sum + extremum_index];

    if( threadIdx.x < ORIENTATION_MAX_COUNT ) {
        if( valid ) {
            float chosen_bin = refined_angle[best_index.x];
            if( chosen_bin >= ORI_NBINS ) chosen_bin -= ORI_NBINS;
            // float th = __fdividef(M_PI2 * chosen_bin , ORI_NBINS) - M_PI;
            float th = ::fmaf( M_PI2 * chosen_bin, 1.0f/ORI_NBINS, - M_PI );
            ext->orientation[threadIdx.x] = th;
            written = true;
        }
    }

    int angles = __popc( popsift::ballot( written ) );
    if( threadIdx.x == 0 ) {
        ext->xpos    = iext->xpos;
        ext->ypos    = iext->ypos;
        ext->lpos    = iext->lpos;
        ext->sigma   = iext->sigma;
        ext->octave  = octave;
        ext->num_ori = angles;
    }
}

}; // namespace popsift

class ExtremaRead
{
    const Extremum* const _oris;
public:
    inline __device__
    ExtremaRead( const Extremum* const d_oris ) : _oris( d_oris ) { }

    inline __device__
    int get( int n ) const { return _oris[n].num_ori; }
};

class ExtremaWrt
{
    Extremum* _oris;
public:
    inline __device__
    ExtremaWrt( Extremum* d_oris ) : _oris( d_oris ) { }

    inline __device__
    void set( int n, int value ) { _oris[n].idx_ori = value; }
};

class ExtremaTot
{
    int& _extrema_counter;
public:
    inline __device__
    ExtremaTot( int& extrema_counter ) : _extrema_counter( extrema_counter ) { }

    inline __device__
    void set( int value ) { _extrema_counter = value; }
};

class ExtremaWrtMap
{
    int* _featvec_to_extrema_mapper;
    int  _max_feat;
public:
    inline __device__
    ExtremaWrtMap( int* featvec_to_extrema_mapper, int max_feat )
        : _featvec_to_extrema_mapper( featvec_to_extrema_mapper )
        , _max_feat( max_feat )
    { }

    inline __device__
    void set( int base, int num, int value )
    {
        int* baseptr = &_featvec_to_extrema_mapper[base];
        do {
            num--;
            if( base + num < _max_feat ) {
                baseptr[num] = value;
            }
        } while( num > 0 );
    }
};

__global__
void ori_prefix_sum( const int total_ext_ct, const int num_octaves )
{
    int       total_ori       = 0;
    Extremum* extremum        = dobuf.extrema;
    int*      feat_to_ext_map = dobuf.feat_to_ext_map;

    ExtremaRead r( extremum );
    ExtremaWrt  w( extremum );
    ExtremaTot  t( total_ori );
    ExtremaWrtMap wrtm( feat_to_ext_map, max( d_consts.max_orientations, dbuf.ori_allocated ) );
    ExclusivePrefixSum::Block<ExtremaRead,ExtremaWrt,ExtremaTot,ExtremaWrtMap>( total_ext_ct, r, w, t, wrtm );

    __syncthreads();

    if( threadIdx.x == 0 && threadIdx.y == 0 ) {
        dct.ext_ps[0] = 0;
        for( int o=1; o<MAX_OCTAVES; o++ ) {
            dct.ext_ps[o] = dct.ext_ps[o-1] + dct.ext_ct[o-1];
        }

        for( int o=0; o<MAX_OCTAVES; o++ ) {
            if( dct.ext_ct[o] == 0 ) {
                dct.ori_ct[o] = 0;
            } else {
                int fe = dct.ext_ps[o  ];   /* first extremum for this octave */
                int le = dct.ext_ps[o+1]-1; /* last  extremum for this octave */
                int lo_ori_index = dobuf.extrema[fe].idx_ori;
                int num_ori      = dobuf.extrema[le].num_ori;
                int hi_ori_index = dobuf.extrema[le].idx_ori + num_ori;
                dct.ori_ct[o] = hi_ori_index - lo_ori_index;
            }
        }

        dct.ori_ps[0] = 0;
        for( int o=1; o<MAX_OCTAVES; o++ ) {
            dct.ori_ps[o] = dct.ori_ps[o-1] + dct.ori_ct[o-1];
        }

        dct.ori_total = dct.ori_ps[MAX_OCTAVES-1] + dct.ori_ct[MAX_OCTAVES-1];
        dct.ext_total = dct.ext_ps[MAX_OCTAVES-1] + dct.ext_ct[MAX_OCTAVES-1];
    }
}

__host__
void Pyramid::orientation( const Config& conf )
{
    nvtxRangePushA( "reading extrema count" );
    readDescCountersFromDevice( );
    nvtxRangePop( );

    nvtxRangePushA( "filtering grid" );
    int ext_total = 0;
    for( int o=0; o<MAX_OCTAVES; o++ ) {
        if( hct.ext_ct[o] > 0 ) {
            ext_total += hct.ext_ct[o];
        }
    }

    // Filter functions are only called if necessary. They are very expensive,
    // therefore add 10% slack.
    if( conf.getFilterMaxExtrema() > 0 && int(conf.getFilterMaxExtrema()*1.1) < ext_total )
    {
        ext_total = extrema_filter_grid( conf, ext_total );
    }
    nvtxRangePop( );

    nvtxRangePushA( "reallocating extrema arrays" );
    reallocExtrema( ext_total );
    nvtxRangePop( );

    int ext_ct_prefix_sum = 0;
    for( int octave=0; octave<_num_octaves; octave++ ) {
        hct.ext_ps[octave] = ext_ct_prefix_sum;
        ext_ct_prefix_sum += hct.ext_ct[octave];
    }
    hct.ext_total = ext_ct_prefix_sum;

    cudaStream_t oct_0_str = _octaves[0].getStream();

    // for( int octave=0; octave<_num_octaves; octave++ )
    for( int octave=_num_octaves-1; octave>=0; octave-- )
    {
        Octave&      oct_obj = _octaves[octave];

        cudaStream_t oct_str = oct_obj.getStream();

        int num = hct.ext_ct[octave];

        if( num > 0 ) {
            dim3 block;
            dim3 grid;

            block.x = 32;
            block.y = 1;
            grid.x  = num;

            ori_par
                <<<grid,block,0,oct_str>>>
                ( octave,
                  hct.ext_ps[octave],
                  oct_obj.getDataTexPoint( ),
                  oct_obj.getWidth( ),
                  oct_obj.getHeight( ) );
            POP_SYNC_CHK;

            if( octave != 0 ) {
                cuda::event_record( oct_obj.getEventOriDone(), oct_str,   __FILE__, __LINE__ );
                cuda::event_wait  ( oct_obj.getEventOriDone(), oct_0_str, __FILE__, __LINE__ );
            }
        }
    }

    /* Compute and set the orientation prefixes on the device */
    dim3 block;
    dim3 grid;
    block.x = 32;
    block.y = 32;
    grid.x  = 1;
    ori_prefix_sum
        <<<grid,block,0,oct_0_str>>>
        ( ext_ct_prefix_sum, _num_octaves );
    POP_SYNC_CHK;

    cudaDeviceSynchronize();
}

