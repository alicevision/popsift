/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/debug_macros.h"
#include "common/excl_blk_prefix_sum.h"
#include "common/warp_bitonic_sort.h"
#include "s_gradiant.h"
#include "sift_config.h"
#include "sift_constants.h"
#include "sift_pyramid.h"

#include <cinttypes>
#include <cmath>
#include <cstdio>

#if POPSIFT_IS_DEFINED(POPSIFT_USE_NVTX)
#include <nvToolsExtCuda.h>
#else
#define nvtxRangePushA(a)
#define nvtxRangePop()
#endif

using namespace popsift;
using namespace std;

namespace popsift
{

/*
 * Histogram smoothing helper
 */
__device__ inline static
float smoothe( const float* const src, const int bin )
{
    const int prev = (bin == 0) ? ORI_NBINS-1 : bin-1;
    const int next = (bin == ORI_NBINS-1) ? 0 : bin+1;

    const float f  = ( src[prev] + src[bin] + src[next] ) / 3.0f;

    return f;
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

    if( popsift::all( extremum_index >= dct.ext_ct[octave] ) ) return; // a few trailing warps

    const int              iext_off =  dobuf.i_ext_off[octave][extremum_index];
    const InitialExtremum* iext     = &dobuf.i_ext_dat[octave][iext_off];

    __shared__ float hist         [64];
    __shared__ float sm_hist      [64];
    __shared__ float refined_angle[64];
    __shared__ float yval         [64];

    hist[threadIdx.x+ 0] = 0.0f;
    hist[threadIdx.x+32] = 0.0f;

    /* keypoint fractional geometry */
    const float x     = iext->xpos;
    const float y     = iext->ypos;
    const int   level = iext->lpos; // old_level;
    const float sig   = iext->sigma;

    /* orientation histogram radius */
    const float  sigw = ORI_WINFACTOR * sig;
    const int32_t rad  = max( (int)floorf((3.0f * sigw)), 1 );

    const float factor = __fdividef( -0.5f, (sigw * sigw) );
    const int sq_thres  = rad * rad;

    // int xmin = max(1,     (int)floor(x - rad));
    // int xmax = min(w - 2, (int)floor(x + rad));
    // int ymin = max(1,     (int)floor(y - rad));
    // int ymax = min(h - 2, (int)floor(y + rad));
    int xmin = max(0,     (int)roundf(x) - rad);
    int xmax = min(w - 1, (int)roundf(x) + rad);
    int ymin = max(0,     (int)roundf(y) - rad);
    int ymax = min(h - 1, (int)roundf(y) + rad);

    int wx = xmax - xmin + 1;
    int hy = ymax - ymin + 1;
    int loops = wx * hy;

    __syncthreads();
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

            grad /= 2; // our grad is twice that of VLFeat - weird
            if( theta < 0 ) theta += M_PI2;

            float dx = xx - x;
            float dy = yy - y;

            float sq_dist  = dx * dx + dy * dy;
            if (sq_dist <= sq_thres + 0.6f)
            {
                float weight = grad * expf(sq_dist * factor);

                int bidx = (int)roundf( __fdividef( float(ORI_NBINS) * (theta + M_PI), M_PI2 ) );

                while( bidx < 0 )          bidx += ORI_NBINS;
                while( bidx >= ORI_NBINS ) bidx -= ORI_NBINS;

                atomicAdd( &hist[bidx], weight );
            }
        }
    }
    __syncthreads();

    for( int i=0; i<3 ; i++ )
    {
        sm_hist[threadIdx.x+ 0] = smoothe( hist, threadIdx.x+ 0 );
        sm_hist[threadIdx.x+32] = smoothe( hist, threadIdx.x+32 );
        __syncthreads();
        hist[threadIdx.x+ 0]    = smoothe( sm_hist, threadIdx.x+ 0 );
        hist[threadIdx.x+32]    = smoothe( sm_hist, threadIdx.x+32 );
        __syncthreads();
    }

    sm_hist[threadIdx.x+ 0] = hist[threadIdx.x+ 0];
    sm_hist[threadIdx.x+32] = hist[threadIdx.x+32];
    __syncthreads();

    if( threadIdx.x+32 >= ORI_NBINS ) sm_hist[threadIdx.x+32] = -INFINITY;
    float maxval = fmaxf( sm_hist[threadIdx.x+ 0], sm_hist[threadIdx.x+32] );
    float neigh;
    neigh  = popsift::shuffle_down( maxval, 16 ); maxval = fmaxf( maxval, neigh );
    neigh  = popsift::shuffle_down( maxval,  8 ); maxval = fmaxf( maxval, neigh );
    neigh  = popsift::shuffle_down( maxval,  4 ); maxval = fmaxf( maxval, neigh );
    neigh  = popsift::shuffle_down( maxval,  2 ); maxval = fmaxf( maxval, neigh );
    neigh  = popsift::shuffle_down( maxval,  1 ); maxval = fmaxf( maxval, neigh );
    maxval = popsift::shuffle( maxval,  0 );

    // sub-cell refinement of the histogram cell index, yielding the angle
    // not necessary to initialize, every cell is computed

    for( int bin = threadIdx.x; popsift::any( bin < ORI_NBINS ); bin += blockDim.x )
    {
        const int prev = ( bin - 1 + ORI_NBINS ) % ORI_NBINS;
        const int next = ( bin + 1 )             % ORI_NBINS;

        bool predicate = ( bin < ORI_NBINS ) &&
                         ( sm_hist[bin] > max( sm_hist[prev], sm_hist[next] ) ) &&
                         ( sm_hist[bin] > 0.8f * maxval );

        const float num  = predicate ?   3.0f * sm_hist[prev]
                                       - 4.0f * sm_hist[bin]
                                       + 1.0f * sm_hist[next]
                                     : 0.0f;
        const float denB = predicate ? 2.0f * ( sm_hist[prev] - 2.0f * sm_hist[bin] + sm_hist[next] ) : 1.0f;

        const float newbin = __fdividef( num, denB ); // verified: accuracy OK

        refined_angle[bin] = predicate ? prev + newbin : -1;
        yval[bin]          = predicate ?  -(num*num) / (4.0f * denB) + hist[prev] : -INFINITY;
    }
    __syncthreads();

    int2 best_index = make_int2( threadIdx.x, threadIdx.x + 32 );

    BitonicSort::Warp32<float> sorter( yval );
    sorter.sort64( best_index );
    __syncthreads();

    // All threads retrieve the yval of thread 0, the largest
    // of all yvals.
    const bool  valid    = ( yval[best_index.x] > 0 );
    bool        written  = false;

    Extremum* ext = &dobuf.extrema[ext_ct_prefix_sum + extremum_index];

    if( threadIdx.x < ORIENTATION_MAX_COUNT ) {
        if( valid ) {
            float chosen_bin = refined_angle[best_index.x];
            if( chosen_bin >= ORI_NBINS ) chosen_bin -= ORI_NBINS;
            if( chosen_bin <  0         ) chosen_bin += ORI_NBINS;
            float th = __fdividef(M_PI2 * chosen_bin , ORI_NBINS); // - M_PI;
            th -= M_PI;
            if( th < 0.0f ) th += M_PI2;
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
    explicit ExtremaRead( const Extremum* const d_oris ) : _oris( d_oris ) { }

    inline __device__
    int get( int n ) const { return _oris[n].num_ori; }
};

class ExtremaWrt
{
    Extremum* _oris;
public:
    inline __device__
    explicit ExtremaWrt( Extremum* d_oris ) : _oris( d_oris ) { }

    inline __device__
    void set( int n, int value ) { _oris[n].idx_ori = value; }
};

class ExtremaTot
{
    int& _extrema_counter;
public:
    inline __device__
    explicit ExtremaTot( int& extrema_counter ) : _extrema_counter( extrema_counter ) { }

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
    readDescCountersFromDevice( );

    int ext_total = 0;
    for(int o : hct.ext_ct)
    {
        if( o > 0 )
        {
            ext_total += o;
        }
    }

    // Filter functions are only called if necessary. They are very expensive,
    // therefore add 10% slack.
    if( conf.getFilterMaxExtrema() > 0 && int(conf.getFilterMaxExtrema()*1.1) < ext_total )
    {
        ext_total = extrema_filter_grid( conf, ext_total );
    }

    reallocExtrema( ext_total );

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
                <<<grid,block,4*64*sizeof(float),oct_str>>>
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

