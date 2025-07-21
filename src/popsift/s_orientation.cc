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
#include <numeric>

using namespace popsift;
using namespace std;

/* Smoothing like VLFeat is the default mode.
 * If you choose to undefine it, you get the smoothing approach taken by OpenCV
 */
#define WITH_VLFEAT_SMOOTHING

namespace popsift
{

inline float compute_angle( int bin, float hc, float hn, float hp )
{
    /* interpolate */
    float di = bin + 0.5f * (hn - hp) / (hc+hc-hn-hp);

    /* clamp */
    di = (di < 0) ? 
            (di + ORI_NBINS) : 
            ((di >= ORI_NBINS) ? (di - ORI_NBINS) : (di));

    // float th = __fdividef( M_PI2 * di, ORI_NBINS ) - M_PI;
    float th = ((M_PI2 * di) / ORI_NBINS) - M_PI;
    return th;
}

/*
 * Histogram smoothing helper
 */
inline static
float smoothe( const float* const src, const int bin )
{
    const int prev = (bin == 0) ? ORI_NBINS-1 : bin-1;
    const int next = (bin == ORI_NBINS-1) ? 0 : bin+1;

    const float f  = ( src[prev] + src[bin] + src[next] ) / 3.0f;

    return f;
}

/*
 * Compute the keypoint orientations for each extremum.
 * Direct curve fitting approach.
 */
void ori_par( Grid&                g,
              const int            octave,
              const int            ext_ct_prefix_sum,
              const Plane2D_float& layer,
              const int            w,
              const int            h )
{
  g.reset();
  do {
    const int extremum_index  = g.blockIdx.x;

    if( extremum_index >= dct.ext_ct[octave] ) continue; // a few trailing warps

    const int              iext_off =  dobuf.i_ext_off[octave][extremum_index];
    const InitialExtremum* iext     = &dobuf.i_ext_dat[octave][iext_off];

    float hist[64] = { 0 };

    /* keypoint fractional geometry */
    const float x     = iext->xpos;
    const float y     = iext->ypos;
    const int   level = iext->lpos; // old_level;
    const float sig   = iext->sigma;

    /* orientation histogram radius */
    const float  sigw = ORI_WINFACTOR * sig;
    const int32_t rad  = (int)roundf((3.0f * sigw));

    // const float factor = __fdividef( -0.5f, (sigw * sigw) );
    const float factor = -0.5f / (sigw * sigw);
    const int sq_thres  = rad * rad;

    // int xmin = max(1,     (int)floor(x - rad));
    // int xmax = min(w - 2, (int)floor(x + rad));
    // int ymin = max(1,     (int)floor(y - rad));
    // int ymax = min(h - 2, (int)floor(y + rad));
    int xmin = std::max(1,     (int)roundf(x) - rad);
    int xmax = std::min(w - 2, (int)roundf(x) + rad);
    int ymin = std::max(1,     (int)roundf(y) - rad);
    int ymax = std::min(h - 2, (int)roundf(y) + rad);

    int wx = xmax - xmin + 1;
    int hy = ymax - ymin + 1;
    int loops = wx * hy;

    for( int i = 0; i < loops; i += g.blockDim.x )
    {
        if( i < loops ) {
            int yy = i / wx + ymin;
            int xx = i % wx + xmin;

            float grad;
            float theta;
            get_gradiant32( grad,
                            theta,
                            xx,
                            yy,
                            layer,
                            level );

            float dx = xx - x;
            float dy = yy - y;

            int sq_dist  = dx * dx + dy * dy;
            if (sq_dist <= sq_thres)
            {
                float weight = grad * expf(sq_dist * factor);

                // int bidx = (int)rintf( __fdividef( ORI_NBINS * (theta + M_PI), M_PI2 ) );
                // int bidx = (int)roundf( __fdividef( float(ORI_NBINS) * (theta + M_PI), M_PI2 ) );
                int bidx = (int)roundf( float(ORI_NBINS) * (theta + M_PI) / M_PI2 );

                if( bidx > ORI_NBINS ) {
                    printf("Crashing: bin %d theta %f :-)\n", bidx, theta);
                }
                if( bidx < 0 ) {
                    printf("Crashing: bin %d theta %f :-)\n", bidx, theta);
                }

                bidx = (bidx == ORI_NBINS) ? 0 : bidx;

                hist[bidx] += weight;
            }
        }
    }

    float sm_hist[64];

#ifdef WITH_VLFEAT_SMOOTHING
    for( int i=0; i<3 ; i++ ) {
        for( int j=0; j<64; j++ ) {
            sm_hist[j] = smoothe( hist, j );
        }
        for( int j=0; j<64; j++ ) {
            hist[j] = smoothe( sm_hist, j );
        }
    }

    for( int i=0; i<64; i++ ) {
        sm_hist[i] = hist[i];
    }
#else // not WITH_VLFEAT_SMOOTHING
    for( int bin = 0; bin < ORI_NBINS; bin += 32 )
    {
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
#endif // not WITH_VLFEAT_SMOOTHING

    float yval[64];
    float refined_angle[64];

    // sub-cell refinement of the histogram cell index, yielding the angle
    // not necessary to initialize, every cell is computed

    for( int bin = 0; bin < ORI_NBINS; bin ++ )
    {
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

        // const float newbin = __fdividef( num, denB ); // verified: accuracy OK
        const float newbin = num / denB;

        predicate   = ( predicate && newbin >= 0.0f && newbin <= 2.0f );

        refined_angle[bin] = predicate ? prev + newbin : -1;
        yval[bin]          = predicate ?  -(num*num) / (4.0f * denB) + sm_hist[prev] : -INFINITY;
    }


    int best_index[64];

    /* initialize array best_index with the indices of array yval */
    std::iota( best_index, best_index+64, 0 );

    /* sort array best_index contain yval indices in order of _decreasing_ yval values */
    std::sort( best_index, best_index+64, [&]( int l, int r ) {
                                              return ( yval[best_index[l]] > yval[best_index[r]] );
                                          } );

    Extremum* ext = &dobuf.extrema[ext_ct_prefix_sum + extremum_index];

    int angles = 0;

    // All threads retrieve the yval of thread 0, the largest
    // of all yvals.
    for( int i=0; i<ORIENTATION_MAX_COUNT; i++ )
    {
        const float best_val = yval[best_index[i]];
        const float yval_ref = 0.8f * yval[best_index[0]];
        const bool  valid    = ( best_val >= yval_ref );

        if( valid )
        {
            float chosen_bin = refined_angle[best_index[i]];
            if( chosen_bin >= ORI_NBINS ) chosen_bin -= ORI_NBINS;
            // float th = __fdividef(M_PI2 * chosen_bin , ORI_NBINS) - M_PI;
            float th = std::fmaf( M_PI2 * chosen_bin, 1.0f/ORI_NBINS, - M_PI );
            ext->orientation[i] = th;

            angles += 1;
        }
    }

    ext->xpos    = iext->xpos;
    ext->ypos    = iext->ypos;
    ext->lpos    = iext->lpos;
    ext->sigma   = iext->sigma;
    ext->octave  = octave;
    ext->num_ori = angles;
  } while( g.next() );
}

}; // namespace popsift

class ExtremaRead
{
    const Extremum* const _oris;
public:
    inline 
    explicit ExtremaRead( const Extremum* const d_oris ) : _oris( d_oris ) { }

    inline 
    int get( int n ) const { return _oris[n].num_ori; }
};

class ExtremaWrt
{
    Extremum* _oris;
public:
    inline 
    explicit ExtremaWrt( Extremum* d_oris ) : _oris( d_oris ) { }

    inline 
    void set( int n, int value ) { _oris[n].idx_ori = value; }
};

class ExtremaTot
{
    int& _extrema_counter;
public:
    inline 
    explicit ExtremaTot( int& extrema_counter ) : _extrema_counter( extrema_counter ) { }

    inline 
    void set( int value ) { _extrema_counter = value; }
};

class ExtremaWrtMap
{
    int* _featvec_to_extrema_mapper;
    int  _max_feat;
public:
    inline 
    ExtremaWrtMap( int* featvec_to_extrema_mapper, int max_feat )
        : _featvec_to_extrema_mapper( featvec_to_extrema_mapper )
        , _max_feat( max_feat )
    { }

    inline 
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

void ori_prefix_sum( const int total_ext_ct, const int num_octaves )
{
    if( total_ext_ct < 1 )
    {
        POP_FATAL("Calling " << __FUNCTION__ << " with " << total_ext_ct << " found extrema");
    }

    Extremum* extremum = dobuf.extrema;

    int* ori_count  = new int[total_ext_ct];
    int* ori_offset = new int[total_ext_ct+1];

    /* collect the numbers of orientation for every extremum in ori_count */
    for( int i=0; i<total_ext_ct; i++ )
    {
        ori_count[i] = extremum->num_ori;
    }

    /* set ori_offset[0] to 0,
     * then compute an inclusive prefix sum for the values in ori_count into
     * the target array ori_offset, but starting at offset 1 instead of 0.
     * The last entry of ori_offset is the total number of orientations, which
     * we also want to keep. */
    ori_offset[0] = 0;
    std::inclusive_scan( &ori_count[0],
                         &ori_count[total_ext_ct-1],
                         &ori_offset[1] );
    const int total_ori = ori_offset[total_ext_ct];

    for( int i=0; i<total_ext_ct; i++ )
    {
        extremum->idx_ori = ori_offset[i];
    }

    /* For every orientation (there are total_ori of them), store the extremum
     * to which they belong in the array feat_to_ext_map. */
    int* feat_to_ext_map = dobuf.feat_to_ext_map;

    int ftem = 0;
    for( int extr=0; extr<total_ext_ct; extr++ )
    {
        for( int ori=0; ori<ori_offset[extr+1]; ori++ )
        {
            feat_to_ext_map[ftem++] = extr;
        }
    }

    /* Copy the extreme count for every octave from the (already initialized)
     * array ext_ct to the (uninitialized) array ext_ps. */
    std::copy( &dct.ext_ct[0],
               &dct.ext_ct[MAX_OCTAVES],
               &dct.ext_ps[0] );
    /* Compute the exclusive prefix sum on the array ext_ps. */
    std::exclusive_scan( &dct.ext_ps[0],
                         &dct.ext_ps[MAX_OCTAVES],
                         &dct.ext_ps[0],
                         0 );

    /* Fill the array ori_ct with the number of orientations that belong
     * the octave given by the index. */
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

    /* Like above, compute the exclusive prefix sum of all orientations
     * in ori_ps. */
    std::copy( &dct.ori_ct[0],
               &dct.ori_ct[MAX_OCTAVES],
               &dct.ori_ps[0] );
    std::exclusive_scan( &dct.ori_ps[0],
                         &dct.ori_ps[MAX_OCTAVES],
                         &dct.ori_ps[0],
                         0 );

    /* Store the total number of orientations and the total number of
     * extrema as well. */
    dct.ori_total = dct.ori_ps[MAX_OCTAVES-1] + dct.ori_ct[MAX_OCTAVES-1];
    dct.ext_total = dct.ext_ps[MAX_OCTAVES-1] + dct.ext_ct[MAX_OCTAVES-1];
}

void Pyramid::orientation( const Config& conf )
{
    POP_INFO2( conf.silent(), "enter " << __PRETTY_FUNCTION__ );

    int ext_total = 0;
    for(int o : dct.ext_ct)
    {
        POP_INFO2( conf.silent(), "octave has " << o << " extrema" );
        if( o > 0 )
        {
            ext_total += o;
        }
    }
    POP_INFO2( conf.silent(), "total number of extrema is " << ext_total );

    // Filter functions are only called if necessary. They are very expensive,
    // therefore add 10% slack.
    if( conf.getFilterMaxExtrema() > 0 && int(conf.getFilterMaxExtrema()*1.1) < ext_total )
    {
        ext_total = extrema_filter_grid( conf, ext_total );
    }

    reallocExtrema( ext_total );

    int ext_ct_prefix_sum = 0;
    for( int octave=0; octave<_num_octaves; octave++ ) {
        dct.ext_ps[octave] = ext_ct_prefix_sum;
        ext_ct_prefix_sum += dct.ext_ct[octave];
    }
    dct.ext_total = ext_ct_prefix_sum;

    // for( int octave=0; octave<_num_octaves; octave++ )
    for( int octave=_num_octaves-1; octave>=0; octave-- )
    {
        Octave&      oct_obj = _octaves[octave];

        int num = dct.ext_ct[octave];

        if( num > 0 ) {
            Grid g;
            g.setGridDim( num );
            g.setBlockDim( 1 );

            ori_par( g,
                     octave,
                     dct.ext_ps[octave],
                     oct_obj.getData( ),
                     oct_obj.getWidth( ),
                     oct_obj.getHeight( ) );
        }
    }

    /* Compute and set the orientation prefixes on the device */
    ori_prefix_sum( ext_ct_prefix_sum,
                    _num_octaves );
}

