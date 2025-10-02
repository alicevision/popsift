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
#include <iterator>

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
float smoothe( const std::vector<float>& src, const int bin )
{
    const int prev = (bin <= 0) ? ORI_NBINS-1 : bin-1;
    const int next = (bin >= ORI_NBINS-1) ? 0 : bin+1;

    const float f  = ( src.at(prev) + src.at(bin) + src.at(next) ) / 3.0f;

    return f;
}

/*
 * Compute the keypoint orientations for each extremum.
 * Direct curve fitting approach.
 */
static
void compute_all_orientations( const int            extremum_index,
                               const int            octave,
                               const Plane2D_float& layer,
                               const int            w,
                               const int            h )
{
    const int              iext_off = dct.initial_extrema_offset   [octave][extremum_index];
    const InitialExtremum& iext     = dct.initial_extrema_in_octave[octave][iext_off];

    std::vector<float> hist( ORI_NBINS, 0 );

    /* keypoint fractional geometry */
    const float x     = iext.xpos;
    const float y     = iext.ypos;
    const int   level = iext.lpos; // old_level;
    const float sig   = iext.sigma;

    /* orientation histogram radius */
    const float  sigw = ORI_WINFACTOR * sig;
    const int32_t rad  = (int)roundf((3.0f * sigw));

    // const float factor = __fdividef( -0.5f, (sigw * sigw) );
    const float factor = -0.5f / (sigw * sigw);

    // GRIFF: BUG?
    // This is from the historical PopSift code. When I actually print the samples, this is no circle and it makes no sense
    const int sq_thres  = rad * rad;
    // const int sq_thres  = rad;

    // int xmin = max(1,     (int)floor(x - rad));
    // int xmax = min(w - 2, (int)floor(x + rad));
    // int ymin = max(1,     (int)floor(y - rad));
    // int ymax = min(h - 2, (int)floor(y + rad));
    int xmin = std::max(1,     (int)roundf(x) - rad);
    int xmax = std::min(w - 2, (int)roundf(x) + rad);
    int ymin = std::max(1,     (int)roundf(y) - rad);
    int ymax = std::min(h - 2, (int)roundf(y) + rad);

    const int wx = xmax - xmin + 1;
    const int hy = ymax - ymin + 1;
    // int loops = wx * hy;

    std::ostringstream debug_ostr;
    std::ostringstream debug_ostr2;
    std::ostringstream debug_ostr3;
    std::ostringstream debug_ostr4;
    std::ostringstream debug_ostr5;

    debug_ostr << "Histogram around (" << x << ", " << y << "):" << std::endl;
    // debug_ostr2 << "Gradiants:" << std::endl;
    debug_ostr2 << "Gradients:" << std::endl;
    debug_ostr3 << "Bin:" << std::endl;
    debug_ostr4 << "Delta to theta:" << std::endl;
    debug_ostr5 << "Area:" << std::endl;

    for( int y_idx = 0; y_idx < hy; y_idx++ )
    {
        for( int x_idx = 0; x_idx < wx; x_idx++ )
        // for( int i = 0; i < loops; i++ )
        {
            // int yy = i / wx + ymin;
            // int xx = i % wx + xmin;
            int yy = y_idx + ymin;
            int xx = x_idx + xmin;

            const float dx = xx - x;
            const float dy = yy - y;

            debug_ostr5 << setprecision(3) << layer.get( level, yy, xx ) << " ";

            const int sq_dist  = dx * dx + dy * dy;
            if (sq_dist <= sq_thres)
            {
                debug_ostr << "(" << xx << ", " << yy << ") ";

                float grad;
                float theta;
                get_gradiant32( grad,
                                theta,
                                xx,
                                yy,
                                layer,
                                level,
                                debug_ostr4 );

                debug_ostr2 << std::setprecision(3) << grad << " ";

                float weight = grad * expf(sq_dist * factor);

                // int bidx = (int)rintf( __fdividef( ORI_NBINS * (theta + M_PI), M_PI2 ) );
                // int bidx = (int)roundf( __fdividef( float(ORI_NBINS) * (theta + M_PI), M_PI2 ) );
                int bidx = (int)roundf( float(ORI_NBINS) * (theta + M_PI) / M_PI2 );

                // debug_ostr3 << std::setprecision(3) << theta << " ";
                debug_ostr3 << std::setprecision(3) << theta << " (" << bidx << ") ";

                if( bidx > ORI_NBINS ) {
                    printf("Crashing: bin %d theta %f :-)\n", bidx, theta);
                }
                if( bidx < 0 ) {
                    printf("Crashing: bin %d theta %f :-)\n", bidx, theta);
                }

                bidx = (bidx == ORI_NBINS) ? 0 : bidx;

                hist.at(bidx) += weight;
                // debug_ostr2 << std::setprecision(3) << (int)weight << " ";
            }
        }
        debug_ostr << std::endl;
        debug_ostr2 << std::endl;
        debug_ostr3 << std::endl;
        debug_ostr4 << std::endl;
        debug_ostr5 << std::endl;
    }

    // POP_INFO2( false, debug_ostr.str() );
    // POP_INFO2( false, debug_ostr2.str() );
    // POP_INFO2( false, debug_ostr3.str() );
    // POP_INFO2( false, debug_ostr4.str() );
    // POP_INFO2( false, debug_ostr5.str() );

    std::vector<float> sm_hist(ORI_NBINS);

#ifdef WITH_VLFEAT_SMOOTHING
    // outer loop: smoothe 3 times
    for( int i=0; i<3 ; i++ )
    {
        for( int j=0; j<ORI_NBINS; j++ ) {
            sm_hist[j] = smoothe( hist, j );
        }
        for( int j=0; j<ORI_NBINS; j++ ) {
            hist[j] = smoothe( sm_hist, j );
        }
    }

    for( int i=0; i<ORI_NBINS; i++ ) {
        sm_hist[i] = hist[i];
    }
#else // not WITH_VLFEAT_SMOOTHING
    for( int bin = 0; bin < ORI_NBINS; bin++ )
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

    // debug output block: smoothed histogram
#if 0
    {
        std::ostringstream debug_ostr;
        debug_ostr << "smoothed histogram: " << setprecision(4);
        std::copy( sm_hist.begin(),
                   sm_hist.end(),
                   std::ostream_iterator<float>(debug_ostr, " ") );
        POP_INFO2( false, debug_ostr.str() );
    }
#endif

    std::vector<float> yval         (ORI_NBINS);
    std::vector<float> refined_angle(ORI_NBINS);

    // sub-cell refinement of the histogram cell index, yielding the angle
    // not necessary to initialize, every cell is computed
    //
    // Note: without GPU or SIMD, it would be faster to initialize yval to all -INFINITY
    //       and refined_angle to all -1, and use if() instead of the predicates.
    //       With GPU or SIMD, probably slower.

    for( int bin = 0; bin < ORI_NBINS; bin ++ )
    {
        const int prev = bin == 0 ? ORI_NBINS-1 : bin-1;
        const int next = bin == ORI_NBINS-1 ? 0 : bin+1;

        bool predicate = ( sm_hist[bin] > max( sm_hist[prev], sm_hist[next] ) );

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

#if 0
    // debug output block: prev + newbin ???
    {
        std::ostringstream debug_ostr;
        debug_ostr << "refined angles: " << setprecision(4);
        std::copy( refined_angle.begin(),
                   refined_angle.end(),
                   std::ostream_iterator<float>(debug_ostr, " ") );
        debug_ostr << std::endl
                   << "refined peak value: " << setprecision(4);
        std::copy( yval.begin(),
                   yval.end(),
                   std::ostream_iterator<float>(debug_ostr, " ") );
        POP_INFO2( false, debug_ostr.str() );
    }
#endif

    std::vector<int> best_index(ORI_NBINS);

    /* initialize array best_index with the indices of array yval */
    std::iota( best_index.begin(), best_index.end(), 0 );

    /* Sort array best_index contain yval indices in order of _decreasing_ yval values.
     * Change from sort to stable_sort to see more clearly if anything goes wrong. */
    std::sort( best_index.begin(),
               best_index.end(),
               [&]( const int& l, const int& r ) {
                   const int l_idx = l;
                   const int r_idx = r;
                   return ( yval[l_idx] > yval[r_idx] );
               } );

#if 0
    // debug output block
    {
        std::ostringstream debug_ostr;
        debug_ostr << "best index array after sorting: ";
        std::copy( best_index.begin(),
                   best_index.end(),
                   std::ostream_iterator<int>(debug_ostr, " ") );
        debug_ostr << std::endl;
        POP_INFO2( false, debug_ostr.str() );
    }
#endif

    Extremum ext;

    int angles = 0;

    // All threads retrieve the yval of thread 0, the largest
    // of all yvals.
    for( int i=0; i<ORIENTATION_MAX_COUNT; i++ )
    {
        /* An alternative orientation is only accepted if its smoothed
         * value is greater or equal 80% of the best value. */
        const int   this_idx = best_index[i];
        const float this_val = yval[this_idx];
        const float acceptance_threshold = 0.8f * yval[best_index[0]];
        const bool  accepted    = ( this_val >= acceptance_threshold );

        if( accepted )
        {
            /* Convert the bin index (default 0..35) into a rotation expressed
             * in a fraction of 2 PI. */
            float chosen_bin = refined_angle[this_idx];
            if( chosen_bin >= ORI_NBINS ) chosen_bin -= ORI_NBINS;
            // float th = __fdividef(M_PI2 * chosen_bin , ORI_NBINS) - M_PI;
            // float th = std::fmaf( M_PI2 * chosen_bin, 1.0f/ORI_NBINS, - M_PI );
            float th = M_PI2 * chosen_bin / ORI_NBINS - M_PI;
            ext.orientation[i] = th;

            /* Increase the number of accepted angles. */
            angles += 1;
        }
    }

#if 1
    // the selected angles
    {
        std::ostringstream debug_ostr;
        debug_ostr << "Result for the pixel at ( " << iext.xpos << "," << iext.ypos << "," << iext.lpos << ") is : " << angles << " selected angles: ";
        for( int i=0; i<angles; i++ )
            debug_ostr << std::fixed << std::setprecision(3) << ext.orientation[i] / M_PI2 * 360.0f << " ";
        debug_ostr << std::endl;
        POP_INFO2( false, debug_ostr.str() );
    }
#endif

    ext.xpos    = iext.xpos;
    ext.ypos    = iext.ypos;
    ext.lpos    = iext.lpos;
    ext.sigma   = iext.sigma;
    ext.octave  = octave;
    ext.num_ori = angles;

    dbuf.extrema.emplace_back( ext );
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

void ori_prefix_sum( const int num_octaves )
{
    if( dct.extrema_count_total < 1 )
    {
        POP_FATAL("Calling " << __FUNCTION__ << " with " << dct.extrema_count_total << " found extrema");
    }

    std::vector<Extremum>& all_extrema = dbuf.extrema;

    assert( all_extrema.size() == dct.extrema_count_total );

    vector<int> ori_count;

    /* collect the numbers of orientation for every extremum in ori_count */
    // for( int i=0; i<dct.extrema_count_total; i++ )
    // {
        // ori_count[i] = all_extrema.num_ori;
    // }
    std::ostringstream debug_ostr;
    debug_ostr << "Number of orientations:";
    for( auto ext : all_extrema )
    {
        debug_ostr << ext.num_ori << " ";
        ori_count.push_back( ext.num_ori );
    }
    POP_INFO2( false, debug_ostr.str() );

    if( ori_count.size() != dct.extrema_count_total )
    {
        POP_FATAL( "Number of counted extrema: " << dct.extrema_count_total << ", orientation counters pushed to ori_count: " << ori_count.size() );
    }

    /* set ori_offset[0] to 0,
     * then compute an inclusive prefix sum for the values in ori_count into
     * the target array ori_offset, but starting at offset 1 instead of 0.
     * The last entry of ori_offset is the total number of orientations, which
     * we also want to keep.
     */
    vector<int> ori_offset( dct.extrema_count_total+1 );
    ori_offset[0] = 0;
    std::inclusive_scan( ori_count.begin(), ori_count.end(), &ori_offset[1] );
    const int total_ori = ori_offset[dct.extrema_count_total];
#if 1
    { 
        ostringstream debug_ostr1;
        debug_ostr1 << "Orientation count array: ";
        for( int i=0; i<ori_count.size(); i++ )
            debug_ostr1 << ori_count[i] << " ";

        ostringstream debug_ostr2;
        debug_ostr2 << "Orientation offset array: ";
        for( int i=0; i<ori_offset.size(); i++ )
            debug_ostr2 << ori_offset[i] << " ";
        debug_ostr2 << "(prefix sum of count aray)";

        POP_INFO2( false, debug_ostr1.str() );
        POP_INFO2( false, debug_ostr2.str() );
    }
#endif
    POP_INFO2( false, "Total number of orientations: " << total_ori );

    for( int i=0; i<dct.extrema_count_total; i++ )
    {
        all_extrema[i].idx_ori = ori_offset[i];
    }

    /* For every orientation (there are total_ori of them), store the extremum
     * to which they belong in the array feat_to_ext_map. */
    std::vector<int>& feat_to_ext_map = dbuf.feat_to_ext_map;

    if( dbuf.feat_to_ext_map.size() != 0 )
    {
        POP_FATAL( "Programming error in " << __FILE__ << ":" << __LINE__ << ": reverse map size should be 0 but it is " << dbuf.feat_to_ext_map.size() );
    }

    for( int extr=0; extr<dct.extrema_count_total; extr++ )
    {
        for( int ori=0; ori<ori_count[extr]; ori++ )
        {
            feat_to_ext_map.push_back( extr );
        }
    }

#if 1
    {
        ostringstream debug_ostr;
        debug_ostr << "Reverse map from orientation index to extremum index: ";
        for( auto map : feat_to_ext_map )
        {
            debug_ostr << map << " ";
        }
        POP_INFO2( false, debug_ostr.str() );
    }
#endif

    /* Fill the array ori_ct with the number of orientations that belong
     * to the octave given by the index. */
    for( int o=0; o<MAX_OCTAVES; o++ ) {
        if( dct.extrema_count_per_octave[o] == 0 ) {
            dct.ori_ct[o] = 0;
        } else {
            int fe = dct.extrema_count_prefix_sum[o  ];   /* first extremum for this octave */
            int le = dct.extrema_count_prefix_sum[o+1]-1; /* last  extremum for this octave */
            int lo_ori_index = dbuf.extrema[fe].idx_ori;
            int num_ori      = dbuf.extrema[le].num_ori;
            int hi_ori_index = dbuf.extrema[le].idx_ori + num_ori;
            dct.ori_ct[o] = hi_ori_index - lo_ori_index;
        }
    }

    /* Like above, compute the exclusive prefix sum of all orientations
     * in ori_ps. */
#if 1
    std::cerr << "Orientations by octave (dct.ori_ct): ";
    for( int i=0; i<MAX_OCTAVES; i++ ) std::cerr << dct.ori_ct[i] << " ";
    std::cerr << std::endl;
#endif

    std::copy( &dct.ori_ct[0],
               &dct.ori_ct[MAX_OCTAVES],
               &dct.ori_ps[0] );
#if 1
    std::cerr << "Orientations by octave (dct.ori_ps): ";
    for( int i=0; i<MAX_OCTAVES; i++ ) std::cerr << dct.ori_ps[i] << " ";
    std::cerr << std::endl;
#endif
    std::exclusive_scan( &dct.ori_ps[0],
                         &dct.ori_ps[MAX_OCTAVES],
                         &dct.ori_ps[0],
                         0 );
#if 1
    std::cerr << "Exclusive prefix sum of orientations (dct.ori_ps): ";
    for( int i=0; i<MAX_OCTAVES; i++ ) std::cerr << dct.ori_ps[i] << " ";
    std::cerr << std::endl;
#endif

    /* Store the total number of orientations and the total number of
     * extrema as well. */
    dct.ori_total = dct.ori_ps[MAX_OCTAVES-1] + dct.ori_ct[MAX_OCTAVES-1];
}

void Pyramid::orientation( const Config& conf )
{
    POP_INFO2( conf.silent(), "enter " << __PRETTY_FUNCTION__ );

    int debug_octave = 0;
    int ext_total = 0;
    for( int n : dct.extrema_count_per_octave )
    {
        POP_INFO2( conf.silent(), "octave " << debug_octave << " has " << n << " extrema" );
        if( n > 0 )
        {
            ext_total += n;
        }
        debug_octave++;
    }
    POP_INFO2( conf.silent(), "total number of extrema is " << ext_total );

    // Filter functions are only called if necessary. They are very expensive,
    // therefore add 10% slack.
    if( conf.getFilterMaxExtrema() > 0 && int(conf.getFilterMaxExtrema()*1.1) < ext_total )
    {
        ext_total = extrema_filter_grid( conf, ext_total );
    }

    reallocExtrema( ext_total );

    for( int octave=0; octave<_num_octaves; octave++ )
    // for( int octave=_num_octaves-1; octave>=0; octave-- )
    {
        Octave&      oct_obj = _octaves[octave];

        int extrema_count = dct.extrema_count_per_octave[octave];

        if( extrema_count > 0 )
        {
            POP_INFO2( conf.silent(), "Octave " << octave << " has " << extrema_count << " extrema" );
            for( int ext_idx=0; ext_idx<extrema_count; ext_idx++ )
            {
                compute_all_orientations( ext_idx,
                                          octave,
                                          oct_obj.getData( ),
                                          oct_obj.getWidth( ),
                                          oct_obj.getHeight( ) );
            }
            POP_INFO2( conf.silent(), "Found all orientations" );
        }
    }

    POP_INFO2( conf.silent(), "Computing prefix sums for all orientations" );
    /* Compute and set the orientation prefixes on the device */
    ori_prefix_sum( _num_octaves );
    POP_INFO2( conf.silent(), "Done" );
}

