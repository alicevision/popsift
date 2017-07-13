/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <math.h>
#include <stdio.h>
#include <inttypes.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/sort.h>

#include "common/assist.h"
#include "sift_pyramid.h"
#include "sift_constants.h"
#include "s_gradiant.h"
#include "common/excl_blk_prefix_sum.h"
#include "common/warp_bitonic_sort.h"
#include "common/debug_macros.h"

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
template<int HEIGHT>
__global__
void ori_par( const int           octave,
              const int           ext_ct_prefix_sum,
              cudaTextureObject_t layer,
              const int           w,
              const int           h )
{
    const int extremum_index  = blockIdx.x * blockDim.y + threadIdx.y;

    if( extremum_index >= dct.ext_ct[octave] ) return; // a few trailing warps

    const InitialExtremum* iext = &dobuf.i_ext[octave][extremum_index];

    __shared__ float hist   [HEIGHT][ORI_NBINS];
    __shared__ float sm_hist[HEIGHT][ORI_NBINS];

    for( int i = threadIdx.x; i < ORI_NBINS; i += blockDim.x )  hist[threadIdx.y][i] = 0.0f;

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

    for( int i = threadIdx.x; ::__any(i < loops); i += blockDim.x )
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

                atomicAdd( &hist[threadIdx.y][bidx], weight );
            }
        }
        __syncthreads();
    }

#ifdef WITH_VLFEAT_SMOOTHING
    for( int i=0; i<3; i++ ) {
        for( int bin = threadIdx.x; bin < ORI_NBINS; bin += blockDim.x ) {
            int prev = bin == 0 ? ORI_NBINS-1 : bin-1;
            int next = bin == ORI_NBINS-1 ? 0 : bin+1;
            sm_hist[threadIdx.y][bin] = ( hist[threadIdx.y][prev] + hist[threadIdx.y][bin] + hist[threadIdx.y][next] ) / 3.0f;
        }
        __syncthreads();
        for( int bin = threadIdx.x; bin < ORI_NBINS; bin += blockDim.x ) {
            int prev = bin == 0 ? ORI_NBINS-1 : bin-1;
            int next = bin == ORI_NBINS-1 ? 0 : bin+1;
            hist[threadIdx.y][bin] = ( sm_hist[threadIdx.y][prev] + sm_hist[threadIdx.y][bin] + sm_hist[threadIdx.y][next] ) / 3.0f;
        }
        __syncthreads();
    }
    for( int bin = threadIdx.x; bin < ORI_NBINS; bin += blockDim.x ) {
        sm_hist[threadIdx.y][bin] = hist[threadIdx.y][bin];
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
        sm_hist[threadIdx.y][bin] = (   hist[threadIdx.y][prev2] + hist[threadIdx.y][next2]
                         + ( hist[threadIdx.y][prev1] + hist[threadIdx.y][next1] ) * 4.0f
                         +   hist[threadIdx.y][bin] * 6.0f ) / 16.0f;
    }
    __syncthreads();
#endif // not WITH_VLFEAT_SMOOTHING

    // sub-cell refinement of the histogram cell index, yielding the angle
    // not necessary to initialize, every cell is computed
    __shared__ float refined_angle[HEIGHT][64];
    __shared__ float yval         [HEIGHT][64];

    for( int bin = threadIdx.x; ::__any( bin < ORI_NBINS ); bin += blockDim.x ) {
        const int prev = bin == 0 ? ORI_NBINS-1 : bin-1;
        const int next = bin == ORI_NBINS-1 ? 0 : bin+1;

        bool predicate = ( bin < ORI_NBINS ) && ( sm_hist[threadIdx.y][bin] > max( sm_hist[threadIdx.y][prev], sm_hist[threadIdx.y][next] ) );

        // const float num  = predicate ? 3.0f *   sm_hist[threadIdx.y][prev] - 4.0f * sm_hist[threadIdx.y][bin] + sm_hist[threadIdx.y][next]   : 0.0f;
        const float num  = predicate ?   2.0f * sm_hist[threadIdx.y][prev]
                                       - 4.0f * sm_hist[threadIdx.y][bin]
                                       + 2.0f * sm_hist[threadIdx.y][next]
                                     : 0.0f;
        const float denB = predicate ? 2.0f * ( sm_hist[threadIdx.y][prev] - 2.0f * sm_hist[threadIdx.y][bin] + sm_hist[threadIdx.y][next] ) : 1.0f;

        const float newbin = __fdividef( num, denB ); // verified: accuracy OK

        predicate   = ( predicate && newbin >= 0.0f && newbin <= 2.0f );

        refined_angle[threadIdx.y][bin] = predicate ? prev + newbin : -1;
        yval[threadIdx.y][bin]          = predicate ?  -(num*num) / (4.0f * denB) + sm_hist[threadIdx.y][prev] : -INFINITY;
    }

    int2 best_index = make_int2( threadIdx.x, threadIdx.x + 32 );

    BitonicSort::Warp32<float> sorter( yval[threadIdx.y] );
    sorter.sort64( best_index );
    __syncthreads();

    // All threads retrieve the yval of thread 0, the largest
    // of all yvals.
    const float best_val = yval[threadIdx.y][best_index.x];
    const float yval_ref = 0.8f * __shfl( best_val, 0 );
    const bool  valid    = ( best_val >= yval_ref );
    bool        written  = false;

    Extremum* ext = &dobuf.extrema[ext_ct_prefix_sum + extremum_index];

    if( threadIdx.x < ORIENTATION_MAX_COUNT ) {
        if( valid ) {
            float chosen_bin = refined_angle[threadIdx.y][best_index.x];
            if( chosen_bin >= ORI_NBINS ) chosen_bin -= ORI_NBINS;
            // float th = __fdividef(M_PI2 * chosen_bin , ORI_NBINS) - M_PI;
            float th = ::fmaf( M_PI2 * chosen_bin, 1.0f/ORI_NBINS, - M_PI );
            ext->orientation[threadIdx.x] = th;
            written = true;
        }
    }

    int angles = __popc( __ballot( written ) );
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

struct FunctionExtractCell
{
    __device__ thrust::tuple<int,float> operator()( const thrust::tuple<int,int>& val) const
    {
        const int octave = thrust::get<0>(val);
        const int idx    = thrust::get<1>(val);
        InitialExtremum& e = dobuf.i_ext[octave][idx];

        return  thrust::make_tuple( e.cell, e.sigma * powf( 2.0f, octave ) );
    }

};

struct FunctionReversePosition
{
    const int _total;
    FunctionReversePosition( int total ) : _total(total) { }

    __host__ __device__ int operator()(int val) const
    {
        return _total - val - 1;
    }
};

struct FunctionIsAbove
{
    int _limit;
    FunctionIsAbove( int limit ) : _limit(limit) { }

    __host__ __device__ bool operator()( int val ) const
    {
        return val > _limit;
    }
};

__host__
void Pyramid::extrema_filter_grid( const Config& conf, int ext_total )
{
    /* At this time, we have host-side information about ext_ct[o], the number
     * of extrema we have found in octave o, and we have summed it up on the
     * host size. However, other values in the hct and dct data structures
     * have not been computed yet.
     * The extrema are only known in the InitialExtrema structure. We want to
     * perform grid filtering before their orientation is computed and they
     * are copied into the larger Extrema data structure.
     */
    const int slots = conf.getGridSize();

    thrust::device_vector<int>   octave_index( ext_total );
    thrust::device_vector<int>   iext_index  ( ext_total );
    thrust::device_vector<int>   cell_values ( ext_total );
    thrust::device_vector<float> scale_values( ext_total );
    thrust::device_vector<int>   permutator  ( ext_total );
    thrust::device_vector<int>   grid        ( ext_total );
    thrust::device_vector<int>   cell_counts ( slots * slots );
    thrust::device_vector<int>   cell_offsets( slots * slots );

    thrust::sequence( permutator.begin(), permutator.end() );

    int sum = 0;
    for( int o=0; o<MAX_OCTAVES; o++ ) {
        const int ocount = hct.ext_ct[o];
        if( ocount > 0 ) {
            // fill a continuous device array with octave of all initial extrema
            thrust::fill(     octave_index.begin() + sum, octave_index.begin() + sum + ocount, o );
            // fill a continuous device array with index within octave of all initial extrema
            thrust::sequence( iext_index.begin()   + sum, iext_index  .begin() + sum + ocount );
            sum += ocount;
        }
    }

    FunctionExtractCell fun_extract_cell;

    // extract cell and scale value for all initial extrema
    thrust::transform( thrust::make_zip_iterator( thrust::make_tuple( octave_index.begin(),
                                                                      iext_index.begin() ) ),
                       thrust::make_zip_iterator( thrust::make_tuple( octave_index.end(),
                                                                      iext_index.end() ) ),
                       thrust::make_zip_iterator( thrust::make_tuple( cell_values.begin(),
                                                                      scale_values.begin() ) ),
                       fun_extract_cell );
    // sort (octave,index,scale) tuples by their cell values
    thrust::sort_by_key( cell_values.begin(), cell_values.end(),
                         thrust::make_zip_iterator( thrust::make_tuple( octave_index.begin(),
                                                                        iext_index.  begin(),
                                                                        scale_values.begin() ) ) );

    // count the number of entries in all cells (in one operation instead of several reduce_if)
    thrust::reduce_by_key( cell_values.begin(), cell_values.end(),
                           thrust::make_constant_iterator(1),
                           thrust::make_discard_iterator(),
                           cell_counts.begin() );

    // compute the offsets from cell_values start for each of the (pre-sorted) cell values
    thrust::exclusive_scan( cell_counts.begin(), cell_counts.end(), cell_offsets.begin() );

    const int n = slots * slots;
    thrust::host_vector<int> h_cell_counts         ( n );
    thrust::host_vector<int> h_cell_permute        ( n );
    thrust::host_vector<int> cell_count_prefix_sums( n );
    thrust::host_vector<int> cell_count_sumup      ( n );

    // move to host code - computing the limits on the GPU is too wasteful
    h_cell_counts = cell_counts;

    std::cout << "BEGIN cell value counters" << std::endl;
    std::copy( h_cell_counts  .begin(), h_cell_counts  .end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
    std::cout << "END cell value counters" << std::endl;

    // the cell filter algorithm requires the cell counts in increasing order, cell_permute
    // maps new position to original index
    thrust::sequence( h_cell_permute.begin(), h_cell_permute.end() );
    thrust::sort_by_key( h_cell_counts.begin(), h_cell_counts.end(), h_cell_permute.begin() );

    // several steps to find the cells that must loose extrema

    // inclusive prefix sum
    thrust::inclusive_scan( h_cell_counts.begin(), h_cell_counts.end(), cell_count_prefix_sums.begin() );

    FunctionReversePosition fun_reverse_pos( n );

    // sumup[i] = prefix sum[i] + sum( cell[i] copied into remaining cells )
    thrust::transform( h_cell_counts.begin(), h_cell_counts.end(),
                       thrust::make_transform_iterator( thrust::make_counting_iterator<int>(0),
                                                        fun_reverse_pos ),
                       cell_count_sumup.begin(),
                       thrust::multiplies<int>() );
    thrust::transform( cell_count_sumup.begin(), cell_count_sumup.end(),
                       cell_count_prefix_sums.begin(),
                       cell_count_sumup.begin(),
                       thrust::plus<int>() );

    FunctionIsAbove function_is_above( conf.getMaxExtrema() );

    // count cells that are above the extrema limit after the summing. Those must share the
    // reduction of extrema
    int ct = thrust::count_if( cell_count_sumup.begin(), cell_count_sumup.end(),
                               function_is_above );

    float tailaverage = float( thrust::reduce( &h_cell_counts[n-ct], &h_cell_counts[n] ) ) / ct;

    int   newlimit    = ::ceilf( tailaverage - ( ext_total - conf.getMaxExtrema() ) / ct );

    // clamp all cells to the computed limit - the total is now less than n extrema off
    thrust::transform( h_cell_counts.begin(), h_cell_counts.end(), 
                       thrust::make_constant_iterator<int>(newlimit),
                       h_cell_counts.begin(),
                       thrust::minimum<int>() );

    // back to original order
    thrust::sort_by_key( h_cell_permute.begin(), h_cell_permute.end(), h_cell_counts.begin() );

    // transfer counts back to device
    cell_counts = h_cell_counts;

    if( 1 ) // filter condition is random selection
    {
    }
    else if( 0 ) // filter condition is largest scale first
    {
    }
    else if( 0 ) // filter condition is smallest scale first
    {
    }

#if 1
    // thrust::host_vector<int> h_octave_index( ext_total );
    // h_octave_index = octave_index;
    // std::cout << "BEGIN octave numbers" << std::endl;
    // std::copy( h_octave_index.begin(), h_octave_index.end(), std::ostream_iterator<int>(std::cout, " "));
    // std::cout << std::endl;
    // std::cout << "END octave numbers" << std::endl;

    // thrust::host_vector<int> h_iext_index(   ext_total );
    // h_iext_index   = iext_index;
    // std::cout << "BEGIN in-octave index" << std::endl;
    // std::copy( h_iext_index  .begin(), h_iext_index  .end(), std::ostream_iterator<int>(std::cout, " "));
    // std::cout << std::endl;
    // std::cout << "END in-octave index" << std::endl;

    // thrust::host_vector<int> h_cell_values(   ext_total );
    // h_cell_values   = cell_values;
    // std::cout << "BEGIN cell values" << std::endl;
    // std::copy( h_cell_values  .begin(), h_cell_values  .end(), std::ostream_iterator<int>(std::cout, " "));
    // std::cout << std::endl;
    // std::cout << "END cell values" << std::endl;

    std::cout << "BEGIN cell value counters" << std::endl;
    std::copy( h_cell_counts  .begin(), h_cell_counts  .end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
    std::cout << "END cell value counters" << std::endl;

    std::cout << "BEGIN original cell index" << std::endl;
    std::copy( h_cell_permute  .begin(), h_cell_permute  .end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
    std::cout << "END original cell index" << std::endl;
#endif
}

__host__
void Pyramid::orientation( const Config& conf )
{
    nvtxRangePushA( "reading extrema count" );
    readDescCountersFromDevice( );
    nvtxRangePop( );

    nvtxRangePushA( "reallocating extrema arrays" );
    int ext_total = 0;
    for( int o=0; o<MAX_OCTAVES; o++ ) {
        if( hct.ext_ct[o] > 0 ) {
            ext_total += hct.ext_ct[o];
        }
    }

    extrema_filter_grid( conf, ext_total );

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

            if( num > 32 ) {
                block.x = 32;
                block.y = 2;
                grid.x  = grid_divide( num, 2 );

                ori_par<2>
                    <<<grid,block,0,oct_str>>>
                    ( octave,
                      hct.ext_ps[octave],
                      oct_obj.getDataTexPoint( ),
                      oct_obj.getWidth( ),
                      oct_obj.getHeight( ) );
            } else {
                block.x = 32;
                block.y = 1;
                grid.x  = num;

                ori_par<1>
                    <<<grid,block,0,oct_str>>>
                    ( octave,
                      hct.ext_ps[octave],
                      oct_obj.getDataTexPoint( ),
                      oct_obj.getWidth( ),
                      oct_obj.getHeight( ) );
            }
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

    cudaDeviceSynchronize();
}

