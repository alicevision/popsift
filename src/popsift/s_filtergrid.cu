/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "sift_config.h"
#include "sift_extremum.h"
#include "sift_pyramid.h"

#if POPSIFT_IS_DEFINED(POPSIFT_USE_NVTX)
#include <nvToolsExtCuda.h>
#else
#define nvtxRangePushA(a)
#define nvtxRangePop()
#endif

#if ! POPSIFT_IS_DEFINED(POPSIFT_DISABLE_GRID_FILTER)

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

namespace popsift
{

struct FunctionSort_IncCell_DecScale
{
    __device__
    inline bool operator()( const thrust::tuple<int,float>& l, const thrust::tuple<int,float>& r ) const
    {
        return ( ( thrust::get<0>(l)  < thrust::get<0>(r) ) ||
                 ( thrust::get<0>(l) == thrust::get<0>(r) && thrust::get<1>(l) > thrust::get<1>(r) ) );
    }
};

struct FunctionSort_IncCell_IncScale
{
    __device__
    inline bool operator()( const thrust::tuple<int,float>& l, const thrust::tuple<int,float>& r ) const
    {
        return ( ( thrust::get<0>(l)  < thrust::get<0>(r) ) ||
                 ( thrust::get<0>(l) == thrust::get<0>(r) && thrust::get<1>(l) < thrust::get<1>(r) ) );
    }
};

struct FunctionExtractCell
{
    __device__
    inline thrust::tuple<int,float> operator()( const thrust::tuple<int,int>& val) const
    {
        /* During the filter stage, the i_ext_dat array is still compact (all intial
         * extrema do still have ignore==false), so that we can access every entry
         * directly).
         */
        const int octave = thrust::get<0>(val);
        const int idx    = thrust::get<1>(val);
        InitialExtremum& e = dobuf.i_ext_dat[octave][idx];

        return  thrust::make_tuple( e.cell, e.sigma * powf( 2.0f, octave ) );
    }
};

struct FunctionIsAbove
{
    int _limit;
    explicit FunctionIsAbove( int limit ) : _limit(limit) { }

    __host__ __device__
    inline bool operator()( int val ) const
    {
        return val > _limit;
    }
};

struct FunctionDisableExtremum
{
    __device__
    inline void operator()( const thrust::tuple<int,int>& val) const
    {
        const int octave = thrust::get<0>(val);
        const int idx    = thrust::get<1>(val);
        InitialExtremum& e = dobuf.i_ext_dat[octave][idx];
        e.ignore = true;
    }
};

struct FunctionExtractIgnored
{
    __device__
    inline int operator()( int idx, int octave ) const
    {
        InitialExtremum& e = dobuf.i_ext_dat[octave][idx];
        if( e.ignore )
            return 0;
        else
            return 1;
    }
};

/* discard extrema that exceed a conf.getFilterMaxExtrema() */
__host__
int Pyramid::extrema_filter_grid( const Config& conf, int ext_total )
{
    /* At this time, we have host-side information about ext_ct[o], the number
     * of extrema we have found in octave o, and we have summed it up on the
     * host size. However, other values in the hct and dct data structures
     * have not been computed yet.
     * The extrema are only known in the InitialExtrema structure. We want to
     * perform grid filtering before their orientation is computed and they
     * are copied into the larger Extrema data structure.
     */
    const int slots = conf.getFilterGridSize();

    thrust::device_vector<int>   octave_index( ext_total );
    thrust::device_vector<int>   iext_index  ( ext_total );
    thrust::device_vector<int>   cell_values ( ext_total );
    thrust::device_vector<float> scale_values( ext_total );
    thrust::device_vector<int>   cell_counts ( slots * slots );
    thrust::device_vector<int>   cell_offsets( slots * slots );

    int sum = 0;
    for( int o=0; o<MAX_OCTAVES; o++ ) {
        const int ocount = hct.ext_ct[o];
        if( ocount > 0 ) {
            cudaStream_t oct_str = _octaves[o].getStream();

            // fill a continuous device array with octave of all initial extrema
            thrust::fill(     thrust::cuda::par.on(oct_str),
                              octave_index.begin() + sum,
                              octave_index.begin() + sum + ocount,
                              o );
            // fill a continuous device array with index within octave of all initial extrema
            thrust::sequence( thrust::cuda::par.on(oct_str),
                              iext_index.begin() + sum,
                              iext_index.begin() + sum + ocount );
            sum += ocount;
        }
    }

    cudaDeviceSynchronize();

    // extract cell and scale value for all initial extrema
    FunctionExtractCell          fun_extract_cell;

    thrust::transform( thrust::make_zip_iterator( thrust::make_tuple( octave_index.begin(),
                                                                      iext_index.begin() ) ),
                       thrust::make_zip_iterator( thrust::make_tuple( octave_index.end(),
                                                                      iext_index.end() ) ),
                       thrust::make_zip_iterator( thrust::make_tuple( cell_values.begin(),
                                                                      scale_values.begin() ) ),
                       fun_extract_cell );
    if( conf.getFilterSorting() == Config::LargestScaleFirst )
    {
        FunctionSort_IncCell_DecScale fun_sort;
        thrust::sort_by_key(
            thrust::make_zip_iterator( thrust::make_tuple( cell_values.begin(),
                                                           scale_values.begin() ) ),
            thrust::make_zip_iterator( thrust::make_tuple( cell_values.end(),
                                                           scale_values.end() ) ),
            thrust::make_zip_iterator( thrust::make_tuple( octave_index.begin(),
                                                           iext_index.  begin() ) ),
            fun_sort );
    }
    else if( conf.getFilterSorting() == Config::SmallestScaleFirst )
    {
        FunctionSort_IncCell_IncScale fun_sort;
        thrust::sort_by_key(
            thrust::make_zip_iterator( thrust::make_tuple( cell_values.begin(),
                                                           scale_values.begin() ) ),
            thrust::make_zip_iterator( thrust::make_tuple( cell_values.end(),
                                                           scale_values.end() ) ),
            thrust::make_zip_iterator( thrust::make_tuple( octave_index.begin(),
                                                           iext_index.  begin() ) ),
            fun_sort );
    }
    else
    {
        // sort (octave,index,scale) tuples by their cell values (in which cell are they located)
        thrust::sort_by_key(
            cell_values.begin(),
            cell_values.end(),
            thrust::make_zip_iterator( thrust::make_tuple( octave_index.begin(),
                                                           iext_index.  begin(),
                                                           scale_values.begin() ) ) );
    }

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
    thrust::host_vector<int> h_cell_offsets        ( n );
    thrust::host_vector<int> h_cell_limits         ( n );
    thrust::host_vector<int> cell_count_prefix_sums( n );
    thrust::host_vector<int> cell_count_sumup      ( n );

    // move to host code - computing the limits on the GPU is too wasteful
    h_cell_counts = cell_counts;

    // offset to beginning of each cell value - recompute faster than copy
    thrust::exclusive_scan( h_cell_counts.begin(), h_cell_counts.end(), h_cell_offsets.begin() );

    // offset to end indeces of each cell value - could shift h_cell_offsets and sum up counts
    thrust::inclusive_scan( h_cell_counts.begin(), h_cell_counts.end(), h_cell_limits .begin() );

    // the cell filter algorithm requires the cell counts in increasing order, cell_permute
    // maps new position to original index
    thrust::sequence( h_cell_permute.begin(), h_cell_permute.end() );
    thrust::sort_by_key( h_cell_counts.begin(), h_cell_counts.end(), h_cell_permute.begin() );

    // several steps to find the cells that must loose extrema

    // inclusive prefix sum
    thrust::inclusive_scan( h_cell_counts.begin(), h_cell_counts.end(), cell_count_prefix_sums.begin() );

    thrust::host_vector<int> h_reverse_index(n);
    thrust::sequence( h_reverse_index.begin(), h_reverse_index.end(),
                      n-1,
                      -1 );

    // sumup[i] = prefix sum[i] + sum( cell[i] copied into remaining cells )
    thrust::transform( h_cell_counts.begin(), h_cell_counts.end(),
                       h_reverse_index.begin(),
                       cell_count_sumup.begin(),
                       thrust::multiplies<int>() );
    thrust::transform( cell_count_sumup.begin(), cell_count_sumup.end(),
                       cell_count_prefix_sums.begin(),
                       cell_count_sumup.begin(),
                       thrust::plus<int>() );

    FunctionIsAbove function_is_above( conf.getFilterMaxExtrema() );

    // count cells that are above the extrema limit after the summing. Those must share the
    // reduction of extrema
    int ct = thrust::count_if( cell_count_sumup.begin(), cell_count_sumup.end(),
                               function_is_above );

    float tailaverage = float( thrust::reduce( &h_cell_counts[n-ct], &h_cell_counts[n] ) ) / ct;

    int   newlimit    = ::ceilf( tailaverage - ( ext_total - conf.getFilterMaxExtrema() ) / ct );

    // clamp all cells to the computed limit - the total is now less than n extrema off
    thrust::transform( h_cell_counts.begin(), h_cell_counts.end(), 
                       thrust::make_constant_iterator<int>(newlimit),
                       h_cell_counts.begin(),
                       thrust::minimum<int>() );

    // back to original order
    thrust::sort_by_key( h_cell_permute.begin(), h_cell_permute.end(), h_cell_counts.begin() );

    // transfer counts back to device
    cell_counts = h_cell_counts;

    for( int i=0; i<h_cell_counts.size(); i++ )
    {
        FunctionDisableExtremum fun_disable_extremum;

        int from = h_cell_offsets[i] + h_cell_counts[i];
        int to   = h_cell_limits [i];

        thrust::for_each(
            thrust::make_zip_iterator( thrust::make_tuple( octave_index.begin() + from,
                                                           iext_index  .begin() + from ) ),
            thrust::make_zip_iterator( thrust::make_tuple( octave_index.begin() + to,
                                                           iext_index  .begin() + to ) ),
            fun_disable_extremum );
    }

    thrust::device_vector<int>   grid( ext_total );

    int ret_ext_total = 0;

    for( int o=0; o<MAX_OCTAVES; o++ ) {
        const int ocount = hct.ext_ct[o];

        if( ocount > 0 ) {
            FunctionExtractIgnored fun_extract_ignore;
            thrust::identity<int>  fun_id;

            grid.resize( ocount );

            thrust::transform(
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(ocount),
                thrust::make_constant_iterator(o),
                grid.begin(),
                fun_extract_ignore );

            thrust::device_ptr<int> off_ptr = thrust::device_pointer_cast( dobuf_shadow.i_ext_off[o] );

            thrust::copy_if( thrust::make_counting_iterator(0),
                             thrust::make_counting_iterator(ocount),
                             grid.begin(),
                             off_ptr,
                             fun_id );

            hct.ext_ct[o] = thrust::reduce( grid.begin(), grid.end() );

            ret_ext_total += hct.ext_ct[o];
        }
    }

    nvtxRangePushA( "writing back count" );
    writeDescCountersToDevice( );
    nvtxRangePop( );

    return ret_ext_total;
}
}; // namespace popsift

#else // not defined(DISABLE_GRID_FILTER)

namespace popsift
{
/* do nothing unless we have CUDA v 8 or newer */
__host__
int Pyramid::extrema_filter_grid( const Config& conf, int ext_total )
{
    return ext_total;
}
}; // namespace popsift

#endif // not defined(DISABLE_GRID_FILTER)

