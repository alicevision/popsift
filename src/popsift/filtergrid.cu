/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "filtergrid.h"

#include "sift_config.h"
#include "sift_extremum.h"
#include "sift_pyramid.h"
#include "common/debug_macros.h"
#include "common/assist.h"

#if ! POPSIFT_IS_DEFINED(POPSIFT_DISABLE_GRID_FILTER)

#include <thrust/device_vector.h>
#include <thrust/sort.h>

namespace popsift
{

FilterGrid::FilterGrid( const Config& conf )
    : _slots( conf.getFilterGridSize() * conf.getFilterGridSize() )
    , _alloc_size( 0 )
    , _sorting_index( 0 )
    , _cells( 0 )
    , _scales( 0 )
    , _initial_extrema_pointers( 0 )
{
    if( _slots > 1 )
    {
        /* refresh the exclsuive prefix sum over the per-octave extrema counts */
        _histogram      = popsift::cuda::malloc_mgdT<int>( MAX_OCTAVES*_slots, __FILE__, __LINE__);
        _histogram_full = popsift::cuda::malloc_mgdT<int>( _slots, __FILE__, __LINE__);
        _histogram_eps  = popsift::cuda::malloc_mgdT<int>( _slots+1, __FILE__, __LINE__);

        // initialize histogram to zero, implicitly move to device
        popcuda_memset_sync( _histogram, 0, MAX_OCTAVES * _slots * sizeof(int) );
        popcuda_memset_sync( _histogram_full, 0, _slots * sizeof(int) );
    }
}

FilterGrid::~FilterGrid( )
{
    if( _slots > 1 )
    {
        cudaFree( _histogram_eps );
        cudaFree( _histogram_full );
        cudaFree( _histogram );
        cudaFree( _sorting_index );
        cudaFree( _cells );
        cudaFree( _scales );
        cudaFree( _initial_extrema_pointers );
    }
}

__global__ void
fg_init( const int         octave,
         ExtremaCounters*  ct,
         ExtremaBuffers*   buf,
         int*              indices,
         int*              cells,
         float*            scales,
         InitialExtremum** ext_compact )
{
    const int my_idx_in_octave  = blockIdx.x * blockDim.x + threadIdx.x;
    const int extrema_in_octave = ct->ext_ct[octave];

    if( my_idx_in_octave > extrema_in_octave ) return;

    const int base_index = ct->getExtremaBase( octave );
    const int my_idx     = base_index + my_idx_in_octave;

    /* This is the array we are going to use for sorting and filtering and so on. */
    indices[my_idx]     = my_idx;

    InitialExtremum* ie = &buf->i_ext_dat[octave][my_idx_in_octave];

    cells[my_idx]  = ie->getCell();
    scales[my_idx] = ie->getScale();

    /* Keep pointers in this pointer array. The indirection is not elegant for CUDA,
     * we can improve that later by making additional arrays for .cell and .scale
     * values.
     */
    ext_compact[my_idx] = ie;
}

void FilterGrid::init( ExtremaBuffers* buf, ExtremaCounters* ct )
{
    _buf = buf;
    _ct  = ct;
    _ct->make_extrema_prefix_sums();


    if( _alloc_size < _ct->getTotalExtrema() )
    {
        if( _alloc_size > 0 )
        {
            cudaFree( _sorting_index );
            cudaFree( _cells );
            cudaFree( _scales );
            cudaFree( _initial_extrema_pointers );
        }
        _alloc_size = _ct->getTotalExtrema();
        _sorting_index = popsift::cuda::malloc_mgdT<int>( _alloc_size, __FILE__, __LINE__);
        _cells         = popsift::cuda::malloc_mgdT<int>( _alloc_size, __FILE__, __LINE__);
        _scales        = popsift::cuda::malloc_mgdT<float>( _alloc_size, __FILE__, __LINE__);
        _initial_extrema_pointers = popsift::cuda::malloc_mgdT<InitialExtremum*>( _alloc_size, __FILE__, __LINE__);
    }

    popcuda_memset_sync( _histogram_full, 0, _slots * sizeof(int) );

    for( int o=0; o<MAX_OCTAVES; o++ )
    {
        if( _ct->ext_ct[o] == 0 ) continue;

        dim3 block( 32 );
        dim3 grid( grid_divide( _ct->ext_ct[o], block.x ) );
        fg_init
            <<<grid,block>>>
            ( o,
              _ct,
              _buf,
              _sorting_index,
              _cells,
              _scales,
              _initial_extrema_pointers );
    }
}

__global__ void
fg_countcells( const int  ext_total,
               const int* cells,
               const int  cell_histogram_size,
               int*       cell_histogram )
{
    // The size of this chared memory region is computed from _slots in the
    // calling host code.
    extern __shared__ int cellcounts[];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx >= ext_total ) return;

    const int cell = cells[idx];

    atomicAdd( &cellcounts[cell], 1 );
    __syncthreads();

    for( int c=tid; c<cell_histogram_size; c+=blockDim.x )
    {
        atomicAdd( &cell_histogram[c], cellcounts[c] );
    }
}

void FilterGrid::count_cells( )
{
    dim3 block( 32 );
    dim3 grid( grid_divide( _ct->getTotalExtrema(), block.x ) );

    fg_countcells
        <<<grid,block,sizeof(int)*_slots>>>
        ( _ct->getTotalExtrema(),
          _cells,
          _slots,
          _histogram_full );
}

void FilterGrid::make_histogram_prefix_sum( )
{
    // We could to this on the device !
    cudaDeviceSynchronize();

    _histogram_eps[0] = 0;

    for( int i=1; i<=_slots; i++ )
    {
        _histogram_eps[i] = _histogram_eps[i-1] + _histogram_full[i-1];
    }
}


class CellScaleCompare
{
    int*   _cells;
    float* _scales;

public:
    CellScaleCompare( int* cell, float* scales )
        : _cells( cell )
        , _scales( scales )
    { }

    __host__ __device__
    inline bool operator()( int left, int right ) const
    {
        if( _cells[left] < _cells[right] ) return true;
        if( ( _cells[left] == _cells[right] ) &&
            ( _scales[left] < _scales[right] ) ) return true;
        return false;
    }
};

void FilterGrid::sort_by_cell_and_scale( )
{
    /* Thanks to managed memory, we can sort our indeces either on the
     * host or on the device. Currently, we must use thrust for compiling
     * for the device. With the HPC SDK, we will be able to use OpenACC
     * and never know the difference.
     */
    CellScaleCompare tc( _cells, _scales );
#if 0
    cudaDeviceSynchronize();
    int* ptr = _sorting_index;
    std::sort( ptr, ptr + _ct->getTotalExtrema(), tc );
#else
    thrust::device_ptr<int> ptr = thrust::device_pointer_cast( _sorting_index );
    thrust::sort( ptr, ptr + _ct->getTotalExtrema(), tc );
#endif
}

void FilterGrid::level_histogram( const int max_extrema )
{
    int  avg_in_cell = max_extrema / _slots;

    bool stable;
    do
    {
        stable = true;

        int limited_slots = 0;
        int sum = 0;

        for( int i=0; i<_slots; i++ )
        {
            if( _histogram_full[i] > avg_in_cell )
            {
                sum += avg_in_cell;
                limited_slots += 1;
            }
            else
            {
                sum += _histogram_full[i];
            }
        }

        const int diff = max_extrema - sum;

        if( limited_slots > 0 && diff > 0 )
        {
            stable = false;
            const std::div_t res   = std::div( diff, limited_slots );
            const int        widen = res.quot + ( res.rem > 0 ? 1 : 0 );


            avg_in_cell += widen;
        }
    }
    while( !stable );


    for( int i=0; i<_slots; i++ )
    {
        _histogram_full[i] = min( _histogram_full[i], avg_in_cell );
    }
}

void FilterGrid::prune_extrema( GridFilterConfig::Mode mode )
{
    for( int cell=0; cell<_slots; cell++ )
    {
        const int base = _histogram_eps[cell];
        const int have = _histogram_eps[cell+1] - base;
        const int want = _histogram_full[cell];
        const int diff = have - want;
        if( diff <= 0 )
        {
            continue;
        }

        if( mode == GridFilterConfig::LargestScaleFirst )
        {
            const int base = _histogram_eps[cell];
            for( int d=0; d<diff; d++ )
            {
                const int sorted_index = _sorting_index[base+d];
                InitialExtremum* ie = _initial_extrema_pointers[sorted_index];
                ie->setIgnore();
            }
        }
        else // mode == GridFilterConfig::SmallestScaleFirst
        {
            const int base = _histogram_eps[cell+1];
            for( int d=0; d<diff; d++ )
            {
                const int sorted_index = _sorting_index[base-d-1];
                InitialExtremum* ie = _initial_extrema_pointers[sorted_index];
                ie->setIgnore();
            }
        }
    }
}

/* Discard some extrema that exceed a conf.getFilterMaxExtrema().
 * The caller should ensure that filtering is actually needed. */
__host__
int FilterGrid::filter( const Config& conf, ExtremaCounters* ct, ExtremaBuffers* buf, int ext_total )
{
    /* At this time, we have host-side information about ext_ct[o], the number
     * of extrema we have found in octave o, and we have summed it up on the
     * host size. However, other values in the hct and dct data structures
     * have not been computed yet.
     * The extrema are only known in the InitialExtrema structure. We want to
     * perform grid filtering before their orientation is computed and they
     * are copied into the larger Extrema data structure.
     */
    const int max_extrema = conf.getFilterMaxExtrema( );

    init( buf, ct );
    count_cells( );
    make_histogram_prefix_sum( );
    sort_by_cell_and_scale( );
    level_histogram( max_extrema );
    prune_extrema( conf.getFilterSorting() );

    for( int o=0; o<MAX_OCTAVES; o++ )
    {
        const int max_ct = ct->ext_ct[o];
        int counter = 0;
        for( int i=0; i<max_ct; i++ )
        {
            const InitialExtremum* iext= &buf->i_ext_dat[o][i];
            if( iext->isIgnored() == false )
            {
                buf->i_ext_off[o][counter] = i;
                counter += 1;
            }
        }
        ct->ext_ct[o] = counter;
    }

    ct->make_extrema_prefix_sums();

    return ct->getTotalExtrema();
}
}; // namespace popsift

#endif // not defined(DISABLE_GRID_FILTER)

