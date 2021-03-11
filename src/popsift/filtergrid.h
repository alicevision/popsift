/*
 * Copyright 2021, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "sift_config.h"
#include "sift_conf.h"

#if ! POPSIFT_IS_DEFINED(POPSIFT_DISABLE_GRID_FILTER)

namespace popsift
{

class ExtremaBuffers;
class ExtremaCounters;
class InitialExtremum;

class FilterGrid
{
    int*             _histogram;
    int*             _histogram_full;
    int*             _histogram_eps; // exclusive prefix sum;

    const int        _slots;
    ExtremaBuffers*  _buf;
    ExtremaCounters* _ct;

    int               _alloc_size;
    int*              _sorting_index;
    int*              _cells;
    float*            _scales;
    InitialExtremum** _initial_extrema_pointers;

public:
    FilterGrid( const Config& conf );

    ~FilterGrid( );

    int filter( const Config& conf, ExtremaCounters* ct, ExtremaBuffers* buf, int ext_total );

private:
    /** Initialize the FilterGrid structure with the current counters
     *  and buffers.
     *  Compute the exclusive prefix sum of extrema per octave.
     */
    void init( ExtremaBuffers* buf, ExtremaCounters* ct );

    /** We are filling the _histogram_full by counting how frequent
     *  extrema appear in every cell.
     *  Since order is irrelevant for this operation, we ignore
     *  the indirection through the _sorting_index while we collect
     *  the histogram.
     */
    void count_cells( );

    /** To get the offsets for each cell into the sorted array
     *  _sorting_index, we can simple compute the exclusive prefix
     *  sum over _histogram_full.
     *  For convenience, it has an extra slot for the final sum.
     */
    void make_histogram_prefix_sum( );

    /** We sort the index of extrema (not the extrema themselves).
     *  The end result is in _sorting_index, which gives us indirect
     *  access (sorted by priority) to the arrays _cells, _scales
     *  and _initial_extrema_pointers.
     */
    void sort_by_cell_and_scale( );

    /** Compute a maximum nunmber of entries in every cell and
     *  update _histogram_full.
     *  Each cell has an equal share in the maximum number of
     *  extrema, but when a cell does not use its share fully,
     *  other cells can have it. The total number of extrema
     *  assigned can slightly exceed the given max value.
     */
    void level_histogram( const int max_extrema );

    void prune_extrema( GridFilterConfig::Mode mode );
};

}; // namespace popsift

#else // not defined(DISABLE_GRID_FILTER)

namespace popsift
{
class FilterGrid
{
public:
    FilterGrid( const Config& ) { }

    int filter( const Config&, ExtremaCounters*, ExtremaBuffers*, int ext_total ) {
        return ext_total;
    }
};
}; // namespace popsift

#endif // not defined(DISABLE_GRID_FILTER)

