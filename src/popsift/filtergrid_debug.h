/*
 * Copyright 2021, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "sift_pyramid.h"

#include <string>

namespace popsift
{

enum DebugArraysMode
{
    PrintHistogram,
    PrintUnsortedByOctave,
    PrintUnsortedFlat,
    PrintSortedFlat,
    PrintRest
};

void debug_arrays( int* sorting_index, int* cell_array, float* scale_array, InitialExtremum** ptr_array,
                   ExtremaCounters* ct,
                   int num_cells, int* samples_in_cell, int* samples_prefix_sum,
                   DebugArraysMode mode,
                   const std::string& intro );
};

