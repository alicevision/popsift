/*
 * Copyright 2021, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "filtergrid_debug.h"

#include <iostream>
#include <iomanip>

namespace popsift
{

void debug_arrays( int* sorting_index, int* cell_array, float* scale_array, InitialExtremum** ptr_array,
                   ExtremaCounters* ct,
                   int num_cells, int* samples_in_cell, int* samples_prefix_sum,
                   DebugArraysMode mode,
                   const std::string& intro )
{
std::cout << "    Enter " << __FUNCTION__ << std::endl;
std::cout << intro << std::endl;
    cudaDeviceSynchronize();

    /***************************/
    /* printing cell histogram */
    /***************************/
    if( mode == PrintHistogram )
    {
        int sum_extrema_in_octave = 0;
        for( int o=0; o<MAX_OCTAVES; o++ )
        {
            int count = ct->getExtremaCount(o);
            if( count != 0 )
            {
                std::cout << "      Extrema in octave " << o << ":"
                          << " base=" << ct->getExtremaBase(o)
                          << " count=" << count << std::endl;
                sum_extrema_in_octave += count;
            }
        }
        std::cout << "      Extrema in image " << sum_extrema_in_octave << std::endl << std::endl;

        int sum_samples_in_cells = 0;
        for( int i=0; i<num_cells; i++ )
        {
            const int s = samples_in_cell[i];
            sum_samples_in_cells += s;
            std::cout << "      Cell " << i << " samples " << s << std::endl;
        }
        std::cout << "      Total number of samples in cells: " << sum_samples_in_cells << std::endl << std::endl;
        for( int i=0; i<num_cells+1; i++ )
        {
            const int s = samples_prefix_sum[i];
            std::cout << "      Prefix sum " << i << " samples " << s << std::endl;
        }
    }

    const int extrema_ct_total = ct->getTotalExtrema();

    /**************************/
    /* printing sorting index */
    /**************************/
    if( mode == PrintHistogram )
    {
    }
    else if( mode == PrintUnsortedByOctave )
    {
        for( int o=0; o<MAX_OCTAVES; o++ )
        {
            const int extrema_ct_in_octave   = ct->getExtremaCount(o);
            const int extrema_base_in_octave = ct->getExtremaBase(o);

            if( extrema_ct_in_octave == 0 ) continue;

            std::cout << "      Indexes in octave " << o << ": ";
            for( int i=0; i<extrema_ct_in_octave; i++ )
            {
                if( i % 30 == 0 ) std::cout << std::endl <<  "        ";
                std::cout << sorting_index[extrema_base_in_octave + i] << " ";
            }
            std::cout << std::endl;
        }
    }
    else if( mode == PrintUnsortedFlat || mode == PrintSortedFlat )
    {
        std::cout << "      All indexes: ";
        for( int i=0; i<extrema_ct_total; i++ )
        {
            if( i % 30 == 0 ) std::cout << std::endl <<  "        ";
            std::cout << sorting_index[i] << " ";
        }
        std::cout << std::endl
                  << std::endl;
    }
    else
    {
        std::cout << "      Not printing sorting_index" << std::endl;
    }

    /**************************/
    /* printing cell values   */
    /**************************/
    if( mode == PrintHistogram )
    {
    }
    else if( mode == PrintUnsortedByOctave )
    {
        for( int o=0; o<MAX_OCTAVES; o++ )
        {
            const int extrema_ct_in_octave   = ct->getExtremaCount(o);
            const int extrema_base_in_octave = ct->getExtremaBase(o);

            if( extrema_ct_in_octave == 0 ) continue;

            std::cout << "      cell values in octave " << o << ": ";
            for( int i=0; i<extrema_ct_in_octave; i++ )
            {
                if( i % 30 == 0 ) std::cout << std::endl <<  "        ";
                std::cout << cell_array[extrema_base_in_octave + i] << " ";
            }
            std::cout << std::endl;
        }
    }
    else if( mode == PrintUnsortedFlat )
    {
        std::cout << "      All cell values: ";
        for( int i=0; i<extrema_ct_total; i++ )
        {
            if( i % 30 == 0 ) std::cout << std::endl <<  "        ";
            std::cout << cell_array[i] << " ";
        }
        std::cout << std::endl;
    }
    else if( mode == PrintSortedFlat )
    {
        std::cout << "      All cell values by sorting index: ";
        for( int i=0; i<extrema_ct_total; i++ )
        {
            if( i % 30 == 0 ) std::cout << std::endl <<  "        ";
            std::cout << cell_array[sorting_index[i]] << " ";
        }
        std::cout << std::endl
                  << std::endl;
    }
    else
    {
        for( int c=0; c<num_cells; c++ )
        {
            if( samples_in_cell[c] == 0 ) continue;

            std::cout << "      scale values in cell " << c << ": ";
            for( int i=0; i<samples_in_cell[c]; i++ )
            {
                if( i % 30 == 0 ) std::cout << std::endl <<  "        ";
                std::cout << cell_array[sorting_index[samples_prefix_sum[c] + i]] << " ";
            }
            std::cout << std::endl;
        }
    }

    /**************************/
    /* printing scale values  */
    /**************************/
    std::cout << std::setprecision(3);

    if( mode == PrintHistogram )
    {
    }
    else if( mode == PrintUnsortedByOctave )
    {
        for( int o=0; o<MAX_OCTAVES; o++ )
        {
            const int extrema_ct_in_octave   = ct->getExtremaCount(o);
            const int extrema_base_in_octave = ct->getExtremaBase(o);

            if( extrema_ct_in_octave == 0 ) continue;

            std::cout << "      scale values in octave " << o << ": ";
            for( int i=0; i<extrema_ct_in_octave; i++ )
            {
                if( i % 30 == 0 ) std::cout << std::endl <<  "        ";
                std::cout << scale_array[extrema_base_in_octave + i] << " ";
            }
            std::cout << std::endl;
        }
    }
    else if( mode == PrintUnsortedFlat )
    {
        std::cout << "      All scale values: ";
        for( int i=0; i<extrema_ct_total; i++ )
        {
            if( i % 30 == 0 ) std::cout << std::endl <<  "        ";
            std::cout << scale_array[i] << " ";
        }
        std::cout << std::endl;
    }
    else if( mode == PrintSortedFlat )
    {
        std::cout << "      All scale values by sorting index: ";
        for( int i=0; i<extrema_ct_total; i++ )
        {
            if( i % 30 == 0 ) std::cout << std::endl <<  "        ";
            std::cout << scale_array[sorting_index[i]] << " ";
        }
        std::cout << std::endl
                  << std::endl;
    }
    else
    {
        for( int c=0; c<num_cells; c++ )
        {
            if( samples_in_cell[c] == 0 ) continue;

            std::cout << "      scale values in cell " << c << ": ";
            for( int i=0; i<samples_in_cell[c]; i++ )
            {
                if( i % 30 == 0 ) std::cout << std::endl <<  "        ";
                std::cout << scale_array[sorting_index[samples_prefix_sum[c] + i]] << " ";
            }
            std::cout << std::endl;
        }
    }

    /**************************/
    /* printing isIgnored values  */
    /**************************/
    std::cout << std::setprecision(3);

    if( mode == PrintHistogram )
    {
    }
    else if( mode == PrintUnsortedByOctave )
    {
        for( int o=0; o<MAX_OCTAVES; o++ )
        {
            const int extrema_ct_in_octave   = ct->getExtremaCount(o);
            const int extrema_base_in_octave = ct->getExtremaBase(o);

            if( extrema_ct_in_octave == 0 ) continue;

            std::cout << "      ignore values in octave " << o << ": ";
            for( int i=0; i<extrema_ct_in_octave; i++ )
            {
                if( i % 30 == 0 ) std::cout << std::endl <<  "        ";
                std::cout << ptr_array[extrema_base_in_octave + i]->isIgnored() << " ";
            }
            std::cout << std::endl;
        }
    }
    else if( mode == PrintUnsortedFlat )
    {
        std::cout << "      All ignore values: ";
        for( int i=0; i<extrema_ct_total; i++ )
        {
            if( i % 30 == 0 ) std::cout << std::endl <<  "        ";
            std::cout << ptr_array[i]->isIgnored() << " ";
        }
        std::cout << std::endl;
    }
    else if( mode == PrintSortedFlat )
    {
        std::cout << "      All ignore values by sorting index: ";
        for( int i=0; i<extrema_ct_total; i++ )
        {
            if( i % 30 == 0 ) std::cout << std::endl <<  "        ";
            std::cout << ptr_array[sorting_index[i]]->isIgnored() << " ";
        }
        std::cout << std::endl
                  << std::endl;
    }
    else
    {
        for( int c=0; c<num_cells; c++ )
        {
            // int ct = samples_in_cell[c];
            int ct = samples_prefix_sum[c+1] - samples_prefix_sum[c];

            if( ct == 0 ) continue;

            std::cout << "      ignore values in cell " << c << " (" << ct-samples_in_cell[c] << " out of total " << ct << ", leaving " << samples_in_cell[c] << "): ";
            for( int i=0; i<ct; i++ )
            {
                if( i % 30 == 0 ) std::cout << std::endl <<  "        ";
                std::cout << ptr_array[sorting_index[samples_prefix_sum[c] + i]]->isIgnored() << " ";
            }
            std::cout << std::endl;
        }
    }
std::cout << "    Leave " << __FUNCTION__ << std::endl;
}

}; // namespace popsift

