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
std::cerr << "    Enter " << __FUNCTION__ << std::endl;
std::cerr << intro << std::endl;
    cudaDeviceSynchronize();

    /***************************/
    /* printing cell histogram */
    /***************************/
    if( mode == PrintHistogram )
    {
        int sum = 0;
        for( int i=0; i<num_cells; i++ )
        {
            const int s = samples_in_cell[i];
            sum += s;
            std::cerr << "      Cell " << i << " samples " << s << std::endl;
        }
        for( int i=0; i<num_cells+1; i++ )
        {
            const int s = samples_prefix_sum[i];
            std::cerr << "      Prefix sum " << i << " samples " << s << std::endl;
        }
        std::cerr << "      Total number of samples: " << sum << std::endl << std::endl;
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

            std::cerr << "      Indexes in octave " << o << ": ";
            for( int i=0; i<extrema_ct_in_octave; i++ )
            {
                if( i % 30 == 0 ) std::cerr << std::endl <<  "        ";
                std::cerr << sorting_index[extrema_base_in_octave + i] << " ";
            }
            std::cerr << std::endl;
        }
    }
    else if( mode == PrintUnsortedFlat || mode == PrintSortedFlat )
    {
        std::cerr << "      All indexes: ";
        for( int i=0; i<extrema_ct_total; i++ )
        {
            if( i % 30 == 0 ) std::cerr << std::endl <<  "        ";
            std::cerr << sorting_index[i] << " ";
        }
        std::cerr << std::endl
                  << std::endl;
    }
    else
    {
        std::cerr << "      Not printing sorting_index" << std::endl;
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

            std::cerr << "      cell values in octave " << o << ": ";
            for( int i=0; i<extrema_ct_in_octave; i++ )
            {
                if( i % 30 == 0 ) std::cerr << std::endl <<  "        ";
                std::cerr << cell_array[extrema_base_in_octave + i] << " ";
            }
            std::cerr << std::endl;
        }
    }
    else if( mode == PrintUnsortedFlat )
    {
        std::cerr << "      All cell values: ";
        for( int i=0; i<extrema_ct_total; i++ )
        {
            if( i % 30 == 0 ) std::cerr << std::endl <<  "        ";
            std::cerr << cell_array[i] << " ";
        }
        std::cerr << std::endl;
    }
    else if( mode == PrintSortedFlat )
    {
        std::cerr << "      All cell values by sorting index: ";
        for( int i=0; i<extrema_ct_total; i++ )
        {
            if( i % 30 == 0 ) std::cerr << std::endl <<  "        ";
            std::cerr << cell_array[sorting_index[i]] << " ";
        }
        std::cerr << std::endl
                  << std::endl;
    }
    else
    {
        for( int c=0; c<num_cells; c++ )
        {
            if( samples_in_cell[c] == 0 ) continue;

            std::cerr << "      scale values in cell " << c << ": ";
            for( int i=0; i<samples_in_cell[c]; i++ )
            {
                if( i % 30 == 0 ) std::cerr << std::endl <<  "        ";
                std::cerr << cell_array[sorting_index[samples_prefix_sum[c] + i]] << " ";
            }
            std::cerr << std::endl;
        }
    }

    /**************************/
    /* printing scale values  */
    /**************************/
    std::cerr << std::setprecision(3);

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

            std::cerr << "      scale values in octave " << o << ": ";
            for( int i=0; i<extrema_ct_in_octave; i++ )
            {
                if( i % 30 == 0 ) std::cerr << std::endl <<  "        ";
                std::cerr << scale_array[extrema_base_in_octave + i] << " ";
            }
            std::cerr << std::endl;
        }
    }
    else if( mode == PrintUnsortedFlat )
    {
        std::cerr << "      All scale values: ";
        for( int i=0; i<extrema_ct_total; i++ )
        {
            if( i % 30 == 0 ) std::cerr << std::endl <<  "        ";
            std::cerr << scale_array[i] << " ";
        }
        std::cerr << std::endl;
    }
    else if( mode == PrintSortedFlat )
    {
        std::cerr << "      All scale values by sorting index: ";
        for( int i=0; i<extrema_ct_total; i++ )
        {
            if( i % 30 == 0 ) std::cerr << std::endl <<  "        ";
            std::cerr << scale_array[sorting_index[i]] << " ";
        }
        std::cerr << std::endl
                  << std::endl;
    }
    else
    {
        for( int c=0; c<num_cells; c++ )
        {
            if( samples_in_cell[c] == 0 ) continue;

            std::cerr << "      scale values in cell " << c << ": ";
            for( int i=0; i<samples_in_cell[c]; i++ )
            {
                if( i % 30 == 0 ) std::cerr << std::endl <<  "        ";
                std::cerr << scale_array[sorting_index[samples_prefix_sum[c] + i]] << " ";
            }
            std::cerr << std::endl;
        }
    }

    /**************************/
    /* printing isIgnored values  */
    /**************************/
    std::cerr << std::setprecision(3);

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

            std::cerr << "      ignore values in octave " << o << ": ";
            for( int i=0; i<extrema_ct_in_octave; i++ )
            {
                if( i % 30 == 0 ) std::cerr << std::endl <<  "        ";
                std::cerr << ptr_array[extrema_base_in_octave + i]->isIgnored() << " ";
            }
            std::cerr << std::endl;
        }
    }
    else if( mode == PrintUnsortedFlat )
    {
        std::cerr << "      All ignore values: ";
        for( int i=0; i<extrema_ct_total; i++ )
        {
            if( i % 30 == 0 ) std::cerr << std::endl <<  "        ";
            std::cerr << ptr_array[i]->isIgnored() << " ";
        }
        std::cerr << std::endl;
    }
    else if( mode == PrintSortedFlat )
    {
        std::cerr << "      All ignore values by sorting index: ";
        for( int i=0; i<extrema_ct_total; i++ )
        {
            if( i % 30 == 0 ) std::cerr << std::endl <<  "        ";
            std::cerr << ptr_array[sorting_index[i]]->isIgnored() << " ";
        }
        std::cerr << std::endl
                  << std::endl;
    }
    else
    {
        for( int c=0; c<num_cells; c++ )
        {
            if( samples_in_cell[c] == 0 ) continue;

            std::cerr << "      ignore values in cell " << c << ": ";
            for( int i=0; i<samples_in_cell[c]; i++ )
            {
                if( i % 30 == 0 ) std::cerr << std::endl <<  "        ";
                std::cerr << ptr_array[sorting_index[samples_prefix_sum[c] + i]]->isIgnored() << " ";
            }
            std::cerr << std::endl;
        }
    }
std::cerr << "    Leave " << __FUNCTION__ << std::endl;
}

}; // namespace popsift

