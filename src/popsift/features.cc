/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/grid.h"
#include "common/debug_macros.h"
#include "features.h"
#include "sift_extremum.h"

#include <cerrno>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <limits>
#include <cmath>

using namespace std;

namespace popsift {

/* squared distance between two descriptors (square root not taken) */
static inline float l2_in_t0( const float* lptr, const float* rptr );

/* Compute all distances between all descriptors in two descriptor arrays l and r.
 * Store the result in an array containing left index, right index, and whether
 * the two descriptors are considered accepted matches.
 */
static void compute_distance( int3* match_matrix, Descriptor* l, int l_len, Descriptor* r, int r_len );

/* Take the match matrix of compute_distance() and print the matching descriptors
 * to stdout.
 */
static void
show_distance( int3*       match_matrix,
               Feature*    l_ext, Descriptor* l_ori, int* l_fem, int l_len,
               Feature*    r_ext, Descriptor* r_ori, int* r_fem, int r_len );

/*************************************************************
 * FeaturesHost
 *************************************************************/

FeaturesHost::FeaturesHost( )
    : _num_ext( 0 )
    , _num_ori( 0 )
    , _ext( nullptr )
    , _ori( nullptr )
    , _rev( nullptr )
{ }

FeaturesHost::FeaturesHost( int num_ext, int num_ori )
    : _ext( nullptr )
    , _ori( nullptr )
    , _rev( nullptr )
{
    reset( num_ext, num_ori );
}

FeaturesHost::~FeaturesHost( )
{
    delete [] _ext;
    delete [] _ori;
    delete [] _rev;
}

void FeaturesHost::reset( int num_ext, int num_ori )
{
    delete [] _ext;
    delete [] _ori;
    delete [] _rev;

    _ext = new Feature   [num_ext];
    _ori = new Descriptor[num_ori];
    _rev = new int       [num_ori];

    setFeatureCount( num_ext );
    setDescriptorCount( num_ori );
}

void FeaturesHost::setReverseMap( const std::vector<int>& revmap )
{
    int sz = revmap.size();
    if( _num_ori != sz )
    {
        POP_FATAL( "Programming error in " << __FILE__ << ":" << __LINE__ << ": reverse map size " << sz << " must be the same as number of orientations " << _num_ori );
    }
    if( sz > 0 && _rev == nullptr )
    {
        POP_FATAL( "Programming error in " << __FILE__ << ":" << __LINE__ << ": reverse map is not alloated yet in FeaturesHost" );
    }

    for( int i=0; i<sz; i++ )
    {
        _rev[i] = revmap[i];
    }
}

void FeaturesHost::match( std::unique_ptr<FeaturesHost> other )
{
    int l_len = getDescriptorCount( );
    int r_len = other->getDescriptorCount( );

    int3* match_matrix = new int3[l_len];

    compute_distance( match_matrix, getDescriptors(), l_len, other->getDescriptors(), r_len );

    show_distance( match_matrix,
                   getFeatures(),
                   getDescriptors(),
                   getReverseMap(),
                   l_len,
                   other->getFeatures(),
                   other->getDescriptors(),
                   other->getReverseMap(),
                   r_len );

    delete [] match_matrix;
}

int3* FeaturesHost::matchAndReturn( std::unique_ptr<FeaturesHost>& other )
{
    int l_len = getDescriptorCount( );
    int r_len = other->getDescriptorCount( );

    int3* match_matrix = new int3[l_len];

    compute_distance( match_matrix, getDescriptors(), l_len, other->getDescriptors(), r_len );

    return match_matrix;
}

void FeaturesHost::freeMatches( int3* match_matrix )
{
    delete [] match_matrix;
}

Descriptor* FeaturesHost::getDescriptor( int descIndex )
{
    return &_ori[descIndex];
}

const Descriptor* FeaturesHost::getDescriptor( int descIndex ) const
{
    return &_ori[descIndex];
}

// Feature* FeaturesHost::getFeatureForDescriptor( int descIndex )
// {
//     return &_ext[_rev[descIndex]];
// }

const Feature* FeaturesHost::getFeatureForDescriptor( int descIndex ) const
{
    return &_ext[_rev[descIndex]];
}

void FeaturesHost::print( std::ostream& ostr, bool write_as_uchar, bool write_as_ori ) const
{
    for( int i=0; i<size(); i++ ) {
        _ext[i].print( ostr, write_as_uchar, write_as_ori );
    }
}

std::ostream& operator<<( std::ostream& ostr, const FeaturesHost& feature )
{
    feature.print( ostr, false, false );
    return ostr;
}

/*************************************************************
 * Feature
 *************************************************************/

void Feature::print( std::ostream& ostr, bool write_as_uchar, bool write_as_ori ) const
{
    float sigval =  1.0f / ( sigma * sigma );

    for( int ori=0; ori<num_ori; ori++ )
    {
        if( write_as_ori )
        {
            ostr << xpos << " " << ypos << " "
                 << sigma << " " << orientation[ori] << " ";
        }
        else
        {
            ostr << xpos << " " << ypos << " "
                 << sigval << " 0 " << sigval << " ";
        }
        if( write_as_uchar ) {
            for( int i=0; i<128; i++ ) {
                ostr << std::round(desc[ori]->features[i]) << " ";
            }
        } else {
            ostr << std::setprecision(3);
            for( int i=0; i<128; i++ ) {
                ostr << desc[ori]->features[i] << " ";
            }
            ostr << std::setprecision(6);
        }
        ostr << std::endl;
    }
}

std::ostream& operator<<( std::ostream& ostr, const Feature& feature )
{
    feature.print( ostr, false, false );
    return ostr;
}

/*************************************************************
 * helper function
 *************************************************************/

/* squared distance between two descriptors (square root not taken) */
static inline float l2_in_t0( const float* lptr, const float* rptr )
{
    float result = 0.0f;

    for( int i=0; i<128; i++ )
    {
        float mval = *lptr - *rptr;

        result += ( mval * mval );

        lptr++;
        rptr++;
    }

    return result;
}

static void
compute_distance( int3* match_matrix, Descriptor* l, int l_len, Descriptor* r, int r_len )
{
    for( int left_idx=0; left_idx<l_len; left_idx++ )
    {
        float match_1st_val = std::numeric_limits<float>::infinity();
        float match_2nd_val = std::numeric_limits<float>::infinity();
        int   match_1st_idx = 0;
        int   match_2nd_idx = 0;

        const float* lptr = l[left_idx].features;

        for( int right_idx=0; right_idx<r_len; right_idx++ )
        {
            const float* rptr = r[right_idx].features;

            const float  res  = l2_in_t0( lptr, rptr );

            if( res < match_1st_val )
            {
                match_2nd_val = match_1st_val;
                match_2nd_idx = match_1st_idx;
                match_1st_val = res;
                match_1st_idx = right_idx;
            }
            else if( res < match_2nd_val )
            {
                match_2nd_val = res;
                match_2nd_idx = right_idx;
            }
        }

        bool accept = ( match_1st_val / match_2nd_val < 0.8f );

        #ifdef __CUDACC__
            match_matrix[left_idx] = make_int3(match_1st_idx, match_2nd_idx, accept);
        #else
            match_matrix[left_idx] = int3(match_1st_idx, match_2nd_idx, accept);
        #endif
    }
}

static void
show_distance( int3*       match_matrix,
               Feature*    l_ext,
               Descriptor* l_ori,
               int*        l_fem,
               int         l_len,
               Feature*    r_ext,
               Descriptor* r_ori,
               int*        r_fem,
               int         r_len )
{
    for( int match_idx=0; match_idx<l_len; match_idx++ )
    {
        const float* lptr  = l_ori[match_idx].features;
        const float* rptr1 = r_ori[match_matrix[match_idx].x].features;
        const float* rptr2 = r_ori[match_matrix[match_idx].y].features;
        float d1 = l2_in_t0( lptr, rptr1 );
        float d2 = l2_in_t0( lptr, rptr2 );

        if( match_matrix[match_idx].z )
        {
            Feature* lx = &l_ext[l_fem[match_idx]];
            Feature* rx = &r_ext[r_fem[match_matrix[match_idx].x]];
            printf( "accept feat %4d [%4d] matches feat %4d [%4d] ( 2nd feat %4d [%4d] ) dist %.3f vs %.3f"
                    " (%.1f,%.1f)-(%.1f,%.1f)\n",
                    l_fem[match_idx], match_idx,
                    r_fem[match_matrix[match_idx].x], match_matrix[match_idx].x,
                    r_fem[match_matrix[match_idx].y], match_matrix[match_idx].y,
                    d1, d2,
                    lx->xpos, lx->ypos, rx->xpos, rx->ypos );
        }
        else
        {
            printf( "reject feat %4d [%4d] matches feat %4d [%4d] ( 2nd feat %4d [%4d] ) dist %.3f vs %.3f\n",
                    l_fem[match_idx], match_idx,
                    r_fem[match_matrix[match_idx].x], match_matrix[match_idx].x,
                    r_fem[match_matrix[match_idx].y], match_matrix[match_idx].y,
                    d1, d2 );
        }
    }
}

} // namespace popsift
