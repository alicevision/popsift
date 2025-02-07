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

/*************************************************************
 * FeaturesBase
 *************************************************************/

FeaturesBase::FeaturesBase( )
    : _num_ext( 0 )
    , _num_ori( 0 )
{ }

FeaturesBase::~FeaturesBase( ) = default;

/*************************************************************
 * FeaturesHost
 *************************************************************/

FeaturesHost::FeaturesHost( )
    : _ext( nullptr )
    , _ori( nullptr )
{ }

FeaturesHost::FeaturesHost( int num_ext, int num_ori )
    : _ext( nullptr )
    , _ori( nullptr )
{
    reset( num_ext, num_ori );
}

FeaturesHost::~FeaturesHost( )
{
    delete [] _ext;
    delete [] _ori;
}

void FeaturesHost::reset( int num_ext, int num_ori )
{
    delete [] _ext;
    delete [] _ori;

    _ext = new Feature   [num_ext];
    _ori = new Descriptor[num_ori];

    setFeatureCount( num_ext );
    setDescriptorCount( num_ori );
}

void FeaturesHost::pin( )
{
    /*
     * pin _ext and _ori for efficient CPU-GPU transfer.
     */
}

void FeaturesHost::unpin( )
{
    /*
     * unpin _ext and _ori after efficient CPU-GPU transfer.
     */
}

void FeaturesHost::print( std::ostream& ostr, bool write_as_uchar ) const
{
    for( int i=0; i<size(); i++ ) {
        _ext[i].print( ostr, write_as_uchar );
    }
}

std::ostream& operator<<( std::ostream& ostr, const FeaturesHost& feature )
{
    feature.print( ostr, false );
    return ostr;
}

/*************************************************************
 * FeaturesDev
 *************************************************************/

FeaturesDev::FeaturesDev( )
    : _ext( nullptr )
    , _ori( nullptr )
    , _rev( nullptr )
{ }

FeaturesDev::FeaturesDev( int num_ext, int num_ori )
    : _ext( nullptr )
    , _ori( nullptr )
    , _rev( nullptr )
{
    reset( num_ext, num_ori );
}

FeaturesDev::~FeaturesDev( )
{
    delete [] _ext;
    delete [] _ori;
    delete [] _rev;
}

void FeaturesDev::reset( int num_ext, int num_ori )
{
    if( _ext != nullptr ) { delete [] _ext; _ext = nullptr; }
    if( _ori != nullptr ) { delete [] _ori; _ori = nullptr; }
    if( _rev != nullptr ) { delete [] _rev; _rev = nullptr; }

    _ext = new Feature   [num_ext];
    _ori = new Descriptor[num_ori];
    _rev = new int       [num_ori];

    setFeatureCount( num_ext );
    setDescriptorCount( num_ori );
}

inline float
l2_in_t0( const float* lptr, const float* rptr )
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

void
compute_distance( Grid& g, int3* match_matrix, Descriptor* l, int l_len, Descriptor* r, int r_len )
{
    g.resetBlock();
    do
    {
        if( g.blockIdx.x >= l_len ) continue;

        const int idx = g.blockIdx.x;

        float match_1st_val = std::numeric_limits<float>::infinity();
        float match_2nd_val = std::numeric_limits<float>::infinity();
        int   match_1st_idx = 0;
        int   match_2nd_idx = 0;

        const float* lptr = l[idx].features;

        for( int i=0; i<r_len; i++ )
        {
            const float* rptr = r[i].features;

            const float  res  = l2_in_t0( lptr, rptr );

            if( res < match_1st_val )
            {
                match_2nd_val = match_1st_val;
                match_2nd_idx = match_1st_idx;
                match_1st_val = res;
                match_1st_idx = i;
            }
            else if( res < match_2nd_val )
            {
                match_2nd_val = res;
                match_2nd_idx = i;
            }
        }

        bool accept = ( match_1st_val / match_2nd_val < 0.8f );

        match_matrix[g.blockIdx.x] = int3( match_1st_idx, match_2nd_idx, accept );
    }
    while( g.nextBlock() );
}

void
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
    for( int i=0; i<l_len; i++ )
    {
        const float* lptr  = l_ori[i].features;
        const float* rptr1 = r_ori[match_matrix[i].x].features;
        const float* rptr2 = r_ori[match_matrix[i].y].features;
        float d1 = l2_in_t0( lptr, rptr1 );
        float d2 = l2_in_t0( lptr, rptr2 );

        if( match_matrix[i].z )
        {
            Feature* lx = &l_ext[l_fem[i]];
            Feature* rx = &r_ext[r_fem[match_matrix[i].x]];
            printf( "accept feat %4d [%4d] matches feat %4d [%4d] ( 2nd feat %4d [%4d] ) dist %.3f vs %.3f"
                    " (%.1f,%.1f)-(%.1f,%.1f)\n",
                    l_fem[i], i,
                    r_fem[match_matrix[i].x], match_matrix[i].x,
                    r_fem[match_matrix[i].y], match_matrix[i].y,
                    d1, d2,
                    lx->xpos, lx->ypos, rx->xpos, rx->ypos );
        }
        else
        {
            printf( "reject feat %4d [%4d] matches feat %4d [%4d] ( 2nd feat %4d [%4d] ) dist %.3f vs %.3f\n",
                    l_fem[i], i,
                    r_fem[match_matrix[i].x], match_matrix[i].x,
                    r_fem[match_matrix[i].y], match_matrix[i].y,
                    d1, d2 );
        }
    }
}

void FeaturesDev::match( FeaturesDev* other )
{
    int l_len = getDescriptorCount( );
    int r_len = other->getDescriptorCount( );

    int3* match_matrix = new int3[l_len];

    Grid g;
    g.setGridDim( l_len, 1, 1 );
    g.setBlockDim( 32, 1, 1 );

    compute_distance( g, match_matrix, getDescriptors(), l_len, other->getDescriptors(), r_len );

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

int3* FeaturesDev::matchAndReturn( FeaturesDev* other )
{
    int l_len = getDescriptorCount( );
    int r_len = other->getDescriptorCount( );

    int3* match_matrix = new int3[l_len];

    Grid g;
    g.setGridDim( l_len, 1, 1 );
    g.setBlockDim( 32, 1, 1 );

    compute_distance( g, match_matrix, getDescriptors(), l_len, other->getDescriptors(), r_len );

    return match_matrix;
}

void FeaturesDev::freeMatches( int3* match_matrix )
{
    delete [] match_matrix;
}

Descriptor* FeaturesDev::getDescriptor( int descIndex )
{
    return &_ori[descIndex];
}

const Descriptor* FeaturesDev::getDescriptor( int descIndex ) const
{
    return &_ori[descIndex];
}

Feature* FeaturesDev::getFeatureForDescriptor( int descIndex )
{
    return &_ext[_rev[descIndex]];
}

const Feature* FeaturesDev::getFeatureForDescriptor( int descIndex ) const
{
    return &_ext[_rev[descIndex]];
}

/*************************************************************
 * Feature
 *************************************************************/

void Feature::print( std::ostream& ostr, bool write_as_uchar ) const
{
    float sigval =  1.0f / ( sigma * sigma );

    for( int ori=0; ori<num_ori; ori++ ) {
        ostr << xpos << " " << ypos << " "
             << sigval << " 0 " << sigval << " ";
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
    feature.print( ostr, false );
    return ostr;
}

} // namespace popsift
