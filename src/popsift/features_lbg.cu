/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <limits>
#include <set>
#include <iterator>

#include <stdlib.h> // for rand()

#include <stdlib.h>
#include <errno.h>
#include <math_constants.h>

#include "features_lbg.h"
#include "sift_extremum.h"
#include "common/assist.h"
#include "common/debug_macros.h"

#define PQ_LBQ
#undef BRUTE_FORCE

using namespace std;

namespace popsift {

/* Classical Euclidean distance in 128 dimensions */
struct L2Dist
{
    float operator()( const float* l, const float* r )
    {
        float sum = 0.0f;
        for( int d=0; d<128; d++ )
        {
            float f = l[d] - r[d];
            sum += ( f * f );
        }
        sum = sqrtf( sum );
        return sum;
    }

    float operator()( const Descriptor* l, const Descriptor* r )
    {
        return this->operator()( l->features, r->features );
    }

    float operator()( const Descriptor& l, const Descriptor& r )
    {
        return this->operator()( l.features, r.features );
    }

    float operator()( const float* l, const Descriptor& r )
    {
        return this->operator()( l, r.features );
    }

    float operator()( const float* l, const Descriptor* r )
    {
        return this->operator()( l, r->features );
    }
};

#if 0
static float l2dist( const Descriptor& l, const Descriptor& r )
{
    L2Dist d;
    return d( l, r );
}

static float l2dist( const float* l, const Descriptor& r )
{
    L2Dist d;
    return d( l, r );
}
#endif

#if 0
template<int LEN>
static float subvec_l2dist( int base, const Descriptor& l, const Descriptor& r )
{
    float sum = 0.0f;
    for( int d=0; d<LEN; d++ )
    {
        float f = l.features[base+d] - r.features[base+d];
        sum += ( f * f );
    }
    sum = sqrtf( sum );
    return sum;
}

template<int LEN>
static float subvec_l2dist( int base, const float* l, const Descriptor& r )
{
    float sum = 0.0f;
    for( int d=0; d<LEN; d++ )
    {
        float f = l[d] - r.features[base+d];
        sum += ( f * f );
    }
    sum = sqrtf( sum );
    return sum;
}
#endif

/* hm_dist: the Mahalanobis distance on a 128-dimensional hypersphere
 *     with uniform point distribution on the sphere.
 * Mahalanobis distance between 2 points l and r: sqrt( (l-r)^T S^-1 (l-r) )
 *     where S is the covariance matrix
 * For uniformly distributed points on a 128-dimensional sphere, the
 *     covariance matrix S=1/128*I => S^-1 = 128*I
 *
static float hm_dist( const Descriptor& l, const Descriptor& r )
{
    float sum = 0.0f;
    for( int d=0; d<128; d++ )
    {
        float f = l.features[d] - r.features[d];
        sum += ( f * f );
    }
    sum = sqrtf( 128.0f * sum );
    return sum;
}
 *
 * While the Mahalanobis distance appears interesting for our cause, this
 * simplicity does in fact make it entirely uninteresting for our distance
 * tests that are only concerned with ordering.
 */

/* ProjDist: We know that SIFT feature vectors are normalize to a length
 *     of 1, putting them onto a 128-dimensional hypersphere. We are only
 *     sorting distances to determine best bin, an a projection can do the
 *     job just as well as a Euclidean distance.
 */
struct ProjDist
{
    float operator()( const float* l, const float* r )
    {
        float sum = 0.0f;
        for( int d=0; d<128; d++ )
        {
            float f = l[d] * r[d];
            sum += f;
        }
        return sum;
    }

    float operator()( const Descriptor& l, const Descriptor& r )
    {
        return this->operator()( l.features, r.features );
    }

    float operator()( const float* l, const Descriptor& r )
    {
        return this->operator()( l, r.features );
    }
};

#if 0
static float projdist( const Descriptor& l, const Descriptor& r )
{
    ProjDist d;
    return d( l, r );
}

static float projdist( const float* l, const Descriptor& r )
{
    ProjDist d;
    return d( l, r );
}
#endif

/*************************************************************
 * class Centroid
 *************************************************************/

class Centroid
{
    float _center[128];
    float _maxDist;

public:
    Centroid( )
    { }

    Centroid( const float* values )
    {
        init( values );
    }

    void init( const float* val )
    {
        memcpy( _center, val, 128*sizeof(float) );
        _maxDist = 0.0f;
    }

    void updateMaxDist( float f )
    {
        if( f > _maxDist ) _maxDist = f;
    }

    float getMaxDist( ) const
    {
        return _maxDist;
    }

    const float* getCenter() const
    {
        return _center;
    }

    void printCenter( ostream& ostr ) const
    {
        float len = 0.0f;
        for( int d=0; d<128; d++ )
        {
            ostr<< std::setprecision(2) << _center[d] << " ";
            len += ( _center[d] * _center[d] );
        }
        len = sqrt(len);
        ostr << "(" << len << ")";
    }

    std::set<float> _best_10_nodes;

    void raiseMaxDist( int length )
    {
        if( _best_10_nodes.size() == 0 ) return;

        if( _best_10_nodes.size() > length )
        {
            auto it = _best_10_nodes.begin();
            for( int i=0; i<length-1; i++ ) it++;
            _maxDist = *it;
        }
        else
        {
            _maxDist = *_best_10_nodes.rbegin();
        }
    }

#if 0
    void setBest10MaxDist( )
    {
        if( _best_10_nodes.size() > 10 )
        {
            auto it = _best_10_nodes.begin();
            _maxDist = *it;
        }
    }
#endif
};

/*************************************************************
 * class LindeBuzoGray
 *
 * Some sources claim that Linde-Buzo-Gray is a k-means algorithm.
 * That is not really the case. Centroids are not recomputed from
 * all nodes during refinement, but only internally refined.
 * That provides a more homogeneous sets but it does not create
 * centroids that for a Voronoi diagram.
 *
 * This class implements a true k-means algorithm as well, which
 * can be chosen by calling run(local=false) instead of run().
 *************************************************************/

class LindeBuzoGray
{
    struct Assignment
    {
        const Descriptor* desc;
        std::vector<int>  centroids;
    };

    /* terminate LBG algo after at most _max_rounds */
    const int               _max_rounds;

    /* Initial centroids must be chosen outside LBG. The original memory
     * will be modified to contain the final centroids. */
    std::vector<Centroid>&  _centroids;

    /* Each Descriptor can finally be assigned to one or several centroids */
    std::vector<Assignment> _data;

public:
    LindeBuzoGray( std::vector<Centroid>&         initialCentroids,
                   const std::vector<Descriptor>& descriptorList,
                   const int maxRounds )
        : _max_rounds( maxRounds )
        , _centroids( initialCentroids )
        , _data( descriptorList.size() )
    {
        for( int i=0; i<_data.size(); i++ )
        {
            _data[i].desc = &descriptorList[i];
        }

        std::cout << "Creating Linde-Buzo-Gray with " << _data.size() << " elements "                                   << " looping at most " << _max_rounds << " times to assign " << _centroids.size()
                  << " centroids." << std::endl;
    }

    void run( )
    {
        float totalDist = std::numeric_limits<float>::max();

        vector<int> assignmentCounter( _centroids.size(), 0 );

        for( int i=0; i<_max_rounds; i++ )
        {
            std::cout << "Loop " << i << std::endl;

            assignCentroids( assignmentCounter );
            float newDist = totalDistance<L2Dist>();
            std::cout << "Distance: " << newDist << std::endl;

            if( newDist < totalDist )
            {
                totalDist = newDist;
                newCenters( assignmentCounter );
            }
            else
            {
                /* The current centers are accepted, even if some have
                 * no members.
                 */
                break;
            }
        }
    }

    const Centroid& getCentroid( int centroidIdx ) const
    {
        return _centroids[centroidIdx];
    }

    // int getCenter( int descIdx ) const
    // {
    //     return _centerIdx[descIdx];
    // }

private:
    void assignCentroids( std::vector<int>& assignmentCounter )
    {
        for( int i=0; i<_data.size(); i++ )
        {
            _data[i].centroids.resize(1);
            _data[i].centroids[0] = -1;
        }

        // std::cout << "==== Min distances ====" << std::endl;
        for( int i=0; i<_data.size(); i++ )
        {
            float minDist = std::numeric_limits<float>::max();

            for( int j=0; j<_centroids.size(); j++ )
            {
                float d = L2Dist()( _centroids[j].getCenter(), _data[i].desc );
                if( d < minDist )
                {
                    int oldCentroid = _data[i].centroids[0];
                    minDist = d;
                    if( oldCentroid != -1 )
                        assignmentCounter[oldCentroid] -= 1;
                    assignmentCounter[j] += 1;
                    _data[i].centroids[0] = j;
                }
            }
            // std::cout << i << ":" << minDist << " ";
            // if( i%10==9) std::cout << std::endl;
        }
        // std::cout << std::endl;
    }

    template<class Dist>
    float totalDistance( )
    {
        Dist distance;

        float sum = 0.0f;
        for( int i=0; i<_data.size(); i++ )
        {
            int cIdx = _data[i].centroids[0];
            const Centroid& centroid = _centroids[cIdx];
            sum += distance( centroid.getCenter(), _data[i].desc );
        }
        return sum;
    }

    void newCenters( std::vector<int>& assignmentCounter )
    {
        size_t nop = std::count_if( assignmentCounter.begin(),
                                    assignmentCounter.end(),
                                    []( int val ) { return (val==0); } );

        std::cout << "There are " << nop << " centroids without value" << std::endl;

#if 0
        /* First we sort indices for the assignmentCounter, so that we are able
         * to split the centroids with the largest number of descriptors for
         * each centroid without any descriptors.
         */
        std::vector<int> assignmentSorter( assignmentCounter.size() );

        std::iota( assignmentSorter.begin(), assignmentSorter.end(), 0 ); // init 0..n

        std::sort( assignmentSorter.begin(),
                   assignmentSorter.end(), 
                   [&]( const int& l, const int& r )
                   {
                       int lval = assignmentCounter[l];
                       int rval = assignmentCounter[r];
                       if( lval < rval ) return true;
                       return false;
                   } );

        for( int i=0; i<assignmentCounter.size(); i++ )
        {
            int j = assignmentSorter[i];
            std::cout << "R " << i << ":" << assignmentCounter[j] << " ";
        }
        std::cout << std::endl;
        /*
         * TODO: split centers here
         */
#endif

        /* This is overcomplicated: it is float[_centroids.size()][128] */
        // std::vector<Descriptor> collection( _centroids.size() );
        //
        // for( int j=0; j<_centroids.size(); j++ )
        // {
        //     memset( collection[j].features, 0, 128*sizeof(float) );
        // }
        float collection[_centroids.size()][128];
        float maxdist[_centroids.size()];
        int   hits[_centroids.size()];
        int   total = 0;
        memset( collection, 0, 128*_centroids.size()*sizeof(float) );
        memset( maxdist, 0, _centroids.size()*sizeof(int) );
        memset( hits, 0, _centroids.size()*sizeof(int) );

        // sum all contributing vectors for this centroid;
        // no point in dividing by number of contributors, we normalize
        // anyway
        for( int i=0; i<_data.size(); i++ )
        {
            int cIdx = _data[i].centroids[0];
            for( int d=0; d<128; d++ )
            {
                collection[cIdx][d] += _data[i].desc->features[d];
            }
            maxdist[cIdx] = std::max<float>( maxdist[cIdx], L2Dist()( _centroids[cIdx].getCenter(), _data[i].desc ) );
            hits[cIdx] += 1;
            total++;
        }

        std::cout << "==== Number of members per centroid (total=" << total << ") ====" << std::endl;
        for( int j=0; j<_centroids.size(); j++ )
        {
            std::cout << j << ":" << hits[j] << " ";
            if( j%10==9 ) std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "==== Max dist per centroid before re-centering ====" << std::endl;
        for( int j=0; j<_centroids.size(); j++ )
        {
            std::cout << j << ":" << maxdist[j] << " ";
            if( j%10==9 ) std::cout << std::endl;
        }
        std::cout << std::endl;

        // normalize onto unit hypersphere and set new center
        // std::cout << "==== Printing before-after centroids ====" << std::endl;
        for( int j=0; j<_centroids.size(); j++ )
        {
            float sum = 0.0f;
            for( int d=0; d<128; d++ )
            {
                float v = collection[j][d];
                sum += ( v * v );
            }
            sum = sqrtf( sum );
            for( int d=0; d<128; d++ )
            {
                collection[j][d] /= sum;
            }

            // std::cout << j << "." << std::endl;
            // _centroids[j].printCenter( std::cout );
            // std::cout<< std::endl;

            _centroids[j].init( collection[j] );

            // _centroids[j].printCenter( std::cout );
            // std::cout<< std::endl;
        }

        // DEBUG BEGIN: new max dist after re-centering
        memset( maxdist, 0, _centroids.size()*sizeof(int) );

        for( int i=0; i<_data.size(); i++ )
        {
            int cIdx = _data[i].centroids[0];
            maxdist[cIdx] = std::max<float>( maxdist[cIdx], L2Dist()( _centroids[cIdx].getCenter(), _data[i].desc ) );
        }
        std::cout << "==== Max dist per centroid after re-centering ====" << std::endl;
        for( int j=0; j<_centroids.size(); j++ )
        {
            std::cout << j << ":" << maxdist[j] << " ";
            if( j%10==9 ) std::cout << std::endl;
        }
        std::cout << std::endl;
        // DEBUG END: new max dist after re-centering

        for( int i=0; i<_data.size(); i++ )
        {
            int cIdx = _data[i].centroids[0];
            float v = L2Dist()( _centroids[cIdx].getCenter(), _data[i].desc );
            _centroids[cIdx].updateMaxDist( v );
        }

        for( int j=0; j<_centroids.size(); j++ )
        {
            _centroids[j]._best_10_nodes.clear();
        }

        for( int i=0; i<_data.size(); i++ )
        {
            int cIdx = _data[i].centroids[0];
            for( int j=0; j<_centroids.size(); j++ )
            {
                if( j != cIdx )
                {
                    float v = L2Dist()( _centroids[j].getCenter(), _data[i].desc );
                    if( v > _centroids[j].getMaxDist() )
                    {
                        _centroids[j]._best_10_nodes.insert( v );
                    }
                }
            }
        }

        for( int j=0; j<_centroids.size(); j++ )
        {
            _centroids[j].raiseMaxDist(10);
        }

        for( int j=0; j<_centroids.size(); j++ )
        {
            for( int i=0; i<_data.size(); i++ )
            {
                int cIdx = _data[i].centroids[0];
                if( cIdx != j )
                {
                    float v = L2Dist()( _centroids[j].getCenter(), _data[i].desc );
                    if( v <= _centroids[j].getMaxDist() )
                    {
                        _data[i].centroids.push_back(j);
                    }
                }
            }
        }

        std::cout << "Bins: ";
        for( int i=0; i<_data.size(); i++ )
        {
            std::cout << i << ":" << _data[i].centroids.size() << " ";
            if( i%10==9 ) std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void debugPrintCentroids( ostream& ostr, int round )
    {
        int num = 1 << round;
        for( int i=0; i<num; i++ )
        {
            ostr<< i << ": ";
            _centroids[i].printCenter( ostr );
            ostr<< std::endl;
        }
    }
};

struct Level
{
    vector<Centroid> centroids;
    vector<Descriptor> desc; // copied from caller
    LindeBuzoGray* lbg;

    Level( )
        : lbg( 0 )
    { }

    Level( const std::vector<Descriptor*>& descriptorList, int rounds )
        : desc( descriptorList.size() )
    {
        for( int i=0; i<desc.size(); i++ )
        {
            for( int d=0; d<128; d++ )
            {
                float v = descriptorList[i]->features[d];
                desc[i].features[d] = v;
            }
        }

        makeLbg( rounds );
    }

    ~Level( )
    {
        delete lbg;
    }

    void makeLbg( int rounds )
    {
        int len = std::min<int>( 64, desc.size() );
        centroids.resize( len );

        for( int i=0; i<len; i++ )
        {
            // float sum = 0.0f;
            // for( int d=0; d<128; d++ )
            // {
            //     float v = desc[i]->features[d];
            //     sum += ( v * v );
            // }
            // std::cout << "Len: " << sum << std::endl;

            centroids[i].init( desc[i].features );
        }

        lbg = new LindeBuzoGray( centroids, desc, rounds );
    }
};

/*************************************************************
 * class PqtAnn
 * Inspired by the paper 10.1109/CVPR.2016.223
 *************************************************************/

PqtAnn::PqtAnn( const std::vector<Descriptor*>& descriptorList )
    : _level1( new Level( descriptorList, 100 ) )
{ }

PqtAnn::~PqtAnn( )
{
    delete _level1;
}

void PqtAnn::run( )
{
    _level1->lbg->run( );
}

void PqtAnn::findMatch( const Descriptor& desc )
{
    // int   idx1, idx2;
    // float val1, val2;

#ifdef BRUTE_FORCE
    float minval = std::numeric_limits<float>::max();
    for( int x=0; x<_level1->desc.size(); x++ )
    {
        float f = l2dist( desc, *_level1->desc[x] );
        if( f < minval ) minval = f;
        if( f == 0.0f ) break;
    }

    std::cout << "min: " << std::setprecision(3) << minval << " ";
#endif // BRUTE_FORCE

    // _level1->lbg->findBestMatches( desc, idx1, val1, idx2, val2 );

#ifdef BRUTE_FORCE
    if( *themins.begin() == minval ) std::cout << "+++ "; else std::cout << "--- ";
#endif
}

} // namespace popsift

