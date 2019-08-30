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

#include <stdlib.h>
#include <errno.h>
#include <math_constants.h>

#include "features.h"
#include "sift_extremum.h"
#include "common/assist.h"
#include "common/debug_macros.h"

#undef KFINDER
#define KFINDER_SMOOTH_8
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

    float operator()( const Descriptor& l, const Descriptor& r )
    {
        return this->operator()( l.features, r.features );
    }

    float operator()( const float* l, const Descriptor& r )
    {
        return this->operator()( l, r.features );
    }
};

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

/*************************************************************
 * class Centroid
 *************************************************************/

template<int LEN>
class Centroid
{
    int   _base;
    float _center[LEN];
    float _variance[LEN];
    int   _count;
    int   _memberCount;

public:
    Centroid( ) = delete;

    Centroid( int base )
        : _base( base )
    {
        memset( _center, 0, sizeof(float)*LEN );
    }

    void reset()
    {
        memset( _variance, 0, sizeof(float)*LEN );
        _count = 0;
    }

    void resetMemberCount()
    {
        _memberCount = 0;
    }

    void incMemberCount()
    {
        _memberCount++;
    }

    int getMemberCount() const
    {
        return _memberCount;
    }

    const float* getCenter() const
    {
        return _center;
    }

    void addToCenter( const Descriptor* data )
    {
        for( int d=0; d<LEN; d++ )
        {
            _center[d] += data->features[_base+d];
        }
        _count += 1;
    }

    void normalizeCenter( )
    {
        if( _count > 1 )
            for( int d=0; d<LEN; d++ )
                _center[d] /= _count;
        _count = 0;
    }

    void addToVariance( const Descriptor* data )
    {
        for(int d=0; d<LEN; d++)
        {
            _variance[d] += fabsf( _center[d] - data->features[_base+d] );
        }
        _count += 1;
    }

    void normalizeVariance( )
    {
        if( _count > 1 )
            for( int d=0; d<LEN; d++ )
                _variance[d] /= _count;
        _count = 0;
    }

    void addVariance( const Centroid& c )
    {
        for( int d=0; d<LEN; d++ )
            _center[d] = c._center[d] + c._variance[d];
    }

    void subVariance( const Centroid& c )
    {
        for( int d=0; d<LEN; d++ )
            _center[d] = c._center[d] - c._variance[d];
    }

    void printCenter( ostream& ostr ) const
    {
        for( int d=0; d<LEN; d++ )
        {
            ostr<< std::setprecision(2) << _center[d] << " ";
        }
    }
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

template<int LEN>
class LindeBuzoGray
{
    const int                       _base;  // base+LEN = 128

    const int                       _rounds;

    std::vector<Centroid<LEN>>      _centroids;

    const std::vector<Descriptor*>& _data;
    const int                       _data_len;
    std::vector<int>                _centerIdx;

public:
    LindeBuzoGray( int base,
                   const std::vector<Descriptor*>& descriptorList, int powerOf2 )
        : _base( base )
        , _rounds( powerOf2 )
        , _centroids( 1 << _rounds, Centroid<LEN>(0) )
        , _data( descriptorList )
        , _data_len( descriptorList.size() )
        , _centerIdx( _data_len, 0 )
    {
        std::cout << "Creating Linde-Buzo-Gray with " << _data_len << " elements for " << _rounds << " rounds (" << _centroids.size() << " centroids), " << LEN << " dims, base=" << _base << std::endl;
    }

    void run( bool local )
    {
        initCenterIdx();
        for( int r=0; r<_rounds; r++ )
        {
            computeCentroids();
            newCenters( r );
            if( local )
                findNewCentroidLocal( );
            else
                findNewCentroidGlobal( r+1 );

            // debugPrintCentroids( std::cout, r+1 );

        }
        debugClosestPointCount( std::cout, _rounds );
    }

    const Centroid<LEN>& getCentroid( int centroidIdx ) const
    {
        return _centroids[centroidIdx];
    }

    int getCenter( int descIdx ) const
    {
        return _centerIdx[descIdx];
    }

    void findBestMatches( const Descriptor& desc, int& idx1, float& val1, int& idx2, float& val2 )
    {
        idx1 = idx2 = -1;
        val1 = val2 = std::numeric_limits<float>::max();

        for( int i=0; i<(1<<_rounds); i++ )
        {
            float f = subvec_l2dist<LEN>( _base, _centroids[i].getCenter(), desc );
            if( f < val1 )
            {
                val2 = val1;
                val1 = f;
                idx2 = idx1;
                idx1 = i;
            }
            else if( f < val2 )
            {
                val2 = f;
                idx2 = i;
            }
        }
    }

    void findBestDescriptor( const Descriptor& desc, int idx,
                             Descriptor*& d1, float& val1,
                             Descriptor*& d2, float& val2 )
    {
        d1 = d2 = 0;
        val1 = val2 = std::numeric_limits<float>::max();

        for( int i=0; i<_data_len; i++ )
        {
            if( _centerIdx[i] == idx )
            {
                float f = subvec_l2dist<LEN>( _base, desc, *_data[i] );
                if( f < val1 )
                {
                    val2 = val1;
                    val1 = f;
                    d2   = d1;
                    d1   = _data[i];
                }
                else
                {
                    val2 = f;
                    d2   = _data[i];
                }
            }
        }
    }

private:
    void initCenterIdx()
    {
        for(auto& c : _centerIdx) c = 0;
    }

    void computeCentroids( )
    {
        for(auto& c : _centroids) c.reset();

        for( int i=0; i<_data_len; i++)
        {
            int ctr = _centerIdx[i];
            _centroids[ctr].addToCenter( _data[i] );
        }

        for(auto& c : _centroids) c.normalizeCenter();

        for( int i=0; i<_data_len; i++)
        {
            int ctr = _centerIdx[i];
            _centroids[ctr].addToVariance( _data[i] );
        }

        for(auto& c : _centroids) c.normalizeVariance();
    }

    void newCenters( int round )
    {
        int last = 1 << round;

        for( int i=last-1; i>=0; i-- )
        {
            _centroids[i*2+1].addVariance( _centroids[i] );
            _centroids[i*2+0].subVariance( _centroids[i] );

            std::cerr << i*2+0 << ". ";
            _centroids[i*2+0].printCenter( std::cerr );
            std::cerr << std::endl;
            std::cerr << i*2+1 << ". ";
            _centroids[i*2+1].printCenter( std::cerr );
            std::cerr << std::endl;
        }
    }

    void findNewCentroidLocal( )
    {
        for( int i=0; i<_data_len; i++)
        {
            int ctr = _centerIdx[i];
            _centroids[ctr*2+0].resetMemberCount();
            _centroids[ctr*2+1].resetMemberCount();
            const float dist0 = subvec_l2dist<LEN>( _base, _centroids[ctr*2+0].getCenter(), *_data[i] );
            const float dist1 = subvec_l2dist<LEN>( _base, _centroids[ctr*2+1].getCenter(), *_data[i] );
            const int   centr = dist0 < dist1
                              ? ctr*2+0
                              : ctr*2+1;
            _centerIdx[i] = centr;
            _centroids[centr].incMemberCount();
        }
    }

    void findNewCentroidGlobal( int round )
    {
        int num = 1 << round;
        std::vector<float> dist( num );

        for( int c=0; c<num; c++ )
        {
            _centroids[c].resetMemberCount();
        }

        for( int i=0; i<_data_len; i++)
        {
            for( int c=0; c<num; c++ )
            {
                dist[c] = subvec_l2dist<LEN>( _base, _centroids[c].getCenter(), *_data[i] );
            }
            const int centr = std::min_element( dist.begin(), dist.end() ) - dist.begin();
            _centerIdx[i] = centr;
            _centroids[centr].incMemberCount();
        }
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

    void debugClosestPointCount( ostream& ostr, int round )
    {
        int num = 1 << round;
        for( int i=0; i<num; i++ )
        {
            ostr<< i << ": " << std::count( _centerIdx.begin(), _centerIdx.end(), i ) << std::endl;
        }
    }
};

/*************************************************************
 * class PqtAnn
 * Inspired by the paper 10.1109/CVPR.2016.223
 *************************************************************/

class PqtAnn
{
    static const int level1_rounds   = 4;
    static const int level1_clusters = 1 << level1_rounds;
    static const int level2_rounds   = 5;
    static const int level2_clusters = 1 << level1_rounds;

    struct Level
    {
        vector<Descriptor*> desc; // copied from caller
        LindeBuzoGray<128>* lbg;

        Level( )
            : lbg( 0 )
        { }

        Level( const std::vector<Descriptor*>& descriptorList, int rounds )
            : desc( descriptorList )
        {
            lbg = new LindeBuzoGray<128>( 0, desc, rounds );
        }

        ~Level( )
        {
            delete lbg;
        }

        void makeLbg( int rounds )
        {
            lbg = new LindeBuzoGray<128>( 0, desc, rounds );
        }
    };

    Level              _level1;
    std::vector<Level> _level2;
public:
    PqtAnn( const std::vector<Descriptor*>& descriptorList )
        : _level1( descriptorList, level1_rounds )
    { }

    void run( )
    {
        _level1.lbg->run( false );

        _level2.resize( level1_clusters );

        for( int lvl1ctr=0; lvl1ctr<level1_clusters; lvl1ctr++ )
        {
            const int memCt = _level1.lbg->getCentroid(lvl1ctr).getMemberCount();
            _level2[lvl1ctr].desc.reserve( memCt );
        }

        for( int descIdx=0; descIdx<_level1.desc.size(); descIdx++ )
        {
            int centroidIdx = _level1.lbg->getCenter( descIdx );
            _level2[centroidIdx].desc.push_back( _level1.desc[descIdx] );
        }

        for( int lvl1ctr=0; lvl1ctr<level1_clusters; lvl1ctr++ )
        {
            _level2[lvl1ctr].makeLbg( level2_rounds );
            _level2[lvl1ctr].lbg->run( false );
        }
    }

    void findMatch( const Descriptor& desc )
    {
        int   idx1, idx2;
        float val1, val2;
        Descriptor* d5;
        Descriptor* d6;

#ifdef BRUTE_FORCE
        float minval = std::numeric_limits<float>::max();
        for( int x=0; x<_level1.desc.size(); x++ )
        {
            float f = l2dist( desc, *_level1.desc[x] );
            if( f < minval ) minval = f;
            if( f == 0.0f ) break;
        }

        std::cout << "min: " << std::setprecision(3) << minval << " ";
#endif // BRUTE_FORCE

#if 1
        std::set<float> themins;
#endif

        _level1.lbg->findBestMatches( desc, idx1, val1, idx2, val2 );

        // std::cout << "matches -" << setprecision(3);

        // int h = 0;

        if( idx1 >= 0 )
        {
            int   idx3, idx4;
            float val3, val4, val5, val6;
            _level2[idx1].lbg->findBestMatches( desc, idx3, val3, idx4, val4 );

            _level2[idx1].lbg->findBestDescriptor( desc, idx3, d5, val5, d6, val6 );
            // if( d5 != 0 ) std::cout << " " << h++ << ": " << val5;
            // if( d6 != 0 ) std::cout << " " << h++ << ": " << val6;
#if 1
            if( d5 != 0 ) themins.insert( val5 );
            if( d6 != 0 ) themins.insert( val6 );
#endif

            _level2[idx1].lbg->findBestDescriptor( desc, idx4, d5, val5, d6, val6 );
            // if( d5 != 0 ) std::cout << " " << h++ << ": " << val5;
            // if( d6 != 0 ) std::cout << " " << h++ << ": " << val6;
#if 1
            if( d5 != 0 ) themins.insert( val5 );
            if( d6 != 0 ) themins.insert( val6 );
#endif
        }

        if( idx2 >= 0 )
        {
            int   idx3, idx4;
            float val3, val4, val5, val6;
            _level2[idx2].lbg->findBestMatches( desc, idx3, val3, idx4, val4 );

            _level2[idx1].lbg->findBestDescriptor( desc, idx3, d5, val5, d6, val6 );
            // if( d5 != 0 ) std::cout << " " << h++ << ": " << val5;
            // if( d6 != 0 ) std::cout << " " << h++ << ": " << val6;
#if 1
            if( d5 != 0 ) themins.insert( val5 );
            if( d6 != 0 ) themins.insert( val6 );
#endif

            _level2[idx1].lbg->findBestDescriptor( desc, idx4, d5, val5, d6, val6 );
            // if( d5 != 0 ) std::cout << " " << h++ << ": " << val5;
            // if( d6 != 0 ) std::cout << " " << h++ << ": " << val6;
#if 1
            if( d5 != 0 ) themins.insert( val5 );
            if( d6 != 0 ) themins.insert( val6 );
#endif
        }
        // std::cout << std::endl;

#if 1
#ifdef BRUTE_FORCE
        if( *themins.begin() == minval ) std::cout << "+++ "; else std::cout << "--- ";
#endif
        std::cout << "set: ";
        std::copy( themins.begin(), themins.end(), std::ostream_iterator<float>( std::cout, " " ) );
        std::cout << std::endl;
#endif
    }
}; // class PqtAnn

/*************************************************************
 * K-Finder
 *************************************************************/

namespace kfind
{
    struct entry_t
    {
        uint8_t v[128];
    };

    bool operator<( const entry_t& l, const entry_t& r )
    {
        for( int i=0; i<128; i++ )
        {
            if( l.v[i] < r.v[i] ) return true;
            if( l.v[i] > r.v[i] ) return false;
        }
        return false;
    }

    bool operator==( const entry_t& l, const entry_t& r )
    {
        if( memcmp( &l, &r, sizeof(entry_t) ) == 0 ) return true;
        return false;
    }

    std::ostream& operator<<( std::ostream& ostr, const entry_t& e )
    {
        int sum = 0;
        ostr << "{ ";
#ifdef KFINDER_SMOOTH_8
        for( int i=0; i<4; i++ )
#else
        for( int i=0; i<128; i++ )
#endif
        {
            ostr << int(e.v[i]) << " ";
            sum += e.v[i];
        }
        ostr << "(sum " << sum << ")}";
        return ostr;
    }
};

class KFinder
{
    std::set<kfind::entry_t> dSet;

public:
    void insert( int level, const Descriptor& desc )
    {
        kfind::entry_t a;
#ifdef KFINDER_SMOOTH_8
        for( int i=0; i<4; i++ )
        {
            float p = 0;
            for( int j=0; j<32; j++ )
                p += desc.features[i*8+j];
            a.v[i] = ( (int)(255.0f * p) >> 1 );
        }
        for( int i=4; i<128; i++ )
        {
            a.v[i] = 0;
        }
#else
        for( int i=0; i<128; i++ )
        {
            a.v[i] = ( (int)(255.0f * desc.features[i]) >> (8-level) );
        }
#endif

        dSet.insert( a );
    }

    void printSet( std::ostream& ostr )
    {
        ostr << "Set size: " << dSet.size() << std::endl;

        for( auto& e : dSet )
        {
            ostr << e << std::endl;
        }
    }
};

/*************************************************************
 * FeaturesBase
 *************************************************************/

FeaturesBase::FeaturesBase( )
    : _num_ext( 0 )
    , _num_ori( 0 )
{ }

FeaturesBase::~FeaturesBase( )
{ }

/*************************************************************
 * FeaturesHost
 *************************************************************/

FeaturesHost::FeaturesHost( )
    : _ext( 0 )
    , _ori( 0 )
{ }

FeaturesHost::FeaturesHost( int num_ext, int num_ori )
    : _ext( 0 )
    , _ori( 0 )
{
    reset( num_ext, num_ori );
}

FeaturesHost::~FeaturesHost( )
{
    memalign_free( _ext );
    memalign_free( _ori );
}

void FeaturesHost::reset( int num_ext, int num_ori )
{
    if( _ext != 0 ) { free( _ext ); _ext = 0; }
    if( _ori != 0 ) { free( _ori ); _ori = 0; }

    _ext = (Feature*)memalign( getPageSize(), num_ext * sizeof(Feature) );
    if( _ext == 0 ) {
        cerr << __FILE__ << ":" << __LINE__ << " Runtime error:" << endl
             << "    Failed to (re)allocate memory for downloading " << num_ext << " features" << endl;
        if( errno == EINVAL ) cerr << "    Alignment is not a power of two." << endl;
        if( errno == ENOMEM ) cerr << "    Not enough memory." << endl;
        exit( -1 );
    }
    _ori = (Descriptor*)memalign( getPageSize(), num_ori * sizeof(Descriptor) );
    if( _ori == 0 ) {
        cerr << __FILE__ << ":" << __LINE__ << " Runtime error:" << endl
             << "    Failed to (re)allocate memory for downloading " << num_ori << " descriptors" << endl;
        if( errno == EINVAL ) cerr << "    Alignment is not a power of two." << endl;
        if( errno == ENOMEM ) cerr << "    Not enough memory." << endl;
        exit( -1 );
    }

    setFeatureCount( num_ext );
    setDescriptorCount( num_ori );
}

void FeaturesHost::pin( )
{
    cudaError_t err;
    err = cudaHostRegister( _ext, getFeatureCount() * sizeof(Feature), 0 );
    if( err != cudaSuccess ) {
        cerr << __FILE__ << ":" << __LINE__ << " Runtime warning:" << endl
             << "    Failed to register feature memory in CUDA." << endl
             << "    Features count: " << getFeatureCount() << endl
             << "    Memory size requested: " << getFeatureCount() * sizeof(Feature) << endl
             << "    " << cudaGetErrorString(err) << endl;
    }
    err = cudaHostRegister( _ori, getDescriptorCount() * sizeof(Descriptor), 0 );
    if( err != cudaSuccess ) {
        cerr << __FILE__ << ":" << __LINE__ << " Runtime warning:" << endl
             << "    Failed to register descriptor memory in CUDA." << endl
             << "    Descriptors count: " << getDescriptorCount() << endl
             << "    Memory size requested: " << getDescriptorCount() * sizeof(Descriptor) << endl
             << "    " << cudaGetErrorString(err) << endl;
    }
}

void FeaturesHost::unpin( )
{
    cudaHostUnregister( _ext );
    cudaHostUnregister( _ori );
}

void FeaturesHost::print( std::ostream& ostr, bool write_as_uchar ) const
{
    for( int i=0; i<size(); i++ ) {
        _ext[i].print( ostr, write_as_uchar );
    }
}

void FeaturesHost::writeBinary( std::ostream& ostr, bool write_as_uchar ) const
{
    if( write_as_uchar )
    {
        ostr << "1 # writes descriptor as 128 uchars" << std::endl;
    }
    else
    {
        ostr << "0 # writes descriptor as 128 floats" << std::endl;
    }

    uint32_t num = getDescriptorCount();

    ostr << num << " # number of descriptors" << std::endl;

    for( int i=0; i<size(); i++ )
    {
        _ext[i].writeBinaryKeypoint( ostr );
    }
    int descriptors_written = 0;
    for( int i=0; i<size(); i++ )
    {
        descriptors_written += _ext[i].writeBinaryDescriptor( ostr, write_as_uchar );
    }
    std::cerr << "Written " << descriptors_written << " descriptors" << std::endl;
}

void FeaturesHost::debugCompareBinary( std::istream& verify, bool write_as_uchar ) const
{
    FeaturesHost dummy;
    dummy.readBinary( verify );
    if( getDescriptorCount() == dummy.getDescriptorCount() )
    {
        int read_ori_idx  = 0;
        int read_desc_idx = 0;

        for( int desc=0; desc<getDescriptorCount(); desc++ )
        {
            if( _ext[read_ori_idx].xpos != dummy._ext[desc].xpos )
            {
                std::cerr << "Written xpos bad on re-reading: "
                          << _ext[read_ori_idx].xpos << " vs " << dummy._ext[desc].xpos
                          << " for descriptor #" << desc
                          << std::endl;
                return;
            }

            if( _ext[read_ori_idx].ypos != dummy._ext[desc].ypos )
            {
                std::cerr << "Written ypos bad on re-reading: "
                          << _ext[read_ori_idx].ypos << " vs " << dummy._ext[desc].ypos
                          << " for descriptor #" << desc
                          << std::endl;
                return;
            }

            if( _ext[read_ori_idx].sigma != dummy._ext[desc].sigma )
            {
                std::cerr << "Written sigma bad on re-reading: "
                          << _ext[read_ori_idx].sigma << " vs " << dummy._ext[desc].sigma
                          << " for descriptor #" << desc
                          << std::endl;
                return;
            }

            if( _ext[read_ori_idx].orientation[read_desc_idx] != dummy._ext[desc].orientation[0] )
            {
                std::cerr << "Written orientation bad on re-reading: "
                          << _ext[read_ori_idx].orientation[read_desc_idx] << " vs " << dummy._ext[desc].orientation[0]
                          << " for descriptor #" << desc
                          << std::endl;
                return;
            }

            read_desc_idx++;
            if( read_desc_idx >= _ext[read_ori_idx].num_ori )
            {
                read_ori_idx++;
                read_desc_idx = 0;
            }
        }

        for( int desc=0; desc<getDescriptorCount(); desc++ )
        {
            for( int d=0; d<128; d++ )
            {
                float actual = _ori[desc].features[d];
                float reread = dummy._ori[desc].features[d];
                if( write_as_uchar )
                {
                    actual = (unsigned char)roundf(actual);
                }

                if( actual != reread )
                {
                    std::cerr << "Difference in descriptor " << desc << " dim " << d << ": "
                              << actual << " vs " << reread
                              << std::endl;
                    break;
                }
            }
        }
    }
    else
    {
        std::cerr << "Wrote " << getDescriptorCount() << " descriptors, reading " << dummy.getDescriptorCount() << std::endl;
    }
}

bool FeaturesHost::readBinary( std::istream& ostr )
{
    bool written_as_uchar = false;
    int  num_descriptors  = 0;

    {
        char buffer[1024];
        ostr.getline( buffer, 1024 );
        written_as_uchar = ( buffer[0] == '1' );
        ostr >> num_descriptors;
        ostr.getline( buffer, 1024 ); // read rest of line and discard
    }

    if( num_descriptors <= 0 )
    {
        return false;
    }

    reset( num_descriptors, num_descriptors ); // descriptors written multiple times

    for( int i=0; i<num_descriptors; i++ )
    {
        float desc[4];
        ostr.read( (char*)desc, 4*sizeof(float) );
        _ext[i].debug_octave   = 0;
        _ext[i].xpos           = desc[0];
        _ext[i].ypos           = desc[1];
        _ext[i].sigma          = desc[2];
        _ext[i].num_ori        = 1;
        _ext[i].orientation[0] = desc[3];
        _ext[i].desc[0]        = &_ori[i];
    }

    if( written_as_uchar )
    {
        std::cerr << "Reading " << num_descriptors << " uchar descriptors" << std::endl;
        unsigned char* v = new unsigned char[128 * num_descriptors];
        unsigned char* vit = v;
        ostr.read( (char*)v, 128 * num_descriptors * sizeof(unsigned char) );
        for( int i=0; i<num_descriptors; i++ )
        {
            for( int d=0; d<128; d++ )
            {
                _ori[i].features[d] = *vit;
                vit++;
            }
        }
        delete [] v;
    }
    else
    {
        std::cerr << "Reading " << num_descriptors << " float descriptors" << std::endl;

        // Descriptor contains only features[128], linear read should be equivalent to
        // for( int i=0; i<num_descriptors; i++ ) ostr.read( (char*)(_ori[i].features), 128 * sizeof(float) );
        ostr.read( (char*)(_ori[0].features), 128 * num_descriptors * sizeof(float) );
    }

    return true;
}

__global__
void fix_descriptor_pointers( Feature*    features,
                              int         feature_count,
                              Descriptor* old_base_ptr,
                              Descriptor* new_base_ptr )
{
    const int idx = blockIdx.x * 32 + threadIdx.x;
    if( idx > feature_count ) return;
    Feature& f = features[idx];
    for( int ori=0; ori<f.num_ori; ori++ )
    {
        f.desc[ori] = (Descriptor*)( (char*)(f.desc[ori]) - (char*)(old_base_ptr) + (char*)(new_base_ptr) );
    }
}

__global__
void fix_reverse_map( Feature*    features,
                      int         feature_count,
                      Descriptor* desc_base,
                      int*        reverseMap )
{
    const int idx = blockIdx.x * 32 + threadIdx.x;
    if( idx > feature_count ) return;
    Feature& f = features[idx];
    for( int ori=0; ori<f.num_ori; ori++ )
    {
        
        Descriptor* desc_this = f.desc[ori];
        int offset = desc_this - desc_base;
        reverseMap[offset] = idx;
    }
}

FeaturesDev* FeaturesHost::toDevice()
{
    FeaturesDev* dev_features = new FeaturesDev( getFeatureCount(), getDescriptorCount() );
    pin();
    popcuda_memcpy_sync( dev_features->getFeatures(),
                         getFeatures(),
                         getFeatureCount() * sizeof(Feature),
                         cudaMemcpyHostToDevice );

    popcuda_memcpy_sync( dev_features->getDescriptors(),
                         getDescriptors(),
                         getDescriptorCount() * sizeof(Descriptor),
                         cudaMemcpyHostToDevice );
    unpin();
    
    dim3 grid( grid_divide( getFeatureCount(), 32 ) );
    fix_descriptor_pointers
        <<<grid,32>>>
        ( dev_features->getFeatures(),
          getFeatureCount(),
          getDescriptors(),
          dev_features->getDescriptors() );
    fix_reverse_map
        <<<grid,32>>>
        ( dev_features->getFeatures(),
          dev_features->getFeatureCount(),
          dev_features->getDescriptors(),
          dev_features->getReverseMap() );
    return dev_features;
}

void FeaturesHost::match( FeaturesHost* other )
{
    int         l_len  = getDescriptorCount( );
    Descriptor* l_ori  = getDescriptors( );

#ifdef KFINDER
    KFinder kfind;
    for( int i=0; i<l_len; i++ )
        kfind.insert( 2, l_ori[i] );
    kfind.printSet( std::cout );
#endif

    std::vector<Descriptor*> ori( l_len );
    for( int i=0; i<l_len; i++ )
        ori[i] = &l_ori[i];

#ifdef PQ_LBQ
    const int   rounds = 4;
    LindeBuzoGray<8>* pq_lbg[16];
    for( int i=0; i<16; i++ )
    {
        pq_lbg[i] = new LindeBuzoGray<8>( i*8, ori, rounds );
        pq_lbg[i]->run( false );
    }
#endif
    // const int   rounds = 4;
    // LindeBuzoGray<5> lbg( ori, rounds );
    // lbg.run( true );

    PqtAnn pqt( ori );
    pqt.run();

    int         r_len  = other->getDescriptorCount( );
    Descriptor* r_ori  = other->getDescriptors( );
    for( int i=0; i<r_len; i++ )
    {
        pqt.findMatch( r_ori[i] );
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
    : _ext( 0 )
    , _ori( 0 )
    , _rev( 0 )
{ }

FeaturesDev::FeaturesDev( int num_ext, int num_ori )
    : _ext( 0 )
    , _ori( 0 )
    , _rev( 0 )
{
    reset( num_ext, num_ori );
}

FeaturesDev::~FeaturesDev( )
{
    cudaFree( _ext );
    cudaFree( _ori );
    cudaFree( _rev );
}

void FeaturesDev::reset( int num_ext, int num_ori )
{
    if( _ext != 0 ) { cudaFree( _ext ); _ext = 0; }
    if( _ori != 0 ) { cudaFree( _ori ); _ori = 0; }
    if( _rev != 0 ) { cudaFree( _rev ); _rev = 0; }

    _ext = popsift::cuda::malloc_devT<Feature>   ( num_ext, __FILE__, __LINE__ );
    _ori = popsift::cuda::malloc_devT<Descriptor>( num_ori, __FILE__, __LINE__ );
    _rev = popsift::cuda::malloc_devT<int>       ( num_ori, __FILE__, __LINE__ );

    setFeatureCount( num_ext );
    setDescriptorCount( num_ori );
}

__device__ inline float
l2_in_t0( const float4* lptr, const float4* rptr )
{
    const float4  lval = lptr[threadIdx.x];
    const float4  rval = rptr[threadIdx.x];
    const float4  mval = make_float4( lval.x - rval.x,
			              lval.y - rval.y,
			              lval.z - rval.z,
			              lval.w - rval.w );
    float   res = mval.x * mval.x
	        + mval.y * mval.y
	        + mval.z * mval.z
	        + mval.w * mval.w;
    res += shuffle_down( res, 16 );
    res += shuffle_down( res,  8 );
    res += shuffle_down( res,  4 );
    res += shuffle_down( res,  2 );
    res += shuffle_down( res,  1 );
    return res;
}

__global__ void
compute_distance( int3* match_matrix, Descriptor* l, int l_len, Descriptor* r, int r_len )
{
    if( blockIdx.x >= l_len ) return;
    const int idx = blockIdx.x;

    float match_1st_val = CUDART_INF_F;
    float match_2nd_val = CUDART_INF_F;
    int   match_1st_idx = 0;
    int   match_2nd_idx = 0;

    const float4* lptr = (const float4*)( &l[idx] );

    for( int i=0; i<r_len; i++ )
    {
        const float4* rptr = (const float4*)( &r[i] );

        const float   res  = l2_in_t0( lptr, rptr );

        if( threadIdx.x == 0 )
        {
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
        __syncthreads();
    }

    if( threadIdx.x == 0 )
    {
        bool accept = ( match_1st_val / match_2nd_val < 0.8f );
        match_matrix[blockIdx.x] = make_int3( match_1st_idx, match_2nd_idx, accept );
    }
}

__global__ void
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
        const float4* lptr  = (const float4*)( &l_ori[i] );
        const float4* rptr1 = (const float4*)( &r_ori[match_matrix[i].x] );
        const float4* rptr2 = (const float4*)( &r_ori[match_matrix[i].y] );
        float d1 = l2_in_t0( lptr, rptr1 );
        float d2 = l2_in_t0( lptr, rptr2 );
        if( threadIdx.x == 0 )
        {
            if( match_matrix[i].z )
                printf( "accept feat %4d [%4d] matches feat %4d [%4d] ( 2nd feat %4d [%4d] ) dist %.3f vs %.3f\n",
                        l_fem[i], i,
                        r_fem[match_matrix[i].x], match_matrix[i].x,
                        r_fem[match_matrix[i].y], match_matrix[i].y,
                        d1, d2 );
	    else
                printf( "reject feat %4d [%4d] matches feat %4d [%4d] ( 2nd feat %4d [%4d] ) dist %.3f vs %.3f\n",
                        l_fem[i], i,
                        r_fem[match_matrix[i].x], match_matrix[i].x,
                        r_fem[match_matrix[i].y], match_matrix[i].y,
                        d1, d2 );
        }
        __syncthreads();
    }
}

void FeaturesDev::match( FeaturesDev* other )
{
    int l_len = getDescriptorCount( );
    int r_len = other->getDescriptorCount( );

    int3* match_matrix = popsift::cuda::malloc_devT<int3>( l_len, __FILE__, __LINE__ );

    dim3 grid;
    grid.x = l_len;
    grid.y = 1;
    grid.z = 1;
    dim3 block;
    block.x = 32;
    block.y = 1;
    block.z = 1;

    compute_distance
        <<<grid,block>>>
        ( match_matrix, getDescriptors(), l_len, other->getDescriptors(), r_len );

    POP_SYNC_CHK;

    show_distance
        <<<1,32>>>
        ( match_matrix,
          getFeatures(),
          getDescriptors(),
          getReverseMap(),
          l_len,
          other->getFeatures(),
          other->getDescriptors(),
          other->getReverseMap(),
          r_len );

    POP_SYNC_CHK;

    cudaFree( match_matrix );
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
                ostr << roundf(desc[ori]->features[i]) << " ";
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

void Feature::writeBinaryKeypoint( std::ostream& ostr ) const
{
    float keypoint[4];
    keypoint[0] = xpos;
    keypoint[1] = ypos;
    keypoint[2] = sigma;

    for( int ori=0; ori<num_ori; ori++ )
    {
        keypoint[3] = orientation[ori];
        ostr.write( (const char*)keypoint, 4*sizeof(float) );
    }
}

int Feature::writeBinaryDescriptor( std::ostream& ostr, bool write_as_uchar ) const
{
    int descriptors_written = 0;
    for( int ori=0; ori<num_ori; ori++ )
    {
        if( write_as_uchar )
        {
            unsigned char buffer[128];
            for( int i=0; i<128; i++ )
            {
                buffer[i] = (unsigned char)( roundf(desc[ori]->features[i]) );
            }
            ostr.write( (const char*)buffer, 128 * sizeof(unsigned char) );
            descriptors_written++;
        }
        else
        {
            ostr.write( (const char*)(desc[ori]->features), 128 * sizeof(float) );
            descriptors_written++;
        }
    }
    return descriptors_written;
}

std::ostream& operator<<( std::ostream& ostr, const Feature& feature )
{
    feature.print( ostr, false );
    return ostr;
}

} // namespace popsift

