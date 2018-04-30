/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <iomanip>
#include <iostream>
#include <unistd.h>
#ifndef __APPLE__
#include <malloc.h>
#endif
#include <stdlib.h>
#include <errno.h>
#include <math_constants.h>

#include <thrust/sequence.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/unique.h>

#include "features.h"
#include "sift_extremum.h"
#include "common/debug_macros.h"

using namespace std;

__host__ __device__ bool operator<( const int2& l, const int2& r )
{
    return ( l.x < r.x || ( l.x == r.x && l.y < r.y ) );
}

__host__ __device__ bool operator==( const int2& l, const int2& r )
{
    return ( l.x == r.x && l.y == r.y );
}

namespace popsift {

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
    free( _ext );
    free( _ori );
}

#ifdef __APPLE__
static void* memalign( size_t alignment, size_t size )
{
    void* ret;
    int err = posix_memalign( &ret, alignment, size );
    if( err != 0 ) {
        errno = err;
        ret = 0;
    }
    return ret;
}
#endif

void FeaturesHost::reset( int num_ext, int num_ori )
{
    if( _ext != 0 ) { free( _ext ); _ext = 0; }
    if( _ori != 0 ) { free( _ori ); _ori = 0; }

    _ext = (Feature*)memalign( sysconf(_SC_PAGESIZE), num_ext * sizeof(Feature) );
    if( _ext == 0 ) {
        cerr << __FILE__ << ":" << __LINE__ << " Runtime error:" << endl
             << "    Failed to (re)allocate memory for downloading " << num_ext << " features" << endl;
        if( errno == EINVAL ) cerr << "    Alignment is not a power of two." << endl;
        if( errno == ENOMEM ) cerr << "    Not enough memory." << endl;
        exit( -1 );
    }
    _ori = (Descriptor*)memalign( sysconf(_SC_PAGESIZE), num_ori * sizeof(Descriptor) );
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

std::ostream& operator<<( std::ostream& ostr, const FeaturesHost& feature )
{
    feature.print( ostr, false );
    return ostr;
}

/*************************************************************
 * FeaturesDev - device and global functions
 *************************************************************/

/**
 * Compute L2 distance of two arrays of 32 float4 groups.
 * Kernel configuration must be <<<:,32>>>, 32 threads per warp, 1 warp per block.
 */
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
    res += __shfl_down( res, 16 );
    res += __shfl_down( res,  8 );
    res += __shfl_down( res,  4 );
    res += __shfl_down( res,  2 );
    res += __shfl_down( res,  1 );
    return res;
}

/**
 * Compute L2 distance of two arrays of 128 float groups
 * Kernel configuration must be <<<:,32>>>, 32 threads per warp, 1 warp per block.
 */
__device__ inline float
l2_in_t0( const float* l, const float* r )
{
    const float4* lptr = (const float4*)l;
    const float4* rptr = (const float4*)r;
    return l2_in_t0( lptr, rptr );
}

/**
 * Brute-force computation of descriptor matches.
 * It assume to be called with <<<L,32>>>, ie. L blocks and 32 threads per block,
 * where L is >= the length of the left descriptor array.
 * l and r are descriptor arrays, l_len and r_len is the length of the arrays,
 * respectively.
 * On return, match_matrix contains an array sorted in the order of descriptor array l,
 * containing triples (1st,2nd,accept), where 1st is the index of the first match in
 * r, 2nd is the index of the 2nd match, accept is true if the distance ratio between 1st
 * and 2nd match is < 0.8
 */
__global__ void
compute_distance( int3* match_matrix, Descriptor* l, int l_len, Descriptor* r, int r_len )
{
    if( blockIdx.x >= l_len ) return;
    const int idx = blockIdx.x;

    float match_1st_val = CUDART_INF_F;
    float match_2nd_val = CUDART_INF_F;
    int   match_1st_idx = 0;
    int   match_2nd_idx = 0;

    const float* lptr = l[idx].features;

    for( int i=0; i<r_len; i++ )
    {
        const float* rptr = r[i].features;

        const float  res  = l2_in_t0( lptr, rptr );

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
        // printf( "idx %4d : 1st %7.3f 2nd %7.3f\n", idx, match_1st_val, match_2nd_val );
        bool accept = ( match_2nd_val == 0 ) || ( match_1st_val / match_2nd_val < 0.8f );
        match_matrix[blockIdx.x] = make_int3( match_1st_idx, match_2nd_idx, accept );
    }
}

/**
 * Device-sided printing of keypoint matches.
 * Must be called in <<<1,32>>> configuration. Only one thread marches through all
 * matches, but 32 threads are used to re-compute L2 distances a bit more efficiently.
 * l_key and r_key - arrays of keypoints, each keypoint may have several descriptors
 *                   all orientations and pointers to descriptors are contained in the
 *                   Feature structure
 * l_dsc and r_dsc - arrays of descriptors
 * l_fem and r_fem - reverse mapping from descriptor index to keypoint index
 * l_len and r_len - length of the descriptor arrays (NOT the length of the keypoint array)
 */
__global__ void
show_distance( int3*       match_matrix,
               Feature*    l_key,
               Descriptor* l_dsc,
               int*        l_fem,
               int         l_len,
               Feature*    r_key,
               Descriptor* r_dsc,
               int*        r_fem,
               int         r_len )
{
    if( threadIdx.x == 0 )
    {
        printf("<accept|reject> <l-key> [ <l-desc> ] matches <r-key> [ <r-desc> ] dist <1st dist> vs <2nd dist> \n" );
        printf("========================================\n");
    }

    __syncthreads();
    for( int i=0; i<l_len; i++ )
    {
        const float4* lptr  = (const float4*)( &l_dsc[i] );
        const float4* rptr1 = (const float4*)( &r_dsc[match_matrix[i].x] );
        const float4* rptr2 = (const float4*)( &r_dsc[match_matrix[i].y] );
        float d1 = l2_in_t0( lptr, rptr1 );
        float d2 = l2_in_t0( lptr, rptr2 );
        if( threadIdx.x == 0 )
        {
            if( match_matrix[i].z )
                printf( "accept %4d [%4d] matches %4d [%4d] ( 2nd feat %4d [%4d] ) dist %.3f vs %.3f\n",
                        l_fem[i], i,
                        r_fem[match_matrix[i].x], match_matrix[i].x,
                        r_fem[match_matrix[i].y], match_matrix[i].y,
                        d1, d2 );
            else
                printf( "reject %4d [%4d] matches %4d [%4d] ( 2nd feat %4d [%4d] ) dist %.3f vs %.3f\n",
                        l_fem[i], i,
                        r_fem[match_matrix[i].x], match_matrix[i].x,
                        r_fem[match_matrix[i].y], match_matrix[i].y,
                        d1, d2 );
        }
        __syncthreads();
    }
}

/** Similar to the other show_distance function, but this one takes also an
 *  index vector referring only to accepted matches.
 */
__global__ void
show_distance( int3*       match_matrix,
               int*        l_accepted_map,
               int         l_accepted_len,
               Feature*    l_key,
               Descriptor* l_dsc,
               int*        l_fem,
               int         l_len,
               Feature*    r_key,
               Descriptor* r_dsc,
               int*        r_fem,
               int         r_len )
{
    if( threadIdx.x == 0 )
    {
        printf("feat <l-key> [ <l-desc> ] matches feat <r-key> [ <r-desc> ] dist <dist>\n" );
        printf("========================================\n");
    }
    __syncthreads();

    for( int j=0; j<l_accepted_len; j++ ) // j - index in accepted list
    {
        int i = l_accepted_map[j]; // i - descriptor index in left descriptor list

        if( threadIdx.x == 0 )
        {
#if 0
            /* Test should be irrelevant. Active only for debugging. */
            if( not match_matrix[i].z )
            {
                printf( "rejected element found in accepted index list" );
                return;
            }
#endif
            int      l_feat_idx = l_fem[i];
            Feature& l_feat     = l_key[l_feat_idx];
            int      r_feat_idx = r_fem[match_matrix[i].x];
            Feature& r_feat     = r_key[r_feat_idx];
            printf( "feat %4d ( %4.1f %4.1f ) matches feat %4d ( %4.1f %4.1f )\n",
                     l_feat_idx, l_feat.xpos, l_feat.ypos,
                     r_feat_idx, r_feat.xpos, r_feat.ypos );
        }
        __syncthreads();
    }
}

/** Helper class to call thrust::remove_if on the vector of matches.
 *  Results in an index (generated with thrust::sequence) to contain
 *  only accepted matches.
 */
class RemoveNotAccepted
{
public:
    RemoveNotAccepted( const int3* matchmap ) : _m( matchmap ) { }

    /** Was match idx accepted? Return true if not. */
    __device__
    bool operator()( const int idx )
    {
        return ( _m[idx].z == 0 );
    }
private:
    const int3* _m;
};

/** Helper class to call thrust::sort on the index of features.
 *  Results in an index that sorts the features by x,y,s coordinates.
 */
class SortByCoords
{
public:
    SortByCoords( const Feature* f ) : _f(f) { }

    __device__
    bool operator()( int lidx, int ridx )
    {
        const Feature& l = _f[lidx];
        const Feature& r = _f[ridx];
        if( ( l.xpos  < r.xpos ) ||
            ( l.xpos == r.xpos && l.ypos  < r.ypos ) ||
            ( l.xpos == r.xpos && l.ypos == r.ypos && l.scale  < r.scale ) ||
            ( l.xpos == r.xpos && l.ypos == r.ypos && l.scale == r.scale && l.num_ori < r.num_ori ) )
        {
            return true;
        }
        if( l.xpos == r.xpos && l.ypos == r.ypos && l.scale == r.scale && l.num_ori == r.num_ori )
        {
#if ORIENTATION_MAX_COUNT == 3
            return ( l.orientation[0] <= r.orientation[0] &&
                     l.orientation[1] <= r.orientation[1] &&
                     l.orientation[2] <= r.orientation[2] );
#elif ORIENTATION_MAX_COUNT == 4
            return ( l.orientation[0] <= r.orientation[0] &&
                     l.orientation[1] <= r.orientation[1] &&
                     l.orientation[2] <= r.orientation[2] &&
                     l.orientation[3] <= r.orientation[3] );
#else
#error Code not adapted to new ORIENTATION_MAX_COUNT macro
#endif
        }
        return false;
    }

private:
    const Feature* _f;
};

/** Helper class to call thrust::equal on two feature index sets.
 *  thrust::equal has only one global result, true or false.
 */
class CompareFeatures
{
public:
    CompareFeatures( const Feature* l_f, const Feature* r_f ) : _lf( l_f ), _rf( r_f ) { }

    __device__
    bool operator()( int l, int r )
    {
        const Feature& lp = _lf[l];
        const Feature& rp = _rf[r];
        if( lp.xpos == rp.xpos && lp.ypos == rp.ypos && lp.scale == rp.scale )
        {
            // printf("match at %4.2f %4.2f %4.2f\n", lp.xpos, lp.ypos, lp.scale );
            return true;
        }
        else
        {
            printf("mismatch between %4.2f %4.2f %4.2f and %4.2f %4.2f %4.2f\n",
                   lp.xpos, lp.ypos, lp.scale,
                   rp.xpos, rp.ypos, rp.scale );
            return false;
        }
    }
private:
    const Feature* _lf;
    const Feature* _rf;
};

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

    POP_CHECK_NON_NULL( _ext, "failed to allocate Feature array" );
    POP_CHECK_NON_NULL( _ori, "failed to allocate Descriptor array" );
    POP_CHECK_NON_NULL( _rev, "failed to allocate Reverse Mapping array" );

    setFeatureCount( num_ext );
    setDescriptorCount( num_ori );
}

int3* FeaturesDev::match_AllocMatchTable( ) const
{
    int   l_len = getDescriptorCount( );
    int3* match_matrix = popsift::cuda::malloc_devT<int3>( l_len, __FILE__, __LINE__ );
    POP_CHECK_NON_NULL( match_matrix, "failed to allocate Matching descriptor table" );
    return match_matrix;
}

void FeaturesDev::match_freeMatchTable( int3* match_matrix ) const
{
    cudaFree( match_matrix );
}

void FeaturesDev::match_computeMatchTable( int3* match_matrix, const FeaturesDev* other ) const
{
    int l_len = getDescriptorCount( );
    int r_len = other->getDescriptorCount( );

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
}

int* FeaturesDev::match_getAcceptedIndex( const int3* match_matrix, int& l_accepted_len ) const
{
    int  l_len = getDescriptorCount( );

    // create and fill list of indexes for left descriptors
    int* l_accepted_map = popsift::cuda::malloc_devT<int>( l_len, __FILE__, __LINE__ );
    if( not l_accepted_map )
    {
        POP_FATAL( "Could not allocate index list from accepted matches" );
    }

    thrust::device_ptr<int> sptr = thrust::device_pointer_cast( l_accepted_map );

    thrust::sequence( sptr, sptr+l_len );

    // Using the return value of remove_if on the host side is not a great move,
    // because it requires local memory and a device-host copy operation.
    // Good for coding time.
    RemoveNotAccepted rna( match_matrix );
    thrust::device_ptr<int> last = thrust::remove_if( sptr, sptr+l_len, rna );

    l_accepted_len = last - sptr;

    return l_accepted_map;
}

void FeaturesDev::match_freeAcceptedIndex( int* index ) const
{
    cudaFree( index );
}

class MatchAcceptTransform
{
public:
    __device__
    int2 operator()( const thrust::tuple<int3,int>& v ) const
    {
        const int  l = thrust::get<1>( v );
        const int3 r = thrust::get<0>( v );
        return make_int2( l, r.x );
    }
};

class MatchAcceptFeatTrans
{
public:
    /* We transform a pair of descriptor indices that we find in
     * the input tuple into a pair of keypoint indices.
     */
    MatchAcceptFeatTrans( int*     l_fem,
                          int*     r_fem )
        : _l_fem( l_fem )
        , _r_fem( r_fem )
    { }

    __device__
    int2 operator()( const thrust::tuple<int3,int>& v ) const
    {
        const int  l = thrust::get<1>( v );
        const int3 r = thrust::get<0>( v );
        return make_int2( _l_fem[l], _r_fem[r.x] );
    }

private:
    int*     _l_fem;
    int*     _r_fem;
};

class MatchAcceptPredicate
{
public:
    __device__
    bool operator()( const thrust::tuple<int3,int>& v ) const
    {
        const int3 r = thrust::get<0>( v );
        return ( r.z == true );
    }
};

void FeaturesDev::match_getAcceptedDescriptorMatchesFromMatrix( int3* match_matrix, thrust::device_vector<int2>& accepted_matches ) const
{
    int  l_len = getDescriptorCount( );

    accepted_matches.reserve( l_len );

    /* iterator that exists instead an physically allocated sequence of indices
     * representing the index of "left" descriptors */
    thrust::counting_iterator<int> cptr( 0 );
    thrust::counting_iterator<int> cend  = cptr + l_len;

    /* iterator of the matching table, which has size l_len and which is indexed
     * by the "left" descriptor index. Contains the best and second best "right"
     * match for each "left" index */
    thrust::device_ptr<int3> sptr = thrust::device_pointer_cast( match_matrix );
    thrust::device_ptr<int3> send = sptr + l_len;

    /* merge the two input iterators into a single zip iterator */
    auto t_begin = thrust::make_zip_iterator( thrust::make_tuple( sptr, cptr ) );
    auto t_end   = thrust::make_zip_iterator( thrust::make_tuple( send, cend ) );

    MatchAcceptTransform match_accept_transform;
    MatchAcceptPredicate match_accept_predicate;

    auto m_begin = accepted_matches.begin();
    auto m_end   = thrust::transform_if( t_begin,
                                         t_end,
                                         m_begin,
                                         match_accept_transform,
                                         match_accept_predicate );
    cout << __FILE__ << ":" << __LINE__ << endl
         << "    Number of matching descriptors found: " << ( m_end - m_begin ) << endl;
}

void FeaturesDev::match_getAcceptedFeatureMatchesFromMatrix(
                int3*                        match_matrix,
                thrust::device_vector<int2>& accepted_matches,
                int*                         l_fem,
                int*                         r_fem ) const
{
    int  l_len = getDescriptorCount( );

    accepted_matches.resize( l_len );

    /* iterator that exists instead an physically allocated sequence of indices
     * representing the index of "left" descriptors */
    thrust::counting_iterator<int> cptr( 0 );
    thrust::counting_iterator<int> cend  = cptr + l_len;

    /* iterator of the matching table, which has size l_len and which is indexed
     * by the "left" descriptor index. Contains the best and second best "right"
     * match for each "left" index */
    thrust::device_ptr<int3> sptr = thrust::device_pointer_cast( match_matrix );
    thrust::device_ptr<int3> send = sptr + l_len;

    /* merge the two input iterators into a single zip iterator */
    auto t_begin = thrust::make_zip_iterator( thrust::make_tuple( sptr, cptr ) );
    auto t_end   = thrust::make_zip_iterator( thrust::make_tuple( send, cend ) );

    MatchAcceptFeatTrans match_accept_transform( l_fem, r_fem );
    MatchAcceptPredicate match_accept_predicate;

    /* Write to the output pairs (l,r), where
     * l is the feature remapped from the "left" descriptor index li
     * r is the feature remapped from the best matching "right" descriptor at li
     */
    auto m_begin = accepted_matches.begin();
    auto m_end   = thrust::transform_if( t_begin,
                                         t_end,
                                         m_begin,
                                         match_accept_transform,
                                         match_accept_predicate );
    pop_sync_check_last_error( __FILE__, __LINE__ );

    /* Use sort before unique */
    thrust::sort( m_begin, m_end );
    pop_sync_check_last_error( __FILE__, __LINE__ );

    /* Keep every "left" feature only one in the list. It is possible
     * that a "right" feature matches more than one "left" feature.
     */
    auto m_cut = thrust::unique( m_begin, m_end );
    pop_sync_check_last_error( __FILE__, __LINE__ );

    accepted_matches.erase( m_cut, m_end );
}

void FeaturesDev::matchAndPrint( const FeaturesDev* other ) const
{
    int3* match_matrix = match_AllocMatchTable( );

    match_computeMatchTable( match_matrix, other );

    int l_len = getDescriptorCount( );
    int r_len = other->getDescriptorCount( );

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

    match_freeMatchTable( match_matrix );
}

void FeaturesDev::matchAndPrintAccepted( const FeaturesDev* other ) const
{
    int3* match_matrix = match_AllocMatchTable( );

    match_computeMatchTable( match_matrix, other );

    int l_len = getDescriptorCount( );
    int r_len = other->getDescriptorCount( );

    int  l_accepted_len;
    int* l_accepted_map = match_getAcceptedIndex( match_matrix, l_accepted_len );

    cout << "l_len = " << l_len << endl;
    cout << "accepted = " << l_accepted_len << endl;

    show_distance
        <<<1,32>>>
        ( match_matrix,
          l_accepted_map,
          l_accepted_len,
          getFeatures(),
          getDescriptors(),
          getReverseMap(),
          l_len,
          other->getFeatures(),
          other->getDescriptors(),
          other->getReverseMap(),
          r_len );

    POP_SYNC_CHK;

    match_freeAcceptedIndex( l_accepted_map );
    match_freeMatchTable( match_matrix );
}

void FeaturesDev::checkIdentity( const FeaturesDev* other ) const
{
    int l_len = getFeatureCount( );
    int r_len = other->getFeatureCount( );

    if( l_len != r_len )
    {
        cout << "feature counts for identical images should be the same, but l=" << l_len << " and r=" << r_len << endl;
        return;
    }

    cout << "feature counts for identical images is " << l_len << endl;

    thrust::device_vector<int> l_vec( l_len );
    thrust::device_vector<int> r_vec( r_len );

    thrust::sequence( l_vec.begin(), l_vec.end() );
    thrust::sequence( r_vec.begin(), r_vec.end() );

    SortByCoords l_sbc( getFeatures() );
    SortByCoords r_sbc( other->getFeatures() );

    thrust::sort( l_vec.begin(), l_vec.end(), l_sbc );
    thrust::sort( r_vec.begin(), r_vec.end(), r_sbc );

    CompareFeatures cf( getFeatures(), other->getFeatures() );
    bool eq = thrust::equal( l_vec.begin(), l_vec.end(), r_vec.begin(), cf );

    if( not eq )
    {
        cout << "sorted features do not have the coordinates for identical images" << endl;
        return;
    }

    cout << "sorted features have the coordinates for identical images" << endl;
}

void FeaturesDev::match_getAcceptedDescriptorMatches( thrust::device_vector<int2>& matches, const FeaturesDev* other ) const
{
    int3* match_matrix = match_AllocMatchTable( );

    match_computeMatchTable( match_matrix, other );

    match_getAcceptedDescriptorMatchesFromMatrix( match_matrix, matches );

    match_freeMatchTable( match_matrix );
}

void FeaturesDev::match_getAcceptedFeatureMatches( thrust::device_vector<int2>& matches, const FeaturesDev* that ) const
{
    int3* match_matrix = match_AllocMatchTable( );

    match_computeMatchTable( match_matrix, that );

    match_getAcceptedFeatureMatchesFromMatrix(
            match_matrix,
            matches,
            this->getReverseMap(),
            that->getReverseMap() );

    match_freeMatchTable( match_matrix );
}

/*************************************************************
 * Feature
 *************************************************************/

void Feature::print( std::ostream& ostr, bool write_as_uchar ) const
{
    float sigval =  1.0f / ( scale * scale );

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

std::ostream& operator<<( std::ostream& ostr, const Feature& feature )
{
    feature.print( ostr, false );
    return ostr;
}

} // namespace popsift

