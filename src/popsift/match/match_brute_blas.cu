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

#include "match_brute_blas.h"
#include "../sift_features.h"
#include "../sift_extremum.h"
#include "../common/debug_macros.h"
#include "../common/cublas_init.h"

using namespace std;

__host__ __device__ bool operator<( const int2& l, const int2& r );

__host__ __device__ bool operator==( const int2& l, const int2& r );

namespace popsift {

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

/*************************************************************
 * FeaturesDev - device and global functions
 *************************************************************/

__global__ void
compute_distance( int3* match_matrix, Descriptor* l, int l_len, Descriptor* r, int r_len );

#undef COMPUTE_32

#ifdef COMPUTE_32
/* Find the first and second best index and value from a full brute force projection.
 * All "right" distances for one "left" line are in one row.
 * Start with <<<rows,32>>>.
 */
__global__ void
compute_distance( int3* match_matrix, const float* array, int rows, int cols, Descriptor* l, Descriptor* r )
{
    if( blockIdx.x >= rows ) return;

    float match_1st_val = -CUDART_INF_F;
    float match_2nd_val = -CUDART_INF_F;
    int   match_1st_idx = 0;
    int   match_2nd_idx = 0;

    int   row_offset = blockIdx.x * cols;

    const float* line = &array[row_offset];

    for( int i=threadIdx.x; ::__any(i<cols); i+=32 )
    {
        const float  res  = line[i];

        if( i<cols && res > match_1st_val )
        {
            match_2nd_val = match_1st_val;
            match_2nd_idx = match_1st_idx;
            match_1st_val = res;
            match_1st_idx = i;
        }
        else if( i<cols && res > match_2nd_val )
        {
            match_2nd_val = res;
            match_2nd_idx = i;
        }
        __syncthreads(); // this require syncthreads forces use of ::__any
    }

    const float save_1st_idx = match_1st_idx;
    const float save_1st_val = match_1st_val;

    match_1st_idx = __shfl_down( match_1st_idx, 16 );
    match_1st_idx = __shfl_down( match_1st_idx,  8 );
    match_1st_idx = __shfl_down( match_1st_idx,  4 );
    match_1st_idx = __shfl_down( match_1st_idx,  2 );
    match_1st_idx = __shfl_down( match_1st_idx,  1 );
    match_1st_idx = __shfl     ( match_1st_idx,  0 );
    match_1st_val = line[match_1st_idx]; // quite cheap broadcast op

    match_2nd_idx = __shfl_down( match_2nd_idx, 16 );
    match_2nd_idx = __shfl_down( match_2nd_idx,  8 );
    match_2nd_idx = __shfl_down( match_2nd_idx,  4 );
    match_2nd_idx = __shfl_down( match_2nd_idx,  2 );
    match_2nd_idx = __shfl_down( match_2nd_idx,  1 );
    match_2nd_idx = __shfl     ( match_2nd_idx,  0 );
    match_2nd_val = line[match_2nd_idx];

    if( save_1st_val < match_1st_val && save_1st_val > match_2nd_val )
    {
        match_2nd_val = save_1st_val;
        match_2nd_idx = save_1st_idx;
    }

    match_2nd_idx = __shfl_down( match_2nd_idx, 16 );
    match_2nd_idx = __shfl_down( match_2nd_idx,  8 );
    match_2nd_idx = __shfl_down( match_2nd_idx,  4 );
    match_2nd_idx = __shfl_down( match_2nd_idx,  2 );
    match_2nd_idx = __shfl_down( match_2nd_idx,  1 );
    match_2nd_idx = __shfl     ( match_2nd_idx,  0 );
    match_2nd_val = line[match_2nd_idx];

    if( threadIdx.x == 0 )
    {
        float x1 = fminf( 1.0f, fmaxf( -1.0f, match_1st_val ) );
        float y1 = sinf( acosf( x1 ) );
        float x2 = fminf( 1.0f, fmaxf( -1.0f, match_2nd_val ) );
        float y2 = sinf( acosf( x2 ) );
        match_1st_val = sqrtf( ( x1-1.0f ) * ( x1-1.0f ) + y1*y1 );
        match_2nd_val = sqrtf( ( x2-1.0f ) * ( x2-1.0f ) + y2*y2 );

        // printf( "idx %4d : 1st %7.3f 2nd %7.3f\n", idx, match_1st_val, match_2nd_val );
        bool accept = ( match_2nd_val == 0 ) || ( match_1st_val / match_2nd_val < 0.8f );
        match_matrix[blockIdx.x] = make_int3( match_1st_idx, match_2nd_idx, accept );
    }
}
#else
/* Start with <<<rows,1>>>. */
__global__ void
compute_distance( int3* match_matrix, const float* array, int rows, int cols, Descriptor* l, Descriptor* r )
{
    if( blockIdx.x >= rows ) return;

    float match_1st_val = -CUDART_INF_F;
    float match_2nd_val = -CUDART_INF_F;
    int   match_1st_idx = 0;
    int   match_2nd_idx = 0;

    int   row_offset = blockIdx.x * cols;

    const float* line = &array[row_offset];

    for( int i=0; i<cols; i+=1 )
    {
        const float  res  = line[i];

        if( i<cols && res > match_1st_val )
        {
            match_2nd_val = match_1st_val;
            match_2nd_idx = match_1st_idx;
            match_1st_val = res;
            match_1st_idx = i;
        }
        else if( i<cols && res > match_2nd_val )
        {
            match_2nd_val = res;
            match_2nd_idx = i;
        }
    }

    printf("dist %5.3f vs %5.3f\n",
        l2_in_t0( r[blockIdx.x].features, l[match_1st_idx].features ),
        l2_in_t0( r[blockIdx.x].features, l[match_2nd_idx].features ) );

    // bool accept = ( match_2nd_val < 0.8f );

    float x1 = fminf( 1.0f, fmaxf( -1.0f, match_1st_val ) );
    float y1 = sinf( acosf( x1 ) );
    float x2 = fminf( 1.0f, fmaxf( -1.0f, match_2nd_val ) );
    float y2 = sinf( acosf( x2 ) );
    match_1st_val = sqrtf( ( x1-1.0f ) * ( x1-1.0f ) + y1*y1 );
    match_2nd_val = sqrtf( ( x2-1.0f ) * ( x2-1.0f ) + y2*y2 );

    bool accept = ( match_2nd_val == 0 ) || ( match_1st_val / match_2nd_val < 0.8f );
    match_matrix[blockIdx.x] = make_int3( match_1st_idx, match_2nd_idx, accept );
}
#endif

__global__ void
show_distance( int3*       match_matrix,
               Feature*    l_key,
               Descriptor* l_dsc,
               int*        l_fem,
               int         l_len,
               Feature*    r_key,
               Descriptor* r_dsc,
               int*        r_fem,
               int         r_len );

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
               int         r_len );

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
 * BruteForceBlasMatcher
 *************************************************************/

BruteForceBlasMatcher::BruteForceBlasMatcher( const FeaturesDev* l, const FeaturesDev* r )
    : _l( l )
    , _r( r )
    , _table( r->getDescriptorCount(), l->getDescriptorCount(), CuFortMatrix::OnDevice, CuFortMatrix::Allocate )
{
    /* Note the flipped dimensions in table because of the Fortran layout hidden inside */
}

int3* BruteForceBlasMatcher::match_AllocMatchTable( ) const
{
    int   l_len = _l->getDescriptorCount( );
    int3* match_matrix = popsift::cuda::malloc_devT<int3>( l_len, __FILE__, __LINE__ );
    POP_CHECK_NON_NULL( match_matrix, "failed to allocate Matching descriptor table" );
    return match_matrix;
}

void BruteForceBlasMatcher::match_freeMatchTable( int3* match_matrix ) const
{
    cudaFree( match_matrix );
}

void BruteForceBlasMatcher::match_computeMatchTable( int3* match_matrix )
{
    int l_len = _l->getDescriptorCount( );
    int r_len = _r->getDescriptorCount( );

    CuFortMatrix lmx( 128, l_len, CuFortMatrix::OnDevice, CuFortMatrix::DontAllocate );
    CuFortMatrix rmx( 128, r_len, CuFortMatrix::OnDevice, CuFortMatrix::DontAllocate );

    lmx.setExternalBuf( (float*)(_l->getDescriptors()) );
    rmx.setExternalBuf( (float*)(_r->getDescriptors()) );

    cublasHandle_t handle;
    cublas_init( &handle, __FILE__, __LINE__ );
    // cerr << "Calling setATransMul" << endl;
    _table.setATransMult( handle, rmx, lmx );

    // CuFortMatrix local( _table.rows(), _table.cols(), CuFortMatrix::OnHost );
    // local = _table;
    // cout << local;

    // cerr << "Done with setATransMul" << endl;
    // sleep( 1 );
    cudaDeviceSynchronize();
    cublas_uninit( handle );

    dim3 grid;
    grid.x = _table.cols(); // should be rows, but stupid Fortran array layout
    grid.y = 1;
    grid.z = 1;
    dim3 block;
#ifdef COMPUTE_32
    block.x = 32;
#else
    block.x = 1;
#endif
    block.y = 1;
    block.z = 1;

    compute_distance
        <<<grid,block>>>
        ( match_matrix, _table.data(), _table.cols(), _table.rows(), _l->getDescriptors(), _r->getDescriptors() );

    POP_SYNC_CHK;
}

int* BruteForceBlasMatcher::match_getAcceptedIndex( const int3* match_matrix, int& l_accepted_len ) const
{
    int  l_len = _l->getDescriptorCount( );

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

void BruteForceBlasMatcher::match_freeAcceptedIndex( int* index ) const
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

void BruteForceBlasMatcher::match_getAcceptedDescriptorMatchesFromMatrix( int3* match_matrix, thrust::device_vector<int2>& accepted_matches ) const
{
    int  l_len = _l->getDescriptorCount( );

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

void BruteForceBlasMatcher::match_getAcceptedFeatureMatchesFromMatrix(
                int3*                        match_matrix,
                thrust::device_vector<int2>& accepted_matches,
                int*                         l_fem,
                int*                         r_fem ) const
{
    int  l_len = _l->getDescriptorCount( );

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

void BruteForceBlasMatcher::matchAndPrint( )
{
    int3* match_matrix = match_AllocMatchTable( );

    match_computeMatchTable( match_matrix );

    int l_len = _l->getDescriptorCount( );
    int r_len = _r->getDescriptorCount( );

    show_distance
        <<<1,32>>>
        ( match_matrix,
          _l->getFeatures(),
          _l->getDescriptors(),
          _l->getReverseMap(),
          l_len,
          _r->getFeatures(),
          _r->getDescriptors(),
          _r->getReverseMap(),
          r_len );

    POP_SYNC_CHK;

    match_freeMatchTable( match_matrix );
}

void BruteForceBlasMatcher::matchAndPrintAccepted( )
{
    int3* match_matrix = match_AllocMatchTable( );

    match_computeMatchTable( match_matrix );

    int l_len = _l->getDescriptorCount( );
    int r_len = _r->getDescriptorCount( );

    int  l_accepted_len;
    int* l_accepted_map = match_getAcceptedIndex( match_matrix, l_accepted_len );

    cout << "l_len = " << l_len << endl;
    cout << "accepted = " << l_accepted_len << endl;

    show_distance
        <<<1,32>>>
        ( match_matrix,
          l_accepted_map,
          l_accepted_len,
          _l->getFeatures(),
          _l->getDescriptors(),
          _l->getReverseMap(),
          l_len,
          _r->getFeatures(),
          _r->getDescriptors(),
          _r->getReverseMap(),
          r_len );

    POP_SYNC_CHK;

    match_freeAcceptedIndex( l_accepted_map );
    match_freeMatchTable( match_matrix );
}

void BruteForceBlasMatcher::checkIdentity( ) const
{
    int l_len = _l->getFeatureCount( );
    int r_len = _r->getFeatureCount( );

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

    SortByCoords l_sbc( _l->getFeatures() );
    SortByCoords r_sbc( _r->getFeatures() );

    thrust::sort( l_vec.begin(), l_vec.end(), l_sbc );
    thrust::sort( r_vec.begin(), r_vec.end(), r_sbc );

    CompareFeatures cf( _l->getFeatures(), _r->getFeatures() );
    bool eq = thrust::equal( l_vec.begin(), l_vec.end(), r_vec.begin(), cf );

    if( not eq )
    {
        cout << "sorted features do not have the coordinates for identical images" << endl;
        return;
    }

    cout << "sorted features have the coordinates for identical images" << endl;
}

void BruteForceBlasMatcher::match_getAcceptedDescriptorMatches( thrust::device_vector<int2>& matches )
{
    int3* match_matrix = match_AllocMatchTable( );

    match_computeMatchTable( match_matrix );

    match_getAcceptedDescriptorMatchesFromMatrix( match_matrix, matches );

    match_freeMatchTable( match_matrix );
}

void BruteForceBlasMatcher::match_getAcceptedFeatureMatches( thrust::device_vector<int2>& matches )
{
    int3* match_matrix = match_AllocMatchTable( );

    match_computeMatchTable( match_matrix );

    match_getAcceptedFeatureMatchesFromMatrix(
            match_matrix,
            matches,
            _l->getReverseMap(),
            _r->getReverseMap() );

    match_freeMatchTable( match_matrix );
}

} // namespace popsift

