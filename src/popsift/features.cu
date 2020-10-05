/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/debug_macros.h"
#include "features.h"
#include "sift_extremum.h"

#include <math_constants.h>

#include <cerrno>
#include <cstdlib>
#include <iomanip>
#include <iostream>

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
    memalign_free( _ext );
    memalign_free( _ori );
}

void FeaturesHost::reset( int num_ext, int num_ori )
{
    if( _ext != nullptr ) { free( _ext ); _ext = nullptr; }
    if( _ori != nullptr ) { free( _ori ); _ori = nullptr; }

    _ext = (Feature*)memalign( getPageSize(), num_ext * sizeof(Feature) );
    if( _ext == nullptr ) {
        cerr << __FILE__ << ":" << __LINE__ << " Runtime error:" << endl
             << "    Failed to (re)allocate memory for downloading " << num_ext << " features" << endl;
        if( errno == EINVAL ) cerr << "    Alignment is not a power of two." << endl;
        if( errno == ENOMEM ) cerr << "    Not enough memory." << endl;
        exit( -1 );
    }
    _ori = (Descriptor*)memalign( getPageSize(), num_ori * sizeof(Descriptor) );
    if( _ori == nullptr ) {
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

void FeaturesHost::print( std::ostream& ostr, bool write_as_uchar, bool with_orientation ) const
{
    for( int i=0; i<size(); i++ ) {
        _ext[i].print( ostr, write_as_uchar, with_orientation );
    }
}

std::ostream& operator<<( std::ostream& ostr, const FeaturesHost& feature )
{
    feature.print( ostr, false, false );
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
    cudaFree( _ext );
    cudaFree( _ori );
    cudaFree( _rev );
}

void FeaturesDev::reset( int num_ext, int num_ori )
{
    if( _ext != nullptr ) { cudaFree( _ext ); _ext = nullptr; }
    if( _ori != nullptr ) { cudaFree( _ori ); _ori = nullptr; }
    if( _rev != nullptr ) { cudaFree( _rev ); _rev = nullptr; }

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

void Feature::print( std::ostream& ostr, bool write_as_uchar, bool with_orientation ) const
{
    float sigval =  1.0f / ( sigma * sigma );

    for( int ori=0; ori<num_ori; ori++ ) {
        if( with_orientation )
        {
            float dom_ori = orientation[ori];
            // dom_ori = dom_ori / M_PI2 * 360;
            // if (dom_ori < 0) dom_ori += 360;
            if (dom_ori < 0) dom_ori += M_PI2;

            ostr << std::setprecision(6)
                 << xpos << " " << ypos << " "
                 << sigma << " "
                 << dom_ori << " ";
        }
        else
        {
            ostr << std::setprecision(6)
                 << xpos << " " << ypos << " "
                 << sigval << " 0 "
                 << sigval << " ";
        }
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
    feature.print( ostr, false, false );
    return ostr;
}

} // namespace popsift
