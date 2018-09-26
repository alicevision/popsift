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

#include <stdlib.h>
#include <errno.h>
#include <math_constants.h>

#include "features.h"
#include "sift_extremum.h"
#include "common/assist.h"
#include "common/debug_macros.h"

using namespace std;

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

