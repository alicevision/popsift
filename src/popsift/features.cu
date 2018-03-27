/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <iomanip>
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#ifndef __APPLE__
#include <malloc.h>
#endif
#include <stdlib.h>
#include <errno.h>
#include <math_constants.h>
#include "features.h"
#include "sift_extremum.h"
#include "common/debug_macros.h"
#include "sift_conf.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include "lock.h"

using namespace std;

namespace popsift {
    

/*************************************************************
 * Features
 *************************************************************/

    Features::Features( )
	: _num_ext( 0 )
	, _num_ori( 0 )
    { }
    
    Features::~Features( )
    { }
    
/*************************************************************
 * HostFeatures
 *************************************************************/
    
    HostFeatures::HostFeatures( )
	: _ext( 0 )
	, _ori( 0 )
    { }

    HostFeatures::HostFeatures( int num_ext, int num_ori )
	: _ext( 0 )
	, _ori( 0 )
    {
	reset( num_ext, num_ori );
    }

    HostFeatures::~HostFeatures( )
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

    void HostFeatures::reset( int num_ext, int num_ori )
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

    void HostFeatures::pin( )
    {
	cudaError_t err;
	err = cudaHostRegister( _ext, getFeatureCount() * sizeof(Feature), 0 );
	if( err != cudaSuccess ) {
	    cerr << __FILE__ << ":" << __LINE__ << " Runtime warning:" << endl
		 << "    Failed to register feature memory in CUDA." << endl
		 << "    " << cudaGetErrorString(err) << endl;
	}
	err = cudaHostRegister( _ori, getDescriptorCount() * sizeof(Descriptor), 0 );
	if( err != cudaSuccess ) {
	    cerr << __FILE__ << ":" << __LINE__ << " Runtime warning:" << endl
		 << "    Failed to register descriptor memory in CUDA." << endl
		 << "    " << cudaGetErrorString(err) << endl;
	}
    }

    void HostFeatures::unpin( )
    {
	cudaHostUnregister( _ext );
	cudaHostUnregister( _ori );
    }

    void HostFeatures::print( std::ostream& ostr, bool write_as_uchar ) const
    {
	for( int i=0; i<size(); i++ ) {
	    _ext[i].print( ostr, write_as_uchar );
	}
    }

    std::ostream& operator<<( std::ostream& ostr, const HostFeatures& feature )
    {
	feature.print( ostr, false );
	return ostr;
    }

/*************************************************************
 * DeviceFeatures
 *************************************************************/

    DeviceFeatures::DeviceFeatures( )
	: _ext( 0 )
	, _ori( 0 )
	, _rev( 0 )
    { }

    DeviceFeatures::DeviceFeatures( int num_ext, int num_ori )
	: _ext( 0 )
	, _ori( 0 )
	, _rev( 0 )
    {
	reset( num_ext, num_ori );
    }

    DeviceFeatures::~DeviceFeatures( )
    {
	cudaFree( _ext );
	cudaFree( _ori );
	cudaFree( _rev );
    }

    void DeviceFeatures::reset( int num_ext, int num_ori )
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

	res += __shfl_down( res, 16 );
	res += __shfl_down( res,  8 );
	res += __shfl_down( res,  4 );
	res += __shfl_down( res,  2 );
	res += __shfl_down( res,  1 );

	return res;
    }
    __device__ inline float
    dot_l2_in_t0( const float4* lptr, const float4* rptr )
    {
	const float4  lval = lptr[threadIdx.x];
	const float4  rval = rptr[threadIdx.x];
	const float4  mval = make_float4( lval.x * rval.x,
					  lval.y * rval.y,
					  lval.z * rval.z,
					  lval.w * rval.w );
	float   res = mval.x
	    + mval.y
	    + mval.z
	    + mval.w;

    
	res += __shfl_down( res, 16 );
	res += __shfl_down( res,  8 );
	res += __shfl_down( res,  4 );
	res += __shfl_down( res,  2 );
	res += __shfl_down( res,  1 );
	return res;
    }
  
    __global__ void
    compute_distance_l2( int3* match_matrix, Descriptor* l, int l_len, Descriptor* r, int r_len )
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
    compute_distance_dot( int3* match_matrix, Descriptor* l, int l_len, Descriptor* r, int r_len )
    {
	if( blockIdx.x >= l_len ) return;
	const int idx = blockIdx.x;

	float match_1st_val = -1.0f;
	float match_2nd_val = -1.0f;
	int   match_1st_idx = 0;
	int   match_2nd_idx = 0;

  

	const float4* lptr = (const float4*)( &l[idx] );

	for( int i=0; i<r_len; i++ )
	{
	    const float4* rptr = (const float4*)( &r[i] );
	    const float   res  = dot_l2_in_t0( lptr, rptr );

	
	    if( threadIdx.x == 0 )
	    {
		if( res > match_1st_val )
		{
		    match_2nd_val = match_1st_val;
		    match_2nd_idx = match_1st_idx;
		    match_1st_val = res;
		    match_1st_idx = i;
		}
		else if( res > match_2nd_val )
		{
		    match_2nd_val = res;
		    match_2nd_idx = i;
		}
	    }

	    __syncthreads();	
	}
    
    
	const int one = __shfl(match_1st_idx, 0);
	const int two = __shfl(match_2nd_idx, 0);
  
	const float4* rptr = (const float4*)( &r[one] );
	const float res2 = l2_in_t0( lptr, rptr );
	const float4* rptr2 = (const float4*)( &r[two] );
	const float res3 = l2_in_t0( lptr, rptr2 );

	if( threadIdx.x == 0 )
	{
	    bool accept = (res2/res3 < 0.8f );
	    match_matrix[blockIdx.x] = make_int3( match_1st_idx, match_2nd_idx, accept );
	}
    }

    __global__ void
    compute_dot_in_section( int3* match_matrix, Descriptor* l, int l_len, Descriptor* r, int r_len, thrust::device_ptr<int> indexes, unsigned int *start_idx, unsigned int *stop_idx )
    {
	
	if( blockIdx.x >= l_len ) return; //redundant?
	const int idx = blockIdx.x;
	
	float match_1st_val = -1.0f;
	float match_2nd_val = -1.0f;
	int   match_1st_idx = 0;
	int   match_2nd_idx = 0;

  

	const float4* lptr = (const float4*)( &l[idx] );

	for( int i = start_idx[idx]; i< stop_idx[idx]; i++ )
	{
	    const float4* rptr = (const float4*)( &r[indexes[i]] );
	    const float   res  = dot_l2_in_t0( lptr, rptr );

	
	    if( threadIdx.x == 0 )
	    {
		if( res > match_1st_val )
		{
		    match_2nd_val = match_1st_val;
		    match_2nd_idx = match_1st_idx;
		    match_1st_val = res;
		    match_1st_idx = i;
		}
		else if( res > match_2nd_val )
		{
		    match_2nd_val = res;
		    match_2nd_idx = i;
		}
	    }

	    __syncthreads();	
	}
    
    
	const int one = __shfl(match_1st_idx, 0);
	const int two = __shfl(match_2nd_idx, 0);
  
	const float4* rptr = (const float4*)( &r[indexes[one]] );
	const float res2 = l2_in_t0( lptr, rptr );
	const float4* rptr2 = (const float4*)( &r[indexes[two]] );
	const float res3 = l2_in_t0( lptr, rptr2 );

	if( threadIdx.x == 0 )
	{
	    bool accept = (res2/res3 < 0.8f );
	    match_matrix[blockIdx.x] = make_int3( indexes[match_1st_idx], indexes[match_2nd_idx], accept );
	}
    }

    #define DESC_SEQ 4
        //16 bytes of concecutive memory (4 floats/ints)
    struct Desc 
    {
	//unsigned int descriptor[DESC_SEQ]; 
	float descriptor[DESC_SEQ]; //float makes a difference?
    };

    
    __device__
    unsigned int hamming_distance(unsigned int* A, unsigned int* B) //make const?
    {
	unsigned int g[4];
	unsigned int sum, sum_1, sum_2;

	g[0] = *A ^ *B;
	g[1] = *(A + 4) ^ *(B + 4);
	g[2] = *(A + 8) ^ *(B + 8);
	g[3] = *(A + 12) ^ *(B + 12);

	sum_1 = __popc(*g);
	sum_2 = __popc(*(g + 1));
	sum_1 += __popc(*(g + 2));
	sum_2 += __popc(*(g + 3));
	sum = sum_1 + sum_2;
	return sum;
    }

    __global__ void
    compute_distance_hamming( int3* match_matrix, Descriptor* l, Descriptor* l_tra, int l_len, Descriptor* r, Descriptor* r_tra, int r_len, thrust::device_ptr<int> indexes, unsigned int *start_idx, unsigned int *stop_idx )
    {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	
	const int idx = blockIdx.x;
        int offset = 2;
	
	//float match_1st_val = -1.0f;
	//float match_2nd_val = -1.0f;

        int match_1st_val = 128;
	int match_2nd_val = 128;
	int match_1st_idx = 0;
	int match_2nd_idx = 0;

	

	//while (tid < elements)
	//{
	//const float4* lptr = (const float4*)( &l_tra[idx] );

	//fix end value //another variable setting offset from a kernel in case of multilvl
	//could also have seperate stream for dot product..? based on depth and interval size
	if (start_idx[idx] == 0) offset = 3;
	__syncthreads;
	
        struct Desc *lptr = (struct Desc *)((&l_tra[indexes[idx]])) + offset;

	for( int i = start_idx[idx]; i< stop_idx[idx]; i++ )
	{
	    //const float4* rptr = (const float4*)( &r_[indexes[i]] );
	    const struct Desc *rptr = (struct Desc *)((&r_tra[indexes[idx]])) + offset;

		
	    //const float   res  = dot_l2_in_t0( lptr, rptr );
	    const int res = hamming_distance((unsigned int *)lptr, (unsigned int *)rptr);
	
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

	const int one = __shfl(match_1st_idx, 0);
	const int two = __shfl(match_2nd_idx, 0);


	float result_1 = 0.0f;
	float result_2 = 0.0f;
	//float diff0, diff1, diff2, diff3;
	float diff0 = 0.0f, diff1 = 0.0f, diff2 = 0.0f, diff3 = 0.0f;


	int i = 0;
	int last = 127 - 3;

	// Process 4 items with each loop for efficiency. helps on gpu at all?
	while (i < last) {
	    diff0 = l[indexes[idx]].features[i] - r[indexes[one]].features[i];
	    diff1 = l[indexes[idx]].features[i+1] - r[indexes[one]].features[i+1];
	    diff2 = l[indexes[idx]].features[i+2] - r[indexes[one]].features[i+2];
	    diff3 = l[indexes[idx]].features[i+3] - r[indexes[one]].features[i+3];
	    result_1 += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
	    i += 4;
	}

	i = 0;

	while (i < last) {
	    diff0 = l[indexes[idx]].features[i] - r[indexes[two]].features[i];
	    diff1 = l[indexes[idx]].features[i+1] - r[indexes[two]].features[i+1];
	    diff2 = l[indexes[idx]].features[i+2] - r[indexes[two]].features[i+2];
	    diff3 = l[indexes[idx]].features[i+3] - r[indexes[two]].features[i+3];
	    result_2 += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
	    i += 4;
	}
		
	if( threadIdx.x == 0 )
	{
	    bool accept = (result_1/result_2 < 0.8f );
	    match_matrix[blockIdx.x] = make_int3( indexes[match_1st_idx], indexes[match_2nd_idx], accept );
	}
	
	// tid += stride;
	//}
    }


     __global__ void
    compute_distance_hamming_levels( int3* match_matrix, Descriptor* l, Descriptor* l_tra, int l_len, Descriptor* r, Descriptor* r_tra, int r_len, thrust::device_ptr<int> indexes, unsigned int *start_idx, unsigned int *stop_idx )
    {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	
	const int idx = blockIdx.x;
        int offset = 2;
	
	//float match_1st_val = -1.0f;
	//float match_2nd_val = -1.0f;

        int match_1st_val_1 = 128;
	int match_2nd_val_1 = 128;
	int match_1st_val_2 = 128;
	int match_2nd_val_2 = 128;
	int match_1st_val_3 = 128;
	int match_2nd_val_3 = 128;
	int match_1st_val_4 = 128;
	int match_2nd_val_4 = 128;


	int match_1st_idx = 0;
	int match_2nd_idx = 0;

	

	//while (tid < elements)
	//{
	//const float4* lptr = (const float4*)( &l_tra[idx] );

	//fix end value //another variable setting offset from a kernel in case of multilvl
	//could also have seperate stream for dot product..? based on depth and interval size
	if (start_idx[idx] == 0) offset = 3;
	__syncthreads;
	
        struct Desc *lptr_1 = (struct Desc *)((&l_tra[indexes[idx]])) + offset;
	struct Desc *lptr_2 = (struct Desc *)((&l_tra[indexes[idx]])) + offset + 1;
        struct Desc *lptr_3 = (struct Desc *)((&l_tra[indexes[idx]])) + offset + 2;
	struct Desc *lptr_4 = (struct Desc *)((&l_tra[indexes[idx]])) + offset + 3;

			

	for( int i = start_idx[idx]; i< stop_idx[idx]; i++ )
	{
	    //const float4* rptr = (const float4*)( &r_[indexes[i]] );
	    const struct Desc *rptr_1 = (struct Desc *)((&r_tra[indexes[idx]])) + offset;
	    const struct Desc *rptr_2 = (struct Desc *)((&r_tra[indexes[idx]])) + offset + 1;
	    const struct Desc *rptr_3 = (struct Desc *)((&r_tra[indexes[idx]])) + offset + 2;
	    const struct Desc *rptr_4 = (struct Desc *)((&r_tra[indexes[idx]])) + offset + 3;

		
	    //const float   res  = dot_l2_in_t0( lptr, rptr );
	    const int res_1 = hamming_distance((unsigned int *)lptr_1, (unsigned int *)rptr_1);
	    const int res_2 = hamming_distance((unsigned int *)lptr_2, (unsigned int *)rptr_2);
	    const int res_3 = hamming_distance((unsigned int *)lptr_3, (unsigned int *)rptr_3);
	    const int res_4 = hamming_distance((unsigned int *)lptr_4, (unsigned int *)rptr_4);

	    if( threadIdx.x == 0 ) 
	    {
		
		int not_best = 1;
		if ( res_1 < match_1st_val_1 ) // first level shorter distance
		{
		    match_2nd_val_1 = match_1st_val_1;
		    match_2nd_val_2 = match_1st_val_2;
		    match_2nd_val_3 = match_1st_val_3;
		    match_2nd_val_4 = match_1st_val_4;

		    match_2nd_idx = match_1st_idx;
		    match_1st_idx = i;

		    match_1st_val_1 = res_1;
		    match_1st_val_2 = res_2;
		    match_1st_val_3 = res_3;
		    match_1st_val_4 = res_4;
		    not_best = 0;

		}
		else if ( res_1 == match_1st_val_1 ) // first level equal distance
		{
		    if ( res_2 < match_1st_val_2 ) // second level shorter distance
		    {
			match_2nd_val_1 = match_1st_val_1;
			match_2nd_val_2 = match_1st_val_2;
			match_2nd_val_3 = match_1st_val_3;
			match_2nd_val_4 = match_1st_val_4;
			
			match_2nd_idx = match_1st_idx;
			match_1st_idx = i;
			
			match_1st_val_1 = res_1; //since equal, not nessesary.. other places as well
			match_1st_val_2 = res_2;
			match_1st_val_3 = res_3;
			match_1st_val_4 = res_4;
		        not_best = 0;
			printf("res1: %d\t res2 %d\n", res_1, res_2);
					    
		    }
		    else if ( res_2 == match_1st_val_2 ) // second level equal distance
		    {
			if ( res_3 < match_1st_val_3 ) // third level shorter distance
			{
			    match_2nd_val_1 = match_1st_val_1; 
			    match_2nd_val_2 = match_1st_val_2; 
			    match_2nd_val_3 = match_1st_val_3;
			    match_2nd_val_4 = match_1st_val_4;
			
			    match_2nd_idx = match_1st_idx;
			    match_1st_idx = i;
			
			    match_1st_val_1 = res_1; //equal
			    match_1st_val_2 = res_2; //equal
			    match_1st_val_3 = res_3;
			    match_1st_val_4 = res_4;
			    not_best = 0;
			}
			else if ( res_3 == match_1st_val_3 ) //skip equal, go directly on next if statement?
			{
			    if ( res_4 < match_1st_val_4 ) // forth level shorter distance
			    {
				match_2nd_val_1 = match_1st_val_1; 
				match_2nd_val_2 = match_1st_val_2; 
				match_2nd_val_3 = match_1st_val_3;
				match_2nd_val_4 = match_1st_val_4;
			
				match_2nd_idx = match_1st_idx;
				match_1st_idx = i;
			
				match_1st_val_1 = res_1; //equal
				match_1st_val_2 = res_2; //equal
				match_1st_val_3 = res_3; //equal
				match_1st_val_4 = res_4;
			        not_best = 0;
			    }
			}
		    }
		}
		else if ( not_best == 1 ) // could find a better way to do this i think.. check for 0 instead? set 1 in an else maybe?
		{
		    if ( res_1 < match_2nd_val_1 )
		    {
			match_2nd_val_1 = res_1;
			match_2nd_val_2 = res_2;
			match_2nd_val_3 = res_3;
			match_2nd_val_4 = res_4;
			match_2nd_idx = i;
		    }
		    else if ( res_1 == match_2nd_val_1)
		    {
			if ( res_2 < match_2nd_val_2 )
			{
			    match_2nd_val_1 = res_1; //equal
			    match_2nd_val_2 = res_2;
			    match_2nd_val_3 = res_3;
			    match_2nd_val_4 = res_4;
			    match_2nd_idx = i;
			}
			else if ( res_2 == match_2nd_val_2)
			{
			    if ( res_3 < match_2nd_val_3 )
			    {
				match_2nd_val_1 = res_1; //equal
				match_2nd_val_2 = res_2; //equal
				match_2nd_val_3 = res_3;
				match_2nd_val_4 = res_4;
				match_2nd_idx = i;
			    }
			    else if ( res_3 == match_2nd_val_3)
			    {
				if ( res_4 < match_2nd_val_4 )
				{
				    match_2nd_val_1 = res_1; //equal
				    match_2nd_val_2 = res_2; //equal
				    match_2nd_val_3 = res_3; //equal
				    match_2nd_val_4 = res_4;
				    match_2nd_idx = i;
				}
			    }
			}
		    }
		}
	    }

	    __syncthreads();	
	}

	const int one = __shfl(match_1st_idx, 0);
	const int two = __shfl(match_2nd_idx, 0);


	float result_1 = 0.0f;
	float result_2 = 0.0f;
	//float diff0, diff1, diff2, diff3;
	float diff0 = 0.0f, diff1 = 0.0f, diff2 = 0.0f, diff3 = 0.0f;


	int i = 0;
	int last = 127 - 3;

	// Process 4 items with each loop for efficiency. helps on gpu at all?
	while (i < last) {
	    diff0 = l[indexes[idx]].features[i] - r[indexes[one]].features[i];
	    diff1 = l[indexes[idx]].features[i+1] - r[indexes[one]].features[i+1];
	    diff2 = l[indexes[idx]].features[i+2] - r[indexes[one]].features[i+2];
	    diff3 = l[indexes[idx]].features[i+3] - r[indexes[one]].features[i+3];
	    result_1 += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
	    i += 4;
	}

	i = 0;

	while (i < last) {
	    diff0 = l[indexes[idx]].features[i] - r[indexes[two]].features[i];
	    diff1 = l[indexes[idx]].features[i+1] - r[indexes[two]].features[i+1];
	    diff2 = l[indexes[idx]].features[i+2] - r[indexes[two]].features[i+2];
	    diff3 = l[indexes[idx]].features[i+3] - r[indexes[two]].features[i+3];
	    result_2 += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
	    i += 4;
	}
		
	if( threadIdx.x == 0 )
	{
	    bool accept = (result_1/result_2 < 0.8f );
	    match_matrix[blockIdx.x] = make_int3( indexes[match_1st_idx], indexes[match_2nd_idx], accept );
	}
	
	// tid += stride;
	//}
    }


    __host__ __device__ void
    printBits( unsigned int num )
    {
        for ( int bit = 0; bit < 32; bit++ )
	{
	    printf("%i", num & 0x01);
	    num = num >> 1;
	}
    }

  
    __host__ __device__ void
    printFeature( unsigned int *num )
    {
        for ( int i = 0; i < 128; i += 4 ) {
            for (int j = 0; j < 4; j++) {
		printBits(num[ i + j]);
		printf( " " );
            }
            
            printf( "\n" ); 
        }
        
	printf( "\n\n" );
    }

    __device__ void
    print32x32( unsigned int *num )
    {
        for ( int i = 0; i < 32; i++ ) {
            printBits(num[i]);
            printf( "\n" ); 
        }
        
        printf( "\n\n" );
    }



/*****************************
HASH TABLE - fix seperate file.
******************************/
    

#define HASH_ENTRIES     1024 //increase


/*
 * struct: Entry
 * --------------
 * Table entry for hash table
 * 
 * Each entry holds: 
 * Key: a 128 bit significanse sequence of a discriptor, 
 * Value: an interval of indexes desided by begin and end
 * Next: Null or pointer to the next entry within this 'bucket'. 
 */
    struct Entry
    {
	struct Desc key;
	unsigned int begin;
	unsigned int end; 
	Entry *next = NULL; 
    };


/*
 * struct: Table
 * --------------
 * Hash table
 * 
 * Count: Number of entries in our table.
 * Entries: List of entries. Each address 
 * here is a pointer to an entry. 
 * Pool: Unused entries. Pre allocated.
 */
    struct Table
    {
	size_t count;
	Entry   **entries;
	Entry   *pool; 
    };

    
/*
 * struct: Inner_Table
 * -------------------
 * Hash table within each entry of main hash table
 * 
 * Count: Number of entries in our table.
 * Entries: List of entries. Each address 
 * here is a pointer to an entry. 
 * Pool: Unused entries. Pre allocated.
 */
    struct Inner_Table
    {
	size_t count;
	Entry   **entries;
	Entry   *pool; 
    };

    struct bloom_filter 
    {
	uint8_t *bits;
	size_t size;
    };


    __host__ __device__
    unsigned int djb2(const void *_str)
    {
	const char *str = (const char *)_str;
	unsigned int hash = 5381;
	char c, i = 0;
	while ((i < 16))
	{
	    c = str[i];
	    hash = ((hash << 5) + hash) + c;
	    i++;
	}
	return hash;
    }

    __host__ __device__
    unsigned int jenkins(const void *_str)
    {
	const char *key = (const char *)_str;
	unsigned int hash, i = 0;
	while (i < 16)
	{
	    hash += *key;
	    hash += (hash << 10);
	    hash ^= (hash >> 6);
	    key++;
	    i++;
	}
	hash += (hash << 3);
	hash ^= (hash >> 11);
	hash += (hash << 15);
	return hash;
    }
    
/*
 * Could create a function pointer in the table, and pass different hash 
 * functions for a better testing environment 
 */
    __device__ __host__
    size_t hash(unsigned int * key, size_t count )
    {
	int i = 0;
	size_t sum = 0;
	unsigned char * p  = (unsigned char *)key;

	while (i < 16)
	{
	    sum += p[i];
	    sum += (sum << 10);
	    sum ^= (sum >> 6);
	    i++;
	}

	sum += (sum << 3);
	sum ^= (sum >> 11);
	sum += (sum << 15);

	return sum % count;
    }

    __device__ __host__
    size_t hash2(unsigned int * key, size_t count )
    {
	int i = 0;
	char c;
	size_t sum = 5381;
	unsigned char * p  = (unsigned char *)key;

	while (i < 16)
	{
	    c = p[i];
	    sum = ((sum << 5) + sum) + c;
	    i++;
	}

	return sum % count;
    }


    void initialize_table( Table &table, int entries, int elements )
    {
	printf("init: entries: %d\t elements: %d\n", entries, elements);
	table.count = entries;
	cudaMalloc( (void**)&table.entries, entries * sizeof(Entry*) );
	cudaMemset( table.entries, 0, entries * sizeof(Entry*) );
	cudaMalloc( (void**)&table.pool, elements *sizeof(Entry) ); 
    }


    void free_table( Table &table )
    {
	cudaFree( table.pool );
	cudaFree( table.entries ); 
    }


    void copy_table_to_host( const Table &table, Table &hostTable, unsigned int elements )
    {
	hostTable.count = table.count;
	hostTable.entries = (Entry**)calloc( table.count, sizeof(Entry*) );
	hostTable.pool = (Entry*)malloc( elements * sizeof( Entry ) );
    
	cudaMemcpy( hostTable.entries, table.entries, table.count * sizeof(Entry*), cudaMemcpyDeviceToHost );
	cudaMemcpy( hostTable.pool, table.pool, elements * sizeof( Entry ), cudaMemcpyDeviceToHost );

    
	for (int i=0; i<table.count; i++)
	{
	    if (hostTable.entries[i] != NULL)
		hostTable.entries[i] = (Entry*)((size_t)hostTable.entries[i] - (size_t)table.pool + (size_t)hostTable.pool);
	}
    

	for ( int i=0; i < elements; i++)
	{
	    if (hostTable.pool[i].next != NULL)
		hostTable.pool[i].next = (Entry*)((size_t)hostTable.pool[i].next - (size_t)table.pool + (size_t)hostTable.pool);
	}

    }

    void verify_table( const Table &dev_table, unsigned int elements )
    {
	Table table;
	copy_table_to_host( dev_table, table, elements );
	int count = 0;


	for (size_t i=0; i<table.count; i++)
	{
	    Entry   *current = table.entries[i];
	    //printf("entry ptr %ld\n",  table.entries[i]);

	    while (current != NULL)
	    {
		if (current->end - current->begin > 1)
		    printf("begin %d\t end %d\t table %d\n",  current->begin, current->end, i);
		++count;
		if (hash((unsigned int *)&(current->key), table.count ) != i)
		    printf("begin %d end %d hashed to %ld, but was located "
			   "at %ld\n",
			   current->begin, current->end,
			   hash((unsigned int *)&(current->key), table.count), i ); // *(unsigned int *)*/
		current = current->next;

	    }
	}

	if (count != elements)
	    printf( "%d elements found in hash table.  Should be %d\t missing are likely ignored duplicates\n", count, elements );
	else
	    printf( "All %d elements found in hash table.\n", count );
	free( table.pool );
	free( table.entries ); 
    }

/*
  __device__ //This must be fixed.. //might have been pointer issue.. //pointer from cpu?
  int compareKey(struct Desc *A, struct Desc *B)
  {
  int i = 0;
  while ( i < DESC_SEQ && A->descriptor[i] == B->descriptor[i] ) i++;
  if (i == 4)	{ return -1;}
  return 1;
  }
*/

    /*  __device__ //This must be fixed.. //might have been pointer issue.. //pointer from cpu?
	int compareKey(unsigned int *A, unsigned int *B)
	{
	int i = 0;
	while ( i < DESC_SEQ && A[i] == B[i] )
	{
	i++;
	}
	//printf("cmp: %d %d\n", A[i-1], B[i-1]);
	if (i == 4)	{  return -1;}
	return 1;
	}
    */
    
    __device__ //compare with hamming dinstance here? might be good
    int compareKey(unsigned char *A, unsigned char *B)
    {
	int i = 0;
	while ( i < DESC_SEQ * DESC_SEQ && A[i] == B[i] )
	{
	    i++;
	}

	//printf("i: %d\n", i); 
	//printf("cmp: %d %d\n", A[i-1], B[i-1]);
	if (i == 16)	{  return -1;}
	return 1;
    }

    __device__
    unsigned int bloom_check( uint8_t * bits,  struct Desc *key, unsigned int size) 
    {
	const unsigned int hashkey_1 = hash((unsigned int *)key, size);
	if (bits[hashkey_1] != 1) return 0;
	const unsigned int hashkey_2 = hash2((unsigned int *)key, size);
	if (bits[hashkey_2] != 1) return 0;
	return 1;
    }



//we need a check for equal key
    //might be possible to do this in some sort of log n format due to sorted keys
    __global__ void
    add_to_table( struct Descriptor *keys, thrust::device_ptr<int> values, Table table, Lock *lock, unsigned int elements)
    {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	//struct Desc *keys
	//   float * destination = (float*)(des[block].features);
	//
    
	while (tid < elements)
	{
	    //cast so we only use first 16bytes, we use the indirect lookup sorted list to find
	    //corresponding key value pair... values is host vector? error //pointer caster minnet til pointer?

	
	    struct Desc *key = (struct Desc *)((&keys[values[tid]])) + 2; //I think this works, all zero's in first two layers though.. expected.. skip them?


//struct Desc *key1 = (struct Desc *)((&keys[values[tid]])) + 3;
	    //struct Desc *key2 = (struct Desc *)((&keys[values[tid]])) + 5;
	    //struct Descriptor tmp = keys[values[tid]];

	    /*if (threadIdx.x == 0 && blockIdx.x == 0) { 
	      printf("key0: %d\n", *key);
	      printf("key1: %d\n", *key1);
	      printf("key2: %d\n", *key2);
		
	      //printFeature((unsigned int *)&key->descriptor[0]);
	      }*/
	
	    //printf("key: %p\n", &keys[values[tid]]);
	    //printf("key: %.4f\n", key.descriptor[3]);

	    size_t hashValue = hash((unsigned int *)key, table.count ); //cast to unsigned char?

	    //printf("hashval: %ld\n", hashValue);
	    //printf("key: %d\t %d\t %d\t %d\t\n", key[0], key[1], key[2], key[3]);

	    unsigned int *w = (unsigned int *)key;

//if (hashValue == 1023)
	    //{
	    //printf("key: %d\t %d\t %d\t %d\t hashval: %ld\n", w[0], w[1], w[2], w[3], hashValue);
	    //}
	
	
	
	    for (int i=0; i<32; i++)
	    {
		if ((tid % 32) == i)
		{
		    Entry *location = &(table.pool[tid]);
		    memcpy(&(location->key), key, sizeof(struct Desc));
		    //location->value = values[tid];
		    location->begin = tid; //values[tid]?
		    location->end = tid + 1;
		    lock[hashValue].lock();

		
		    Entry *ptr = table.entries[hashValue];
		    int exists = 1;
		    while (ptr != NULL)
		    {
			//exists = compareKey((unsigned int *)&(ptr->key), (unsigned int *)&(location->key)); //pretty sure this does not work as intended..
			exists = compareKey((unsigned char *)&(ptr->key), (unsigned char *)&(location->key));
			if (exists == -1) break;
			ptr = ptr->next;
		    }
		
		    if (exists == 1)
		    {
			location->next = table.entries[hashValue];
			table.entries[hashValue] = location;
		    }
		    else
		    {
			//printf("exists\n");
			//ptr->end++; //begin / end ---- how am i sure to increase end, not decrease begin? needs fixing..

			/******************************************************
			 *
			 * This solution did not work as expected...huge overlapping sectors..
			 * Is this because of some mistake in the sorting? might be. 
			 * Could also just be some fallacies in the code...
			 *
			 * If mistake in sort, get around this by storing every index,  
			 * creating an interable list? -- SOLVED. Did not get sector from sorted lookup. 
			 *
			 ********************************************************/
			if (location->begin < ptr->begin) ptr->begin = location->begin;
			if (location->end > ptr->end) ptr->end = location->end;

			//if (location->begin < ptr->begin) ptr->begin--;
			//if (location->end > ptr->end) ptr->end++;
		    
		    
		    }
		
		    lock[hashValue].unlock();
		}
	    }
	
	    tid += stride;
	} 
    }



    __global__ void
    get_section_from_table( Table table, struct Descriptor *keys, unsigned int elements, unsigned int l_len, unsigned int *start_idx, unsigned int *stop_idx )
    {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	int check;
	//Search area - set to max if no key match is found - how do i return these in a good way?
	//two unsigned int arrays allocated on the gpu?
	
	while (tid < elements)
	{
	    struct Desc *key = (struct Desc *)(&keys[tid]) + 2; //+2 because second layer is currently stored.
	    // bloom_filter bloom,
	    //check = bloom_check(bloom.bits, key, bloom.size);
	    //if (check == 0)
	    //{
	    //	start_idx[tid] = 0;
	    //	stop_idx[tid] = l_len;
	    //	tid += stride;
	    //	continue;
	    //}
	    
	    size_t hashValue = hash((unsigned int *)key, table.count );

	    Entry *ptr = table.entries[hashValue];
	    int exists = 1;
	    int cnt = 0;
	    //This while loop might not be very fast..splits inside warp if threads
	    while (ptr != NULL)
	    {
		exists = compareKey((unsigned char *)&(ptr->key), (unsigned char *)key);
		if (exists == -1) break;
		ptr = ptr->next;
	    }

	    if (exists == -1 && (ptr->end - ptr->begin) > 1) //must be two or more in set
	    {
		start_idx[tid] = ptr->begin;
		stop_idx[tid] = ptr->end;
	    }
	    else
	    {
		start_idx[tid] = 0;
		stop_idx[tid] = l_len;

		//if (tid == 0)
		//{
		//unsigned char * p1 = (unsigned char *)key;
		//printf("key:\t");
		//printf(" %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n", p1[0], p1[1], p1[2], p1[3], p1[4], p1[5], p1[6], p1[7], p1[8], p1[9], p1[10], p1[11], p1[12], p1[13], p1[14], p1[15]);
		
		//}
	    }
	
	    tid += stride;
	}
    }

/********************************
HASH TABLE END - fix seperate file.
**********************************/




/*****************************
BLOOM FILTER
****************************/



    void initialize_bloom_filter( bloom_filter &bloom, size_t size )
    {
	bloom.size = size;
	cudaMalloc( (void**)&bloom.bits, size );
	cudaMemset( bloom.bits, 0, size ); 
    }



//bytewise bloomfilter
    __device__
    void bloom_add( bloom_filter &filter, const void *item ) 
    {
	uint8_t *bits = (uint8_t *)filter.bits;

	unsigned int hash = jenkins(item);
	printf("hash: %d\n", hash);
	hash %= filter.size;
	printf("hash MOD: %d\n", hash);
	bits[hash] = 1;
	printf("hash/8: %d\n", hash);

	hash = djb2(item);
    }



    __global__ void
    bloom_add_filters( bloom_filter bloom,  struct Descriptor *keys, unsigned int elements)
    {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	uint8_t *bits = (uint8_t *)bloom.bits;
	unsigned int hashkey;
	struct Desc *key;
	while (tid < elements)
	{
	    key = (struct Desc *)((&keys[tid])) + 2;
	    hashkey = hash((unsigned int *)key, bloom.size);
	    bits[hashkey] = 1;
	    hashkey = hash2((unsigned int *)key, bloom.size);
	    bits[hashkey] = 1;
	    tid += stride;
	}
    }

    //even if hit we do not know if we have  two or more in the set... pointless?
    //cant see imediate benefit.
    __global__ void
    bloom_filter_check( bloom_filter bloom,  struct Descriptor *keys, unsigned int elements)
    {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	uint8_t *bits = (uint8_t *)bloom.bits;
	unsigned int hashkey_1;
	unsigned int hashkey_2;

	int check = 1;
	struct Desc *key;
	while (tid < elements)
	{
	    key = (struct Desc *)((&keys[tid])) + 2;
	    hashkey_1 = hash((unsigned int *)key, bloom.size);
	    if (bits[hashkey_1] != 1)
		check = 0;
	    hashkey_2 = hash2((unsigned int *)key, bloom.size);
	    if (bits[hashkey_2] != 1)
		check = 0;
	    if (tid < elements)
		printf("bloom: %d\n", check);
	    tid += stride;
	}
    }



/*****************************
BLOOM FILTER end
****************************/
    

    

    
    __device__ void
    transpose32(unsigned int *A) {
	int j, k;
        unsigned m, t;
        
        m = 0x0000FFFF;
        for (j = 16; j != 0; j = j >> 1, m = m ^ (m << j)) {
            for (k = 0; k < 32; k = (k + j + 1) & ~j) {
                t = (A[k] ^ (A[k+j] >> j)) & m;
                A[k] = A[k] ^ t;
                A[k+j] = A[k+j] ^ (t << j);
            }
        }
    }

        __device__ void
    transpose8rS64( unsigned char* A, unsigned char* B ) 
    {
    	unsigned long long x, t;
    	int i;

	for ( i = 0; i <= 7; i++ )     // Load 8 bytes from the
		x = x << 8 | A[1*i];      // input array and pack
								  // them into x.

	t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AALL;
	x = x ^ t ^ (t << 7);
	t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCLL;
	x = x ^ t ^ (t << 14);
	t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0LL;
	x = x ^ t ^ (t << 28);

	for ( i = 7; i >= 0; i-- ) 
	{   // Store result into
		B[1*i] = x; x = x >> 8;
	}  // output array B.
    }



    __device__ void
    organize( unsigned int* A, unsigned int* B )
    {
	int i, j;
	int cnt = 0;
	for (j = 0; j < 32; j++)
	    for ( i = 0; i < 32 * 4; i += 32 )
	    {
		B[cnt] = A[i + j];
		cnt++;   
	    }
    }
    
    __device__ void
    organize_32( float* A, float* B )
    {
        int i = threadIdx.x;
        int cnt = threadIdx.x * 4;
        for (int j = 0; j < 128; j +=32)
	{
	    B[cnt] = A[i + j];
	    cnt++;   
	}
    }

       
    __device__ void
    organize_A( unsigned int* A, unsigned int* B )
    {
        for (int j = 0; j < 128; j++)
	{
	    B[j] = A[j];	
	}
    }
	
  
    
    __device__ void
    transpose(Descriptor * src, Descriptor *des, int size) {
              
        int block = blockIdx.x;
	int thread = threadIdx.x;

	const float * source = (float*)(src[block].features);       	
        float * destination = (float*)(des[block].features);

	    
        int s = thread * 4;
        int i;

	__shared__ float T[128];
	
	for (i = s; i < s + 4; i++)
	    T[i] = source[i];
	    
	
	__syncthreads();

	 
	//if(block == 0 && thread == 0) 
	//    printFeature((unsigned int*)T);

	 
	if (thread < 4)
	    transpose32((unsigned int*)&T[32 * thread]);     	    
       
	__syncthreads();
	 
	 
	organize_32(T, destination);
	 
	//if(thread == 0 && block == 0)
	//printFeature((unsigned int*)destination);	 	 
	 
	__syncthreads();

       
    }

/*
  __global__ void
  compute_distance_transposed_hamming( int3* match_matrix, Descriptor * l, int l_len, Descriptor* r, int r_len , Descriptor * l_tra, Descriptor *r_tra) {

  if(blockIdx.x > l_len)
  return;

  transpose(l, l_tra, l_len);
	
  //if(blockIdx.x == 0 && threadIdx.x == 0)
  //    printFeature((unsigned int*)l_tra[blockIdx.x].features);		       
	
	
  }*/
    
#define DIMENSIONS 128

    __device__ __constant__ unsigned int gpu_idx[64] =
{
	0, 1, 2, 3,
	4, 5, 6, 7,
	8, 9, 10, 11,
	12, 13, 14, 15,
	128, 129, 130, 131,
	132, 133, 134, 135,
	136, 137, 138, 139,
	140, 141, 142, 143,
	256, 257, 258, 259,
	260, 261, 262, 263,
	264, 265, 266, 267,
	268, 269, 270, 271,
	384, 385, 386, 387,
	388, 389, 390, 391,
	392, 393, 394, 395,
	396, 397, 398, 399,
};


__device__ __constant__ unsigned int gpu_write_back[64] =
{
	384, 256, 128, 0, 
	388, 260, 132, 4, 
	392, 264, 136, 8, 
	396, 268, 140, 12, 
	385, 257, 129, 1, 
	389, 261, 133, 5, 
	393, 365, 137, 9, 
	397, 269, 141, 13, 
	386, 258, 130, 2, 
	390, 262, 134, 6, 
	394, 266, 138, 10, 
	398, 270, 142, 14, 
	387, 259, 131, 3, 
	391, 263, 135, 7, 
	395, 267, 139, 11, 
	399, 271, 143, 15,

};
    
    __global__ void
    transpose_descriptors_64(Descriptor *src, Descriptor *des)
    {
    
	unsigned char *ptr = (unsigned char *)(src + blockIdx.x);
	unsigned char *ptr_res = (unsigned char *)(des + blockIdx.x);

	int i;
	int start_pos, end_pos;
	int offset = 0;
	unsigned char C[8];                        //Local8x8 src
	unsigned char R[8];                        //Local8x8 des

	start_pos = gpu_idx[threadIdx.x];          //get starting index
	end_pos = gpu_write_back[threadIdx.x];     //get ending index
	ptr += start_pos;                          //set starting index
	ptr_res += end_pos;                        //set write back position

	//prepare 8x8 blocks for transpose
	for (i = 0; i < 8; i++) {
	    C[i] = ptr[offset];
	    offset += 16;
	} 

	transpose8rS64(C, R);

	offset = 0;
	for (i = 0; i < 8; i++) {
	    ptr_res[offset] = R[i];
	    offset += 16;
	} 
    }
    
    __global__ void
    transpose_descriptors(Descriptor * src, int len, Descriptor * des) {

        if(blockIdx.x > len)
            return;

        transpose(src, des, len);
       
    }

    __global__ void
    compute_distance_print( int3* match_matrix, Descriptor* l, int l_len, Descriptor* r, int r_len , Descriptor * l_tra, Descriptor *r_tra) {
	printf("address: %d\n", l_tra);


	for(int i = 0; i < 4; i++) {
	    for(int j = 0; j < 10; j++)
		printf("%u\t", l_tra[i].features[j]);
	    printf("\n");
	}
	
	printf("-------\n");	
    }


    //a bit-wise IEEE floating-point standard single
    //precision (32-bit) NaN would be:
    //s111 1111 1xxx xxxx xxxx xxxx xxxx xxxx
    //where s is the sign 
    struct compare_descriptors { //fix for
	// template <typename T>
	__host__ __device__
	// bool operator()(const T &l, const T &r) const {
	int operator()(const Descriptor &l, const Descriptor &r) const {
	    //for(int i = 0; i < 128; i++) {

	    unsigned char *a, *b;
	    a = (unsigned char*)l.features;
	    b = (unsigned char*)r.features;

	    
	    a = (unsigned char*)&l;
	    b = (unsigned char*)&r;
	    
	    int i = 0;

	    //i-1 is used, should be + 1?
	    while(i < 512) {
			    
		if(a[i] < b[i]) {
		    //printf("%d\t%d\n", a[i], b[i]);
		    return 2;
		}
			    
		if(a[i] > b[i]) {
		    //printf("%d\t%d\n", a[i], b[i]);				
		    return 0;
		}
		
		if(a[i+1] < b[i+1])
		    return 2;

		if(a[i+1] > b[i+1])
		    return 0;

		
		if(a[i+2] < b[i+2])
		    return 2;

		if(a[i+2] > b[i+2])
		    return 0;

		
		if(a[i+3] < b[i+3])
		    return 2;

		if(a[i+3] > b[i+3])
		    return 0;
		i+=4;
		//printf("%d\n", a[i]);
		
		/*
		  if((unsigned int)l.features[i] < (unsigned int)r.features[i]) {
			
		  return true;
		  }
		  if((unsigned int)l.features[i] > (unsigned int)r.features[i])
		  return false;
		  i++;
		*/
	    }
	    return 1;
	    

	    /*
	      if(l.features[0] > r.features[0])
	      return true;
	      if(l.features[0] < r.features[0])
	      return false;
	      if(l.features[1] > r.features[1])
	      return true;
	      if(l.features[1] < r.features[1])
	      return false;
	      if(l.features[2] > r.features[2])
	      return true;
	    
	      return false;
	    */
	}
    };

    struct ILookup
    {
	__device__
	inline bool operator()( int a, int b ) const
	    {
		return true; // a < b;
	    }
    };

    struct IndirectLookup
    {
	Descriptor* base;
	
	IndirectLookup( Descriptor* b ) : base(b) {}
	__device__
	inline bool operator()( int a, int b ) const
	    {
		int x = compare_descriptors()(base[a], base[b]);//operator()( base[a], base[b] );
                switch(x)
                {
                case 0 : return false;
                case 2 : return true;
                }
                return ( a < b );
	    }
    };

    struct char_array {
	float v[128];
	//float w;
    };

    struct cmp_test2 {
	template<typename T>
	__host__ __device__
	bool operator()(const T &a, const T &b) const {
	    	    
	    /*
	      for(int i = 0; i <= 1; i++) {
	      if(a.v[i] < b.v[i])
	      return true;
	      if(b.v[i] > a.v[i])
	      return false;
	      }	    
	      return false;
	    */

	    int i = 0;

	    while(i < 128) {
		if(a.v[i] < b.v[i])
		    return true;
		if(b.v[i] < a.v[i])
		    return false;
		i++;
	    }
	    //if(a.v[1] < b.v[1])
	    //return true;
	    
	    return false;
	}
    };

    struct cmp_test {
	char_array *base;
	cmp_test(char_array *b) : base(b){}
	__host__ __device__
	inline bool operator() (int a, int b) const {
	    return cmp_test2()(base[a], base[b]);
	}
    };

    
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
	int counter = 0;
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
		    counter++;
		/*printf( "accept feat %4d [%4d] matches feat %4d [%4d] ( 2nd feat %4d [%4d] ) dist %.3f vs %.3f\n",
		  l_fem[i], i,
		  r_fem[match_matrix[i].x], match_matrix[i].x,
		  r_fem[match_matrix[i].y], match_matrix[i].y,
		  d1, d2 );*/
	  
		//else
		/*printf( "reject feat %4d [%4d] matches feat %4d [%4d] ( 2nd feat %4d [%4d] ) dist %.3f vs %.3f\n",
		  l_fem[i], i,
		  r_fem[match_matrix[i].x], match_matrix[i].x,
		  r_fem[match_matrix[i].y], match_matrix[i].y,
		  d1, d2 );*/
	    }
	
	    __syncthreads();
      
	}
	if( threadIdx.x == 0 )
	    printf("Matches: %d\n", counter);
  
    }

    //__host__ sometimes error?
    Descriptor * gpu_init(int SIZE) {
	Descriptor *tmp;

	cudaError_t err = cudaMalloc((void **)&tmp, SIZE * sizeof(Descriptor));
	if(err != cudaSuccess)
	    printf("%s\n", cudaGetErrorString(err));
//	cudaMemset(tmp, 0, SIZE*sizeof(Descriptor));  //no function??
	
	return tmp;
    }



    
    void DeviceFeatures::match( DeviceFeatures* other, const popsift::Config& config )
    {

	int l_len = getDescriptorCount( );
	int r_len = other->getDescriptorCount( );
        POP_CHK;

	//cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1000000);
        POP_CHK;

	int3* match_matrix = popsift::cuda::malloc_devT<int3>( l_len, __FILE__, __LINE__ );    
        POP_CHK;
    
	dim3 grid;
	grid.x = l_len;
	grid.y = 1;
	grid.z = 1;
	dim3 block;
	block.x = 32;
	block.y = 1;
	block.z = 1;

	if ( config.getModeMatching() == popsift::Config::l2 )
	{
	    compute_distance_l2
		<<<grid,block>>>
		( match_matrix, getDescriptors(), l_len, other->getDescriptors(), r_len );
	}
	else if ( config.getModeMatching() == popsift::Config::dot )
	{
	    compute_distance_dot
		<<<grid,block>>>
		( match_matrix, getDescriptors(), l_len, other->getDescriptors(), r_len );
	}
	else
	{



	    dim3 grid_r;
	    grid.x = r_len;
	    grid.y = 1;
	    grid.z = 1;

	    //transpose first set of descritors

	    //sort the transposed descriptors

	    //transpose and compare with the second set
	
	
	    Descriptor *l_copy = gpu_init(l_len);
	    Descriptor *r_copy = gpu_init(r_len);

	    //TRANSPOSE
            POP_CHK;
	    //compute_distance_transposed_hamming
	    //<<<grid,block>>>
	    //	( match_matrix, getDescriptors(), l_len, other->getDescriptors(), r_len , l_copy, r_copy);

	    //two streams..
	    cudaStream_t stream1, stream2;
	    cudaStreamCreate( &stream1 );
	    cudaStreamCreate( &stream2 );

	    cudaEvent_t start, stop;
	    cudaEventCreate( &start );
	    cudaEventCreate( &stop );
	    float elapsedTime;
	    
	    cudaEventRecord( start, 0 );
/*
	    transpose_descriptors
		<<<l_len,block>>>
		( getDescriptors(), l_len, l_copy);

	    POP_CHK;

	    transpose_descriptors
		<<<r_len,block>>>
		( other->getDescriptors(), r_len , r_copy);
*/
	    /*
	    transpose_descriptors_64
		<<<l_len,64,0,stream1>>>
		( getDescriptors(), l_copy);

	    transpose_descriptors_64
		<<<r_len,64,0,stream2>>>
		( other->getDescriptors(), r_copy);
	    */
	    	    transpose_descriptors_64
		<<<l_len,64,0>>>
		( getDescriptors(), l_copy);

	    transpose_descriptors_64
		<<<r_len,64>>>
		( other->getDescriptors(), r_copy);
	    
	    cudaEventRecord( stop, 0 );
	    cudaEventSynchronize( stop );
	    cudaEventElapsedTime( &elapsedTime, start, stop );
	    printf( "Time to transpose:  %3.1f ms\n", elapsedTime );
	    
	    cudaDeviceSynchronize();
            POP_CHK;
	    
	    const int SIZE = r_len;
	  

	    //thrust::device_ptr<int> off_ptr = thrust::device_pointer_cast();


	    /*thrust::device_ptr<Descriptor> d_ptr(l_copy);
	      thrust::device_vector<int> B(SIZE);
	      thrust::sequence(B.begin(), B.end());
	      thrust::sort(B.begin(), B.end(), IndirectLookup(l_copy));
	      thrust::host_vector<int> H = B;
	      thrust::copy(H.begin(), H.end(), std::ostream_iterator<int>(std::cout, " "));
	      int cnter = 0;
	      for(int i = 0; i < SIZE; i++) {
	      for(int j = 0; j < SIZE; j++) {
	      if(i != j) {
	      if (H[j] == H[i]) {
	      cnter++;
	      }
	      }
		    
	      }
	      }
	      std::cout << "l_len: " <<  SIZE << " cnt: " << cnter << std::endl;
	      std::exit(1);
	    */
	    
#if 0
	    int *desc_index = popsift::cuda::malloc_devT<int>(SIZE, __FILE__, __LINE__ );
            POP_CHK;
	    //thrust::sequence(thrust::device, desc_index, desc_index+SIZE);
	    	   
	    thrust::device_ptr<int> d = thrust::device_pointer_cast(desc_index);	    
	    thrust::sequence(d, d+SIZE);
            thrust::host_vector<int> hsorted( SIZE );
            thrust::copy( d, d+SIZE, hsorted.begin() );
            thrust::sort( hsorted.begin(), hsorted.end() );
            thrust::copy( hsorted.begin(), hsorted.end(), std::ostream_iterator<int>(std::cout, " ") );
            cout << endl;
	    
	    int *t1 = thrust::raw_pointer_cast(d);

	    //int c[SIZE];
	    //cudaMemcpy(c, t1, SIZE*sizeof(int), cudaMemcpyDeviceToHost);


	    //thrust::sort(thrust::device, desc_index, desc_index+SIZE, IndirectLookup(l_copy));
	    
	    int b[SIZE];

	    cudaMemcpy(
		b,
	        t1,
		SIZE*sizeof(int),
		cudaMemcpyDeviceToHost);


	    //thrust::sort(b, b+SIZE, thrust::greater<int>());

            IndirectLookup il_obj( l_copy );
	    thrust::sort( d, d+SIZE, il_obj );

            // thrust::host_vector<int> hsorted( SIZE );
            thrust::copy( d, d+SIZE, hsorted.begin() );
            thrust::sort( hsorted.begin(), hsorted.end() );
            thrust::copy( hsorted.begin(), hsorted.end(), std::ostream_iterator<int>(std::cout, " ") );
            cout << endl;
	   
/*
  thrust::sort(
  thrust::device,
  desc_index,
  desc_index+SIZE,
  IndirectLookup(l_copy) //Issue here
  );
*/

	
	    int a[SIZE];

	    int *t2 = thrust::raw_pointer_cast(d);
	    cudaMemcpy(a, t2, SIZE*sizeof(int), cudaMemcpyDeviceToHost);

#else
	    thrust::device_vector<int> d( SIZE );
	    thrust::sequence( d.begin(), d.end() );

            thrust::host_vector<int> b( SIZE );
            thrust::copy( d.begin(), d.end(), b.begin() );
	    	    
            thrust::host_vector<int> hsorted( SIZE );
            thrust::copy( d.begin(), d.end(), hsorted.begin() );
            thrust::sort( hsorted.begin(), hsorted.end() );
            //thrust::copy( hsorted.begin(), hsorted.end(), std::ostream_iterator<int>(std::cout, " ") );
            cout << endl;

            IndirectLookup il_obj( r_copy ); //lcopy vs rcopy here. think r is best choise
	    thrust::sort( d.begin(), d.end(), il_obj );

            // thrust::host_vector<int> hsorted( SIZE );
            thrust::copy( d.begin(), d.end(), hsorted.begin() );
            thrust::sort( hsorted.begin(), hsorted.end() );
            //thrust::copy( hsorted.begin(), hsorted.end(), std::ostream_iterator<int>(std::cout, " ") );
            cout << endl;

            thrust::host_vector<int> a( SIZE );
            thrust::copy( d.begin(), d.end(), a.begin() );
#endif
	    

	    //cudaMemcpy(a, desc_index, SIZE*sizeof(int), cudaMemcpyDeviceToHost);

	    //thrust::sort(a, a+SIZE, thrust::greater<int>());
	    
	    //    for(int i = 0; i < SIZE; i++)
	    //	  printf("%d\t%d\n", b[i], a[i]);

	    int counter = 0;
	    
	    bool success = true;
	    for(int i = 0; i < SIZE; i++) {
		bool tmp = false;
		for(int j = 0; j < SIZE; j++) {		    
		    if(b[i] == a[j])
			tmp = true;
		}
		if(tmp == false) {
		    success = false;
		    counter++;
		}
	    }

            printf("Number of descriptors %d\n", r_len );
	    Descriptor *tmp = (Descriptor*)malloc(r_len*sizeof(Descriptor));

	    cudaMemcpy(tmp, r_copy, r_len*sizeof(Descriptor), cudaMemcpyDeviceToHost);

	    int print = 0;
	    
	    if(success) {
		printf("SUCCESS\n");
		if(print) {
		    printFeature((unsigned int*)&tmp[a[500]]);
		    printFeature((unsigned int*)&tmp[a[501]]);
		    printFeature((unsigned int*)&tmp[a[502]]);
		}
	    }
	    else
		printf("ERROR %d indexes missing\n", counter);

	    //Hash Table setup
	    //**********************************************
	    // init can be done earlier during kernel call
	    //**********************************************
	  
	    //int *indexes = thrust::raw_pointer_cast(d);

	    thrust::device_ptr<int> indexes = &d[0];
	    

	    cudaEventRecord( start, 0 );

	    //Set up the bloom filter
	    //bloom_filter bloom;
	    //initialize_bloom_filter(bloom, SIZE);
	    //bloom_add_filters
	    //	<<<60, 256,0,stream2>>>
	    //	( bloom, r_copy, SIZE );

	    //cudaDeviceSynchronize();


	    
	    Table table; 
	    initialize_table( table, HASH_ENTRIES, SIZE ); //hash entries set equal to size for max performance
	    

	    
	    //initialize mutual exclution locks.
	    Lock lock[HASH_ENTRIES];
	    Lock *dev_lock;
	    cudaMalloc( (void**)&dev_lock, HASH_ENTRIES * sizeof( Lock ) );
	    cudaMemcpy( dev_lock, lock, HASH_ENTRIES * sizeof( Lock ), cudaMemcpyHostToDevice );

	    add_to_table<<<60,256,0,stream1>>>( r_copy, indexes, table, dev_lock, SIZE );
	    
	    cudaEventRecord( stop, 0 );
	    cudaEventSynchronize( stop );
	    //printf("add table call\n");
	    POP_CHK;
	    float elapsedTimeh;
	    cudaEventElapsedTime( &elapsedTimeh, start, stop );
	    printf( "Time to hash:  %3.1f ms\n", elapsedTimeh );
	    //verify_table( table, SIZE ); 

	    cudaEventDestroy( start );
	    cudaEventDestroy( stop );

	    unsigned int *h_start_idx = (unsigned int *)malloc(l_len * sizeof(unsigned int));
	    unsigned int *h_stop_idx = (unsigned int *)malloc(l_len * sizeof(unsigned int));
	    
	    unsigned int *dev_start_idx;
	    unsigned int *dev_stop_idx;

	    
	    cudaMalloc((void **)&dev_start_idx, l_len * sizeof(unsigned int));
	    cudaMalloc((void **)&dev_stop_idx, l_len * sizeof(unsigned int));
	    POP_CHK;
	    
	    //Probably better to utilize threads as well.
	    get_section_from_table
		<<<60, 256>>>
		( table, l_copy, l_len, r_len, dev_start_idx, dev_stop_idx );
	    
	    cudaDeviceSynchronize();
	    cudaMemcpy( h_start_idx, dev_start_idx, l_len * sizeof(unsigned int), cudaMemcpyDeviceToHost );
	    cudaMemcpy( h_stop_idx, dev_stop_idx, l_len * sizeof(unsigned int), cudaMemcpyDeviceToHost );
	    
	    POP_CHK;
	    
	    int print_func = 0;
	    if (print_func)
	    {
		for (int i = 0; i < l_len; i++) { //is this in the sorted! ?
		    printf("num: %d\t start: %d\t stop: %d\n", i, h_start_idx[i], h_stop_idx[i]);
		}
	    }

/* Working hamming, few matches	    
	    compute_distance_hamming
		<<<l_len,1>>>
		( match_matrix, getDescriptors(), l_copy, l_len, other->getDescriptors(), r_copy, r_len, indexes, dev_start_idx, dev_stop_idx );
*/
	    /*
	    compute_distance_hamming_levels
		<<<l_len,1>>>
		( match_matrix, getDescriptors(), l_copy, l_len, other->getDescriptors(), r_copy, r_len, indexes, dev_start_idx, dev_stop_idx );
	    */
	    
 //working dot product in section
	    compute_dot_in_section
		<<<grid,block>>>
		( match_matrix, getDescriptors(), l_len, other->getDescriptors(),  r_len, indexes, dev_start_idx, dev_stop_idx );

	    cudaDeviceSynchronize();

		/*
		  bloom_filter_check
		  <<<60, 256>>>
		  ( bloom, l_copy, SIZE );
		  cudaDeviceSynchronize();
		  
		*/

	    
	    //create array of keys? This can be done in the transpose kernel of the gpu.
	    //might not be needed, can use (some 128 bit struct...*)descriptor
	    //This way all treads (or blocks depending on implementation) can get
	    //its string by descriptor[tid] and a cast.

	    //Add a pointer to a table inside the table? set to void. points to next "level"

	    //Hash Table setup
	    //**********************************************
	    // init can be done earlier during kernel call
	    //**********************************************
	  

/*
  int *ind = popsift::cuda::malloc_devT<int>(SIZE, __FILE__, __LINE__ );
  thrust::sequence(thrust::device, ind, ind+SIZE);	    	   
	    
  struct char_array *data;
  cudaMalloc(&data, sizeof(char_array)*SIZE);

  float da[SIZE*128];
  for(int i = 0; i < SIZE*128; i++)
  da[i] = rand()%100;

  cudaMemcpy(data, da, sizeof(Descriptor)*SIZE, cudaMemcpyDeviceToDevice);

	    
	  
  int copy1[SIZE];
  cudaMemcpy(copy1, ind, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
  //thrust::sort(copy1, copy1+SIZE, thrust::greater<int>());

  thrust::sort(thrust::device, ind, ind+SIZE, cmp_test(data));
	    
  int copy[SIZE];
  cudaMemcpy(copy, ind, SIZE*sizeof(int), cudaMemcpyDeviceToHost);

  //thrust::sort(copy, copy+SIZE);


  bool error = false;

  for(int i = 0; i < SIZE; i++) {
  bool tmp = false;
  for(int j = 0; j < SIZE; j++)  {
  if(copy1[i] == copy[j])
  tmp = true;
  }
  if(!tmp)
  error = true;
  }

  for(int i = 0; i < SIZE; i++)
  printf("%d\t%d\n", copy1[i], copy[i]);

  if(error)
  printf("ERRORRRRRRRRRR\n");
  else {
  struct char_array *tp = (struct char_array*)malloc(SIZE*sizeof(char_array));
  cudaMemcpy(tp, data, sizeof(char_array)*SIZE, cudaMemcpyDeviceToHost);
  for(int i = 0; i < SIZE-1; i++) {
  for(int j = 0; j < 128; j++){
  if(tp[copy[i]].v[j] < tp[copy[i+1]].v[j])
  continue;
  else if(tp[copy[i]].v[j] < tp[copy[i+1]].v[j])
  printf("%f\t%f\n", tp[copy[0]].v[0], tp[copy[1]].v[0]);
  }
  }
		    
  //printf("%f\t%f\t%f\n", tp[copy[0]].v[0], tp[copy[1]].v[0], tp[copy[2]].v[0]);
  }
*/

	    /*
	     * Stumbled upon the error
	     * seems like the issue is that the floats are an array (memory error?)	    
	     */
	    

		
	    
	    
	}

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

    std::ostream& operator<<( std::ostream& ostr, const Feature& feature )
    {
	feature.print( ostr, false );
	return ostr;
    }

} // namespace popsift
