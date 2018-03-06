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

	int len = stop_idx[idx] - start_idx[idx];

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
BLOOM FILTER
****************************/
typedef unsigned int (*hash_function)(const void *data);
typedef struct bloom_filter * bloom_t;
    
struct bloom_hash 
{
  hash_function func;
  struct bloom_hash *next;
};


struct bloom_filter 
{
  struct bloom_hash *func;
  void *bits;
  size_t size;
};


bloom_t bloom_create(size_t size) 
{
    bloom_t res = (bloom_t)calloc(1, sizeof(struct bloom_filter)); //cudamalloc cudamemset
  res->size = size;
  res->bits = malloc(size);
  return res;
}


void bloom_free(bloom_t filter) 
{
  if (filter) {
    while (filter->func) {
      struct bloom_hash *h;
      filter->func = h->next;
      free(h);
    }
    free(filter->bits);
    free(filter);
  }
}



void bloom_add_hash(bloom_t filter, hash_function func) {
    struct bloom_hash *h = (struct bloom_hash *)calloc(1, sizeof(struct bloom_hash));
  h->func = func;
  struct bloom_hash *last = filter->func;
  while (last && last->next) {
    last = last->next;
  }
  if (last) {
    last->next = h;
  } else {
    filter->func = h;
  }
}

    
//bytewise bloomfilter
void bloom_add(bloom_t filter, const void *item) 
{
  struct bloom_hash *h = filter->func;
  uint8_t *bits = (uint8_t *)filter->bits;
  while (h) {
    unsigned int hash = h->func(item);
    printf("hash: %d\n", hash);
    hash %= filter->size;
    printf("hash MOD: %d\n", hash);
    bits[hash] = 1;
    printf("hash/8: %d\n", hash);
    h = h->next;
  }
}

bool bloom_test(bloom_t filter, const void *item) 
{
    struct bloom_hash *h = (struct bloom_hash *)filter->func;
  uint8_t *bits = (uint8_t *)filter->bits;
  while (h) {
    unsigned int hash = h->func(item);
    hash %= filter->size;
    if (!(bits[hash])) {
      return false;
    }
    h = h->next;
  }
  return true;
}


    
/*****************************
BLOOM FILTER end
****************************/
    

/*****************************
HASH TABLE - fix seperate file.
******************************/
    
#define DESC_SEQ 4
#define HASH_ENTRIES     1024*10

    //16 bytes of concecutive memory (4 floats/ints)
struct Desc 
 {
     //unsigned int descriptor[DESC_SEQ]; 
     float descriptor[DESC_SEQ]; //float makes a difference?
 };

/*
 * struct: Entry
 * --------------
 * Table entry for hash table
 * 
 * Each entry holds: 
 * Key: unsigned integer
 * Value: a 128 bit significanse sequence of a discriptor, 
 * but in theory we can hold anything here. Therefore we 
 * make it a void pointer to indicate this. 
 * Next: Null or pointer to the next entry within this 'bucket'. 
 */
struct Entry
{
    struct Desc key;
    //unsigned int value; //should be int begin end ()void * value...
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
		    * creating an interable list?
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

    //Search area - set to max if no key match is found - how do i return these in a good way?
    //two unsigned int arrays allocated on the gpu?
    
    while (tid < elements)
    {
	struct Desc *key = (struct Desc *)(&keys[tid]) + 2; //+2 because second layer is currently stored.

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
    
    __global__ void
    transpose_descriptors(Descriptor * src, int len, Descriptor * des) {

        if(blockIdx.x > len)
            return;

        transpose(src, des, len);
	
	//if(blockIdx.x == 0 && threadIdx.x == 0)
	//    printFeature((unsigned int*)l_tra[blockIdx.x].features);		       
	
	
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

	    transpose_descriptors
		<<<l_len,block>>>
		( getDescriptors(), l_len, l_copy);

	    POP_CHK;

	    transpose_descriptors
		<<<r_len,block>>>
		( other->getDescriptors(), r_len , r_copy);
	    
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
	    
	    cudaEvent_t start, stop;
	    cudaEventCreate( &start );
	    cudaEventCreate( &stop );
	    cudaEventRecord( start, 0 );
	    
	    //find a good way to set HASH_ENTRIES
	    Table table; 
	    initialize_table( table, HASH_ENTRIES, SIZE );

	    //initialize mutual exclution locks.
	    Lock lock[HASH_ENTRIES];
	    Lock *dev_lock;
	    cudaMalloc( (void**)&dev_lock, HASH_ENTRIES * sizeof( Lock ) );
	    cudaMemcpy( dev_lock, lock, HASH_ENTRIES * sizeof( Lock ), cudaMemcpyHostToDevice );

	    add_to_table<<<60,256>>>( r_copy, indexes, table, dev_lock, SIZE );
	    
	    cudaEventRecord( stop, 0 );
	    cudaEventSynchronize( stop );
	    //printf("add table call\n");
	    POP_CHK;
	    float elapsedTime;
	    cudaEventElapsedTime( &elapsedTime, start, stop );
	    printf( "Time to hash:  %3.1f ms\n", elapsedTime );
	    verify_table( table, SIZE ); 

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
		<<<l_len, 1>>>
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


	    compute_dot_in_section
		<<<grid,block>>>
		( match_matrix, getDescriptors(), l_len, other->getDescriptors(),  r_len, indexes, dev_start_idx, dev_stop_idx );

	    cudaDeviceSynchronize();


		
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
