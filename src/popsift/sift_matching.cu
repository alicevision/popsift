/*
* Copyright 2017, Simula Research Laboratory
*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <float.h>


#include "sift_matching.h"
#include "assist.h"
#include "sift_conf.h"
#include "sift_octave.h"
#include "sift_pyramid.h"
#include "sift_extremum.h"
#include "popsift.h"
#include "common/debug_macros.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace popsift {

__global__
void ConvertDescriptorsToU8(Descriptor* d_desc, int count, U8Descriptor* out) {
    int tid = threadIdx.x;
    for (int i = tid; i < count; i += blockDim.x) {
        for (int x = 0; x < 128; x++) {
            unsigned int tmp = d_desc[i].features[x] * 512;
            out[i].features[x] = tmp;
        }
    }
}

U8Descriptor* ConvertDescriptorsToU8(Descriptor* d_descriptors, int count)
{
    auto u8d_descriptors = popsift::cuda::malloc_devT<U8Descriptor>(count, __FILE__, __LINE__);
    int threads_per_block = 64;
    int block_count = (int)ceil(count / (float)threads_per_block);
    ConvertDescriptorsToU8<<<block_count, threads_per_block>>> (d_descriptors, count, u8d_descriptors);
    return u8d_descriptors;
}

Matching::Matching(Config& config)
 : config(config) {

}

Matching::~Matching() {

}

template<typename T>
__device__
float calc_distance_minret(const T* a, const T* b, const float* min2) {
    float sum = 0.0f;
    for (int i = 0; i < 128; i++) {
        float sub = a[i] - b[i];
        sum += sub*sub;
        if (sum > *min2) return sum;
    }
    return sum;
}

__device__ inline unsigned int swar_sub(unsigned int a, unsigned int b) {
    const unsigned int h = 0x80808080;
    return ((a | h) - (b & ~h)) ^ ((a ^ ~b) & h);
}

__device__ inline void update_sum(unsigned& sum, unsigned &d)
{
    unsigned v = d & 0xFF; d >>= 8;
    sum += v*v;
}

__device__
float calc_distance(const U8Descriptor& aa, const U8Descriptor& bb) {
    unsigned sum = 0;
#if 1
    for (int i = 0; i < 128; i++) {
        unsigned a = aa.features[i] - bb.features[i];
        sum += a*a;
    }
    return sum;
#else
    for (int i = 0; i < 32; i += 4) {
        unsigned a = *(const unsigned*)(aa.features + 4 * i);
        unsigned b = *(const unsigned*)(bb.features + 4 * i);
        unsigned d = swar_sub(a, b);
        update_sum(sum, d);
        update_sum(sum, d);
        update_sum(sum, d);
        update_sum(sum, d);
    }
    return sum;
#endif
}

//~16+sec execution
__global__
void test(Descriptor* d_desc_a, int desc_a_count, Descriptor* d_desc_b, int desc_b_count, int* output) {
    int tid = threadIdx.x;
    
    for (int i = tid; i < desc_a_count; i += blockDim.x) {
        Descriptor& a = d_desc_a[i];
        float min1 = FLT_MAX, min2 = FLT_MAX;
        int min_index;

        for (int x = 0; x < desc_b_count; x++) {
            float dst = calc_distance_minret<float>(&a.features[0], &d_desc_b[x].features[0], &min2);
            //printf("%f", dst);
            if (dst < min1) {
                min2 = min1;
                min1 = dst;
                min_index = x;
            }
            else if (dst < min2) {
                min2 = dst;
            }
        }

        if (min1 / min2 < 0.64f) {
            output[i] = min_index;
        }
        else {
            output[i] = -1;
        }
    }
}

__global__
void diff_test(const U8Descriptor* d_desc_a, int count_a, const U8Descriptor* d_desc_b, int count_b)
{
    __shared__ U8Descriptor a[32];              // 4 kB
    __shared__ U8Descriptor b[32];              // 4 kB
    __shared__ float        d[1024];            // 4 kB

                                                // Grid: X: A index; Y: B index
    const U8Descriptor* base_a = d_desc_a + blockIdx.x * blockDim.x;
    const U8Descriptor* base_b = d_desc_b + blockIdx.y * blockDim.y;

    // Block: X: A descriptor index; Y: component index [0..31]
    *(unsigned*)(a[threadIdx.y].features + 4 * threadIdx.x) = *(unsigned*)(base_a[threadIdx.y].features + 4 * threadIdx.x);
    *(unsigned*)(b[threadIdx.y].features + 4 * threadIdx.x) = *(unsigned*)(base_b[threadIdx.y].features + 4 * threadIdx.x);

#if 0
    //int sad;
    //unsigned int _a, _b;

    for (int i = 0; i < 32; i++) {
        //_a = *(uint32_t*)&a[threadIdx.x].features[0];
        //_b = *(uint32_t*)&b[threadIdx.y].features[0];
        //absdf(sad, a, b);
        unsigned sum = 0;
        for (int x = 0; x < 128; x++) {
            //sum += abs(a[threadIdx.x].features[x] - b[threadIdx.y].features[x]);
        }
        d[(threadIdx.x << 5) + threadIdx.y] = sum;
    }
#endif

#if 0
    for (int i = 0; i < 32; ++i)
        d[(threadIdx.x << 5) + threadIdx.y] = calc_distance(a[threadIdx.x], b[threadIdx.y]);
#endif

}


//~1.2sec execution 128x1
__global__
void u8_test(U8Descriptor* d_desc_a, int desc_a_count, U8Descriptor* d_desc_b, int desc_b_count, int* output) {
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tid >= desc_a_count) return;

    U8Descriptor a;
    a = d_desc_a[tid];
    float min1 = FLT_MAX, min2 = FLT_MAX;
	int min_index = -1;
    const int cache_size = 128;
    const int skip_len = cache_size;// *2;
    __shared__ U8Descriptor cached[cache_size];
    unsigned dst = 0;
    for (int x = 0; x < desc_b_count; x += cache_size) {
        //#pragma unroll
        //for (int i = 0; i < 128; i++) {
           //cached[i].features[threadIdx.x] = d_desc_b[threadIdx.x*blockDim.x + threadIdx.x].features[threadIdx.x];
        //}

        memcpy(cached[threadIdx.x].features, d_desc_b[threadIdx.x + x].features, sizeof(U8Descriptor));
        //cached[threadIdx.x] = d_desc_b[threadIdx.x + x];
        __syncthreads();
		for (int i = 0; i < 32; i++) {
			dst = calc_distance(a, cached[i]);
			if (dst < min1) {
				min2 = min1;
				min1 = dst;
				min_index = x;
			}
			else if (dst < min2) {
				min2 = dst;
			}
		}
    }

#if 1
    if (min1 / min2 < 0.64f) {
        output[tid] = min_index;
    }
    else {
        output[tid] = -1;
    }
#endif
}

//~3sec execution
__global__
void u8_test_shared(U8Descriptor* d_desc_a, int desc_a_count, U8Descriptor* d_desc_b, int desc_b_count, int* output) {
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tid >= desc_a_count) return;

    __shared__ U8Descriptor b[32];
    U8Descriptor desc = d_desc_a[tid];
    float min1 = FLT_MAX, min2 = FLT_MAX;
    int min_index;

    for (int x = 0; x < desc_b_count; x += 32) {
        memcpy(b[threadIdx.x].features, d_desc_b[x + threadIdx.x].features, sizeof(U8Descriptor));

        for (int i = 0; i < 32; i++) {
            float dst = calc_distance_minret<unsigned char>(desc.features, b[i].features, &min2);
            if (dst < min1) {
                min2 = min1;
                min1 = dst;
                min_index = x + i;
            }
            else if (dst < min2) {
                min2 = dst;
            }
        }
    }

    if (min1 / min2 < 0.64f) {
        output[tid] = min_index;
    }
    else {
        output[tid] = -1;
    }
}


__device__ void reduce(float* vals) {
    int tid = threadIdx.x;
    if (tid > 15) return;
    vals[tid] += vals[tid + 16];
    vals[tid] += vals[tid + 8];
    vals[tid] += vals[tid + 4];
    vals[tid] += vals[tid + 2];
    vals[tid] += vals[tid + 1];
}

//needs 32x1 blocksize ~5sec execution
__global__
void char_32thread_1desc(U8Descriptor* d_desc_a, int desc_a_count, U8Descriptor* d_desc_b, int desc_b_count, int* output) {
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tid >= desc_a_count) return;

    float min1 = FLT_MAX, min2 = FLT_MAX;
    int min_index;
    
    U8Descriptor a;
    memcpy(&a.features[threadIdx.x * 4], &d_desc_a[tid].features[threadIdx.x * 4], sizeof(unsigned char) * 4);

    __shared__ U8Descriptor b[32];
    __shared__ float sums[32];

    //could it be benefitial if different blocks started on different B's?
    for (int i = 0; i < desc_b_count; i+=32) {                   
        //memcpy(&b.features[threadIdx.x * 4], &d_desc_b[i].features[threadIdx.x * 4], sizeof(unsigned char) * 4);
        memcpy(&b[threadIdx.x].features[0], &d_desc_b[threadIdx.x + i].features[0], sizeof(U8Descriptor));


        sums[threadIdx.x] = 0.0f;
        for (int x = threadIdx.x*4; x < 128; x++) {
            float sub;// = a.features[x] - b.features[x];
            sub = sub*sub;
            sums[threadIdx.x] += sub;
        }
        __syncthreads();
        reduce(&sums[0]);
        if (threadIdx.x == 0) {
            if (sums[0] < min1) {
                min2 = min1;
                min1 = sums[0];
                min_index = i;
            }
            else if (sums[0] < min2) {
                min2 = sums[0];
            }
        }
    }
    if (threadIdx.x == 0) {
        if (min1 / min2 < 0.64f) {
            output[tid] = min_index;
        }
        else {
            output[tid] = -1;
        }
    }
}


struct MinDiff {
    float m[2];
    int idx;
};

__global__
void char_32x32(U8Descriptor* d_desc_a, int desc_a_count, 
    U8Descriptor* d_desc_b, int desc_b_count, int* output) {

    
    __shared__ U8Descriptor a[32]; //4096B
    __shared__ U8Descriptor b[32]; //4096B
    __shared__ MinDiff c[32]; //check if enough registers to remove shared

    int ltid = threadIdx.y * blockDim.x + threadIdx.x; // 0, 1023
    int gtid = ltid + blockIdx.x + (blockIdx.y*gridDim.x);
    //if (blockDim.x*blockIdx.x + threadIdx.y > desc_a_count) return; //add with ceil in blockdim on launch
    
    memcpy(&a[threadIdx.y].features[threadIdx.x * 4], &d_desc_a[blockIdx.x*blockDim.x].features[threadIdx.x * 4], sizeof(unsigned));
    memcpy(&b[threadIdx.y].features[threadIdx.x * 4], &d_desc_b[blockIdx.x*blockDim.x].features[threadIdx.x * 4], sizeof(unsigned));

    *(unsigned int*)(&a[threadIdx.y].features[threadIdx.x * 4]) = *(unsigned int*)(&d_desc_a[blockIdx.x*blockDim.x].features[threadIdx.x * 4]);
    *(unsigned int*)(&b[threadIdx.y].features[threadIdx.x * 4]) = *(unsigned int*)(&d_desc_b[blockIdx.y*blockDim.y].features[threadIdx.x * 4]);
    __syncthreads();

    //float dst = calc_distance(a[threadIdx.x], b[threadIdx.y]);
    /*
    if (dst < c[threadIdx.y].m[0]) {
        c[threadIdx.y].m[1] = c[threadIdx.y].m[0];
        c[threadIdx.y].m[0]  = dst;
        c[threadIdx.y].idx = gtid;
    }
    else if (dst < c[threadIdx.y].m[1]) {
        c[threadIdx.y].m[1] = dst;
    }
    */

    //memcpy(&a[threadIdx.y].features[threadIdx.x], &d_desc_a[]

}


__global__ 
void distance_test(int* output) {
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0, 1023
    int gtid = tid + blockIdx.x + (blockIdx.y*gridDim.x);
    __shared__ U8Descriptor a;
    __shared__ U8Descriptor b;
    if (tid < 128) {
        a.features[tid] = tid;
        b.features[tid] = tid;
    }
    float dst = calc_distance(a, b);
    if(tid==0)
        output[gtid] = (int)dst;
    if (gtid > 14000) printf("asd");

}

__global__
void char_32x32t_32d(U8Descriptor* d_desc_a, int desc_a_count,
    U8Descriptor* d_desc_b, int desc_b_count, int* output) {
    
    __shared__ U8Descriptor a;
    __shared__ U8Descriptor b;
    __shared__ unsigned sum;
    int tid = threadIdx.x;
    int bid = blockDim.y*blockDim.x + blockIdx.x;
    memcpy(a.features + threadIdx.x * 4, d_desc_a[bid].features + threadIdx.x * 4, sizeof(unsigned)); //copy 1desc

    for (int x = 0; x < desc_b_count; x++) {
        memcpy(b.features + threadIdx.x * 4, d_desc_b[x].features + threadIdx.x * 4, sizeof(unsigned)); //copy 1desc

        if (tid == 0) sum = 0;
        unsigned dist = a.features[tid] - b.features[tid];
        dist = dist*dist;
        if (tid == 0) sum += dist;
    }
#if 0
    const int blocky;
    __shared__ float a[blocky][128];
    __shared__ MinDiff minima[blocky];
    
    for (int j = blockDim.y; __any(j<desc_b_count); j += blockDim.y) {
        if (j<desc_b_count) {
            #pragma unroll
            for (int x = threadIdx.x; x < 128; x += 32) {
                a[e][x] = desc1[i][x] - desc2[j][x];
            }

            if (threadIdx.x == 0) {
                float d = normf(a[threadDim.y], 128);
                if (d < a.m[0]) { a.m[1] = a.m[0]; a.idx[1] = a.idx[0]; a.m[0] = d; a.idx[0] = j; }
                else if (d < a.m[1]) { a.m[1] = d; a.idx[1] = j; }
            }

        } 
        __syncthreads();
    }
#endif

}

std::vector<int> Matching::Match(popsift::Descriptor* d_desc_a, size_t num_desc_a,
    popsift::Descriptor* d_desc_b, size_t num_desc_b) {

	std::cout << "starting test" << std::endl;
#if 1
    U8Descriptor* a_U8Descriptor = ConvertDescriptorsToU8(d_desc_a, num_desc_a);
    U8Descriptor* b_U8Descriptor = ConvertDescriptorsToU8(d_desc_b, num_desc_b);
#endif
    int* d_result = popsift::cuda::malloc_devT<int>(num_desc_a, __FILE__, __LINE__);

    

#if 0
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(num_desc_a / 32, num_desc_b / 32);
    diff_test << <numBlocks, threadsPerBlock >> >(a_U8Descriptor, num_desc_a, b_U8Descriptor, num_desc_b);
#endif

#if 0
    dim3 threadsPerBlock(32, 1); //processing 1x32 descs
    dim3 numBlocks(num_desc_a, num_desc_a/threadsPerBlock.y);
    char_32thread_1desc<<<numBlocks, threadsPerBlock>>>(a_U8Descriptor, num_desc_a, b_U8Descriptor, num_desc_b, d_result);
    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;
#endif

    
#if 0
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(num_desc_a / threadsPerBlock.x, num_desc_a / threadsPerBlock.y);
    distance_test<<<numBlocks, threadsPerBlock >>>(d_result);
    cudaError_t r = cudaGetLastError();

#endif



#if 0
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(num_desc_a / threadsPerBlock.x, num_desc_b / threadsPerBlock.y); //need ceiling
    char_32x32<<<numBlocks,threadsPerBlock>>>(a_U8Descriptor, num_desc_a, b_U8Descriptor, num_desc_b, d_result);
#endif

#if 1
    dim3 threadsPerBlock(128, 1);
    dim3 numBlocks(num_desc_a / threadsPerBlock.x); //need ceiling
    u8_test<<<numBlocks, threadsPerBlock>>>(a_U8Descriptor, num_desc_a, b_U8Descriptor, num_desc_b, d_result);
#endif

    //char_32thread_1desc <<<numBlocks, threadsPerBlock >>>(a_U8Descriptor, num_desc_a, b_U8Descriptor, num_desc_b, d_result);
    std::vector<int> h_result(num_desc_a);

    cudaMemcpyAsync(h_result.data(), d_result, num_desc_a * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaError_t r = cudaGetLastError();
    std::cout << "test done";
    return h_result;
}

// 5.2ms
__global__
void float_pipeline_32x1(Descriptor* d_a, Descriptor* d_b, int num_a, int num_b, int* result) {
	Descriptor a;
	memcpy(&a.features[threadIdx.x*4], &d_a->features[threadIdx.x*4], sizeof(float) * 4);

	Descriptor b;
	memcpy(&b.features[threadIdx.x * 4], &d_b[blockIdx.x].features[threadIdx.x * 4], sizeof(float) * 4);

	__shared__ float res[32];
	res[threadIdx.x] = 0.0f;
	res[threadIdx.x] += a.features[threadIdx.x] * b.features[threadIdx.x];
	res[threadIdx.x] += a.features[threadIdx.x+1] * b.features[threadIdx.x+1];
	res[threadIdx.x] += a.features[threadIdx.x+2] * b.features[threadIdx.x+2];
	res[threadIdx.x] += a.features[threadIdx.x+3] * b.features[threadIdx.x+3];

	reduce(res);
	if (threadIdx.x == 0) {
		result[threadIdx.x + blockIdx.x*blockDim.x] = res[0];
	}
}

//14.2ms
__global__
void float_pipeline_128x1(Descriptor* d_a, Descriptor* d_b, int num_a, int num_b, int* result) {
	Descriptor a;
	memcpy(&a.features[threadIdx.x], &d_a->features[threadIdx.x], sizeof(float));

	Descriptor b;
	memcpy(&b.features[threadIdx.x], &d_b[blockIdx.x].features[threadIdx.x], sizeof(float));

	__shared__ float res[128];
	res[threadIdx.x] = a.features[threadIdx.x] * b.features[threadIdx.x];
	
	int tid = threadIdx.x;
	if (tid < 64) res[tid] += res[tid + 64]; __syncthreads(); 
	if (tid < 32) res[tid] += res[tid + 32]; __syncthreads();
	reduce(res);

	if (threadIdx.x == 0) {
		result[threadIdx.x + blockIdx.x*blockDim.x] = res[0];
	}
}

__global__
void float_pipeline_32x32(Descriptor* d_a, Descriptor* d_b, int num_a, int num_b, int* result) {
	
	__shared__ Descriptor a;
	__shared__ Descriptor b[32];
	__shared__ float dots[32][32];
	int tid = threadIdx.x + threadIdx.y*blockDim.x;

	if (threadIdx.y < 4) {
		memcpy(&a.features[threadIdx.x*4], &d_a->features[threadIdx.x*4], sizeof(float)*4);
	}
	//memcpy(&b[threadIdx.y].features[threadIdx.x * 4], &d_b[blockIdx.x+threadIdx.y].features[threadIdx.x * 4], sizeof(float) * 4);

	/*memcpy(&b[threadIdx.y].features[threadIdx.x +0 ], &d_b[blockIdx.x + threadIdx.y].features[threadIdx.x+0 ], sizeof(float));
	memcpy(&b[threadIdx.y].features[threadIdx.x +32], &d_b[blockIdx.x + threadIdx.y].features[threadIdx.x+32], sizeof(float));
	memcpy(&b[threadIdx.y].features[threadIdx.x +64], &d_b[blockIdx.x + threadIdx.y].features[threadIdx.x+64], sizeof(float));
	memcpy(&b[threadIdx.y].features[threadIdx.x +96], &d_b[blockIdx.x + threadIdx.y].features[threadIdx.x+96], sizeof(float));
	__syncthreads();
	*/
	
	
	float lr = 0.0f;
	for (int i = 0; i < 4; i++) {
		lr += a.features[threadIdx.x+i] * d_b[blockIdx.x*blockDim.x + threadIdx.y].features[threadIdx.x+i];
	}
	dots[threadIdx.y][threadIdx.x] = lr;

	__syncthreads();
	reduce(&dots[threadIdx.y][0]);
	if (threadIdx.x == 0) {
		result[threadIdx.y + blockDim.x*32] = dots[threadIdx.y][0];
	}
	
}

__global__
void float_pipeline_32x32_2(Descriptor* d_a, Descriptor* d_b, int num_a, int num_b, int* result) {

	__shared__ Descriptor a;
	float b[4];
	__shared__ float dots[32][32];
	int tid = threadIdx.x + threadIdx.y*blockDim.x;
	if (threadIdx.y < 4) {
		memcpy(&a.features[threadIdx.x * 4], &d_a->features[threadIdx.x * 4], sizeof(float) * 4);
	}
	/*
	memcpy(b, &d_b[blockIdx.x*blockDim.x + threadIdx.y].features[threadIdx.x], sizeof(float) * 4);
	float lr = 0.0f;
	for (int i = 0; i < 4; i++) {
		lr += a.features[threadIdx.x + i] * b[i];
	}
	dots[threadIdx.y][threadIdx.x] = lr;

	__syncthreads();
	reduce(&dots[threadIdx.y][0]);
	if (threadIdx.x == 0) {
		result[threadIdx.y + blockDim.x * 32] = dots[threadIdx.y][0];
	}
	*/
}

//1.2Mms
__global__
void float_pipeline_128x1_noshared(Descriptor* d_a, Descriptor* d_b, int num_a, int num_b, int* result) {
	int tid = threadIdx.x;

	__shared__ float res[128];
	res[tid] = d_a->features[tid] * d_b[blockIdx.x].features[tid];

	/*
	if (tid < 64) res[tid] += res[tid + 64]; __syncthreads();
	if (tid < 32) res[tid] += res[tid + 32]; __syncthreads();
	reduce(res);
	*/
	if (threadIdx.x == 0) {
		result[blockIdx.x] = res[0];
	}
}

//1.2Mms
__global__
void float_pipeline_128x8_noshared(Descriptor* d_a, Descriptor* d_b, int num_a, int num_b, int* result) {
	int tid = threadIdx.x;
	
	__shared__ Descriptor a;
	if (threadIdx.y == 0) a.features[tid] = d_a->features[tid];
	__syncthreads();
	/*
	__shared__ float res[128];
	res[tid] = a.features[tid] * d_b[8*blockIdx.x].features[tid];

	
	if (tid < 64) res[tid] += res[tid + 64]; __syncthreads();
	if (tid < 32) res[tid] += res[tid + 32]; __syncthreads();
	reduce(res);

	if (threadIdx.x == 0) {
		result[blockIdx.x] = res[0];
	}
	*/
	
}

__global__
void float_pipeline_32x1_noshared(Descriptor* d_a, Descriptor* d_b, int num_a, int num_b, int* result) {
	int tid = threadIdx.x;
	float a[4];
	float b[4];
	memcpy(a, &d_a->features[tid * 4], sizeof(float) * 4);
	memcpy(b, &d_b[blockIdx.x].features[tid * 4], sizeof(float) * 4);

	__shared__ float share_res[32];
	float res = a[0] * b[0];
	res += a[1] * b[1];
	res += a[2] * b[2];
	res += a[3] * b[3];
	share_res[tid] = res;

	reduce(share_res);

	if (tid) {
		result[blockIdx.x] = share_res[0];
	}
}



std::vector<int> Matching::PipelineMatch() 
{
	const int num_db = 20000;
	Descriptor* b = new Descriptor[num_db];
	for (int x = 0; x < num_db; x++) {
		for (int i = 0; i < 128; i++) {
			b[x].features[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
		}
	}

	Descriptor* d_b = (Descriptor*)popsift::cuda::malloc_devT<float>(sizeof(Descriptor)*num_db, __FILE__, __LINE__);
	cudaMemcpyAsync(d_b, b, sizeof(Descriptor)*num_db, cudaMemcpyHostToDevice);

	Descriptor a;
	for (int i = 0; i < 128; i++) {
		a.features[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
	}
	Descriptor* d_a = (Descriptor*)popsift::cuda::malloc_devT<float>(sizeof(Descriptor), __FILE__, __LINE__);
	cudaMemcpyAsync(d_a, &a, sizeof(Descriptor), cudaMemcpyHostToDevice);

	std::vector<int> result(num_db);
	int* d_res = popsift::cuda::malloc_devT<int>(num_db, __FILE__, __LINE__);
	cudaMemcpyAsync(result.data(), d_res, sizeof(int)*result.size(), cudaMemcpyDeviceToHost);
	
#if 0
	dim3 threadsPerBlock(32);
	dim3 numBlocks(num_db); //need ceiling
	float_pipeline_32x1 <<<numBlocks, threadsPerBlock >>>(d_a, d_b, 1, num_db, d_res);
#endif
#if 0
	dim3 threadsPerBlock(128);
	dim3 numBlocks(num_db); //need ceiling
	float_pipeline_128x1 <<<numBlocks, threadsPerBlock >>>(d_a, d_b, 1, num_db, d_res);
#endif
#if 0
	dim3 threadsPerBlock(32,32);
	dim3 numBlocks(num_db/32); //need ceiling
	float_pipeline_32x32_2 <<<numBlocks, threadsPerBlock >>>(d_a, d_b, 1, num_db, d_res);
#endif
#if 1
	dim3 threadsPerBlock(128);
	dim3 numBlocks(num_db); //need ceiling
	float_pipeline_128x1_noshared << <numBlocks, threadsPerBlock >> >(d_a, d_b, 1, num_db, d_res);
#endif
#if 0
	dim3 threadsPerBlock(32);
	dim3 numBlocks(num_db); //need ceiling
	float_pipeline_32x1_noshared << <numBlocks, threadsPerBlock >> >(d_a, d_b, 1, num_db, d_res);
#endif
#if 0
	dim3 threadsPerBlock(128, 8);
	dim3 numBlocks(num_db/8); //need ceiling
	float_pipeline_128x8_noshared << <numBlocks, threadsPerBlock >> >(d_a, d_b, 1, num_db, d_res);
#endif
	cudaDeviceSynchronize();

	cudaFree(d_b);
	cudaFree(d_a);
	return result;
}

}


