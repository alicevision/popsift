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
float calc_distance(const T* a, const T* b) {
    float sum = 0.0f;
    for (int i = 0; i < 128; i++) {
        float sub = a[i] - b[i];
        sum += sub*sub;
    }
    return sum;
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
            float dst = calc_distance<float>(&a.features[0], &d_desc_b[x].features[0]);
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

//~1,9sec execution
__global__
void u8_test(U8Descriptor* d_desc_a, int desc_a_count, U8Descriptor* d_desc_b, int desc_b_count, int* output) {
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tid >= desc_a_count) return;

    U8Descriptor& a = d_desc_a[tid];
    float min1 = FLT_MAX, min2 = FLT_MAX;
    int min_index;

    for (int x = 0; x < desc_b_count; x++) {
        float dst = calc_distance<unsigned char>(&a.features[0], &d_desc_b[x].features[0]);
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
        output[tid] = min_index;
    }
    else {
        output[tid] = -1;
    }
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
            float dst = calc_distance<unsigned char>(desc.features, b[i].features);
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
    
    __shared__ U8Descriptor a;
    __shared__ U8Descriptor b;

    memcpy(&a.features[threadIdx.x * 4], &d_desc_a[tid].features[threadIdx.x * 4], sizeof(unsigned char) * 4);
    __shared__ float sums[32];

    for (int i = 0; i < desc_b_count; i++) {                   
        memcpy(&b.features[threadIdx.x * 4], &d_desc_b[i].features[threadIdx.x * 4], sizeof(unsigned char) * 4);
        
        sums[threadIdx.x] = 0.0f;
        for (int x = threadIdx.x*4; x < 128; x++) {
            float sub = a.features[x] - b.features[x];
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


std::vector<int> Matching::Match(popsift::Descriptor* d_desc_a, size_t num_desc_a,
    popsift::Descriptor* d_desc_b, size_t num_desc_b) {
        
    dim3 threadsPerBlock(128);
    dim3 numBlocks((int)ceil(num_desc_a / (float)threadsPerBlock.x));

    int* d_result = popsift::cuda::malloc_devT<int>(num_desc_a, __FILE__, __LINE__);

    std::cout << "starting test";
#if 1
    U8Descriptor* a_U8Descriptor = ConvertDescriptorsToU8(d_desc_a, num_desc_a);
    U8Descriptor* b_U8Descriptor = ConvertDescriptorsToU8(d_desc_b, num_desc_b);
    u8_test <<<numBlocks, threadsPerBlock >>>(a_U8Descriptor, num_desc_a, b_U8Descriptor, num_desc_b, d_result);
    std::vector<int> h_result(num_desc_a);

#else
    test << <numBlocks, threadsPerBlock >> >(d_desc_a, num_desc_a, d_desc_b, num_desc_b, d_result);
    std::vector<int> h_result(num_desc_a);
#endif
    cudaMemcpyAsync(h_result.data(), d_result, num_desc_a * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::cout << "test done";
    
    
    return h_result;
}

}
