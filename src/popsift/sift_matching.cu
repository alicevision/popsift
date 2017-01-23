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

Matching::Matching(Config& config)
 : config(config) {

}

Matching::~Matching() {

}

template<typename T>
__device__
float calc_distance(T* a, T* b) {
    float sum = 0.0f;
    for (int i = 0; i < 128; i++) {
        float sub = a[i] - b[i];
        sum += sub*sub;
    }
    return sum;
}

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

struct u8desc {
    unsigned char features[128];
};
__global__
void convert(Descriptor* d_desc, int count, u8desc* out) {
    int tid = threadIdx.x;

    for (int i = tid; i < count; i += blockDim.x) {
        for (int x = 0; x < 128; x++) {
            unsigned int tmp = d_desc[i].features[x] * 512;
            out[i].features[x] = tmp;
        }
    }
}

__global__
void u8_test(u8desc* d_desc_a, int desc_a_count, u8desc* d_desc_b, int desc_b_count, int* output) {
    int tid = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tid >= desc_a_count) return;

    u8desc& a = d_desc_a[tid];
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


std::vector<int> Matching::Match(popsift::Descriptor* d_desc_a, size_t num_desc_a,
    popsift::Descriptor* d_desc_b, size_t num_desc_b) {
    dim3 threadsPerBlock(128);
    dim3 numBlocks((int)ceil(num_desc_a / 128.f));
    
    int* d_result = popsift::cuda::malloc_devT<int>(num_desc_a, __FILE__, __LINE__);

    std::cout << "starting test";
#if 1
        u8desc* a_u8desc = popsift::cuda::malloc_devT<u8desc>(num_desc_a, __FILE__, __LINE__);
        u8desc* b_u8desc = popsift::cuda::malloc_devT<u8desc>(num_desc_b, __FILE__, __LINE__);
        convert << <numBlocks, threadsPerBlock >> > (d_desc_a, num_desc_a, a_u8desc);
        convert << <numBlocks, threadsPerBlock >> > (d_desc_b, num_desc_b, b_u8desc);
        
        u8_test<<<numBlocks, threadsPerBlock >>>(a_u8desc, num_desc_a, b_u8desc, num_desc_b, d_result);
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
