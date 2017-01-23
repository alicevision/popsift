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

__device__
float calc_distance(float* a, float* b) {
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
            float dst = calc_distance(&a.features[0], &d_desc_b[x].features[0]);
            printf("%f", dst);
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

std::vector<int> Matching::Match(popsift::Descriptor* d_desc_a, size_t num_desc_a,
    popsift::Descriptor* d_desc_b, size_t num_desc_b) {
    
    int* d_result = popsift::cuda::malloc_devT<int>(num_desc_a, __FILE__, __LINE__);

    dim3 threadsPerBlock(1024);
    dim3 numBlocks(1);
    test <<<numBlocks, threadsPerBlock >>>(d_desc_a, num_desc_a, d_desc_b, num_desc_b, d_result);

    std::vector<int> h_result(num_desc_a);

    cudaMemcpyAsync(h_result.data(), d_result, num_desc_a * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    
    return h_result;
}

}
