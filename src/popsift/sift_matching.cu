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
#include "cublas.h"
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

void calcDistMat(Descriptor* a, Descriptor* b, int numA, int numB, float* res) {
    for (int ai = 0; ai < numA; ai++) {
        for (int bi = 0; bi < numB; bi++) {
            size_t outi = ai + numB*bi;
            res[outi] = 0.0;
            for (int i = 0; i < 128; i++) {
                res[outi] += a[ai].features[i] * b[bi].features[i];
            }
        }
    }
}

// device_desc_a: a set of input descriptors, stored on gpu
// num_a: number of device_desc_a 
// database_descs: vector containing descriptors and number of descriptors for a set of 
//                database descriptors that the input descriptors are matched against.
// returns: vector of vector of best match descriptors
// ToDo: If this function is called in parallel with descriptor-extraction, it may be
//       worth defining other streams than 0->database_descs.size(), and use cudaStreamSynchronize()
std::vector<std::pair<float*, size_t>> Matching::CalcDistances(popsift::Descriptor* device_desc_a, size_t num_a,
    std::vector<std::pair<popsift::Descriptor*, size_t>> database_descs)
{
    std::vector<std::pair<float*, size_t>> result;
    
    
    // Cuda currently support <= 1024 streams. In this scenario a number closer to 10
    // is ideal, but since we will mostly operate with around 5 database descriptor sets,
    // it should be no problem.
    assert(database_descs.size() <= 1024);
    
    std::vector<cudaStream_t> streams;
    streams.resize(database_descs.size());
    for (size_t i = 0; i < streams.size(); i++) {
        cudaStreamCreate(&streams[i]);
    }

    cublasHandle_t handle;
    cublasCreate_v2(&handle);
    static const float alpha = 1.0f;
    static const float beta  = 0.0f;


    for (size_t i = 0; i < database_descs.size(); i++) {
        cublasSetStream_v2(handle, streams[i]);
        const std::pair<popsift::Descriptor*, size_t>& b_descs = database_descs.at(i);
        
        size_t num_result = num_a*b_descs.second;

        float* device_result = popsift::cuda::malloc_devT<float>(num_result, __FILE__, __LINE__);
        cublasSgemm_v2(
            handle,
            CUBLAS_OP_T,            // A transposed
            CUBLAS_OP_N,            // B not transposed
            num_a,                  // m
            b_descs.second,         // n
            128,                    // k
            &alpha,                // alpha
            (float*)device_desc_a,  // A
            128,                    // lda
            (float*)b_descs.first,  // B
            128,                    // ldb
            &beta,                  // beta XXX: was &alpha in old code, is &beta correct?
            device_result,          // C
            num_a                   // ldc, XXX: not sure if num_a or 128 (LDC=max(1,m))
        );
        float* gpu_res = popsift::cuda::malloc_hstT<float>(num_result, __FILE__, __LINE__);
        cudaMemcpyAsync(gpu_res, device_result, sizeof(float)*num_result, cudaMemcpyDeviceToHost, streams[i]);
        result.push_back(std::make_pair(gpu_res, num_result));
        

    }
    
    cudaDeviceSynchronize(); 

#if 1 // bruteforce cpu matching for validation
    for (size_t i = 0; i < database_descs.size(); i++) {
        const std::pair<popsift::Descriptor*, size_t>& b_descs = database_descs.at(i);
        size_t num_result = num_a*b_descs.second;
        float* cpu_res = popsift::cuda::malloc_hstT<float>(num_result, __FILE__, __LINE__);
        calcDistMat(device_desc_a, b_descs.first, num_a, b_descs.second, cpu_res);
        assert(result.at(i).second == num_result);
        int num_match = 0;
        for (int x = 0; x < num_result; x++) {
            // Expecting differences due to different algorithm and gpu and cpu. Can add an epsilon in test.
            if (result.at(i).first[x] == cpu_res[x])
                num_match++;
        }
        std::cout << "popsift cublas matching got " << num_match << "/" << num_result 
                  << " correct distance calculations" << std::endl;
    }
#endif 
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "popsift cublas matching got error: " << cudaGetErrorString(err);
    }
    return result;
}

}
