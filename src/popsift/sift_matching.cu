/*
* Copyright 2017, Simula Research Laboratory
*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

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
void test(Descriptor* d_desc_a, int desc_a_count, Descriptor* d_desc_b, int desc_b_count) {
    int tid = threadIdx.x;
    
    for (int i = tid; i < desc_a_count; i += blockDim.x) {
        Descriptor& a = d_desc_a[i];
        float min1 = FLT_MAX, min2 = FLT_MAX;
        for (int x = 0; x < desc_b_count; x++) {
            float dst = calc_distance(&a.features[0], &d_desc_b[x].features[0]);
            
            if (dst < min1) {
                min2 = min1;
                min1 = dst;
            }
            else if (dst < min2) {
                min2 = dst;
            }
            if (min1 / min2 < 0.8f) {
                min1;
            }
        }


    }
}

void Matching::getFlatDeviceDesc(PopSift& ps, Descriptor*& desc_out_device, int* desc_count) {
    Pyramid& p = ps.pyramid(0);
    Features* feat = ps.getFeatures();


    int total_desc = 0;
    for (int octave = 0; octave < p.getNumOctaves(); octave++) {
        Octave& oct_obj = p.getOctave(octave);
        total_desc += oct_obj.getDescriptorCount();
    }
    
    // + (1024 - (total_desc % 1024) 
    desc_out_device = popsift::cuda::malloc_devT<Descriptor>(total_desc, __FILE__, __LINE__);
    *desc_count = total_desc;

    int pos = 0;
    for (int octave = 0; octave < p.getNumOctaves(); octave++) {
        Octave& oct_obj = p.getOctave(octave);

        for (int lvl = 1; lvl < config.levels - 2; lvl++) {
            size_t desc_count = oct_obj.getFeatVecCountH(lvl);
            size_t desc_size = sizeof(Descriptor) * desc_count;

            cudaMemcpy(desc_out_device + pos,
                oct_obj.getDescriptors(lvl),
                desc_size,
                cudaMemcpyDeviceToDevice);

            pos += desc_count;
        }
    }
}

__host__
tmp_ret Matching::Match(PopSift& a, PopSift& b) {
    Pyramid& pa = a.pyramid(0);
    Pyramid& pb = b.pyramid(0);

    Descriptor* flat_a, *flat_b;
    int flat_a_size, flat_b_size;
    getFlatDeviceDesc(a, flat_a, &flat_a_size);
    getFlatDeviceDesc(b, flat_b, &flat_b_size);

    dim3 threadsPerBlock(1024);
    dim3 numBlocks(1);
    //dim3 grid();
    test <<<numBlocks, threadsPerBlock>>>
        (flat_a, flat_a_size, flat_b, flat_b_size);

    cudaDeviceSynchronize();
        
    return tmp_ret();
}


}
