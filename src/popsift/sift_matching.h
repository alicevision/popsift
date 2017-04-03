/*
* Copyright 2017, Simula Research Laboratory
*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include "sift_conf.h"
#include "sift_extremum.h"
#include "popsift.h"
#include <vector>

class PopSift;

namespace popsift {

// Flatten descriptors from all octaves to a contiguous block on the device.
Descriptor* FlattenDescriptorsAsyncD(PopSift& ps);

// Create a descriptor -> extrema map.
std::vector<unsigned> CreateFeatureToExtremaMap(PopSift& ps);

struct U8Descriptor {
    unsigned char features[128];
};

// Converts float descriptors to unsigned char by multiplying floats by 512.
U8Descriptor* ConvertDescriptorsToU8(Descriptor* d_descriptors, int count, cudaStream_t stream = nullptr);

// Returns vector v indexed by descriptors in fa s.t. v[i] is the best matching descriptor in fb.
// If a descriptor in fa has no match in fb, v[i] == -1.
std::vector<int> Matching_CPU(const std::vector<Descriptor>& da, const std::vector<Descriptor>& db);
std::vector<int> Matching_CPU(const std::vector<U8Descriptor>& da, const std::vector<U8Descriptor>& db);
float L2DistanceSquared(const U8Descriptor& ad, const U8Descriptor& bd);

class Matching
{
public:
    Matching(Config& config);
    ~Matching();

    // device_desc_a:   a set of input descriptors, stored on gpu
    // num_a:           number of device_desc_a 
    // database_descs:  vector containing descriptors and number of descriptors for a set of 
    //                  database descriptors that the input descriptors are matched against.
    // returns:         Vector element K is a vector of pairs. For a pair, P, P.first is 
    //                  an index in device_desc_a and P.second is an index in database_descs[K].first
    std::vector<std::vector<std::pair<size_t, size_t>>> Match(popsift::Descriptor* device_desc_a, size_t num_a,
        std::vector<std::pair<popsift::Descriptor*, size_t>> database_descs);

private:
    const Config& config;
    std::vector<cudaStream_t> streams;

    // Output: Vector element K is a 2d matrix M of dimensions num_a * database_descs[K].second.
    //         element M[i,j] is the distance between device_desc_a[i] and database_descs[K].first[j],
    //         where i is an index [0, num_a] and j an index[0, database-descs[K].second].
    std::vector<float*> CalcDistances(popsift::Descriptor* device_desc_a, size_t num_a,
        std::vector<std::pair<popsift::Descriptor*, size_t>> database_descs);
};

}
