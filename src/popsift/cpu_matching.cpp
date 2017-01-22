/*
* Copyright 2017, Simula Research Laboratory
*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "sift_matching.h"
#include "sift_pyramid.h"
#include "common/debug_macros.h"
#include <assert.h>
#include <float.h>
#include <array>
#include <stdexcept>

namespace popsift {

std::tuple<std::vector<unsigned>, Descriptor*> FlattenDescriptorsD(PopSift& ps)
{
    std::vector<unsigned> d2e_map(ps.getFeatures()->descriptors().size());
    size_t mapi = 0;

    Descriptor* d_descriptors = popsift::cuda::malloc_devT<Descriptor>(d2e_map.size(), __FILE__, __LINE__);
    Descriptor* d_descriptors_orig = d_descriptors;

    Pyramid& pyramid = ps.pyramid(0);
    for (int o = 0; o < pyramid.getNumOctaves(); ++o) {
        Octave& octave = pyramid.getOctave(o);
        for (int l = 0; l < octave.getLevels(); ++l) {
            size_t count = octave.getFeatVecCountH(l);
            if (!count)
                continue;
            
            assert(mapi + count <= d2e_map.size());
            assert(d_descriptors + count <= d_descriptors_orig + d2e_map.size());

            popcuda_memcpy_async(
                d_descriptors,
                octave.getDescriptors(l),       // Returns device pointer!
                count * sizeof(Descriptor),
                cudaMemcpyDeviceToDevice,
                octave.getStream(l));
            
            int* f2emap = octave.getFeatToExtMapH(l);
            std::copy(f2emap, f2emap + count, d2e_map.data() + mapi);

            mapi += count;
            d_descriptors += count;
        }
    }

    assert(mapi == d2e_map.size());

    cudaDeviceSynchronize();
    return std::make_tuple(d2e_map, d_descriptors_orig);
}


/////////////////////////////////////////////////////////////////////////////

static float l2_dist_sq(const Descriptor& a, const Descriptor& b)
{
    float sum = 0;
    for (int i = 0; i < 128; ++i) {
        float d = a.features[i] - b.features[i];
        sum += d*d;
    }
    return sum;
}

// Helper structure put in anon namespace to guard against ODR violations.
namespace {
struct best2_accumulator
{
    std::array<float, 2> distance;
    std::array<int, 2> index;

    best2_accumulator()
    {
        distance.fill(FLT_MAX);
        index.fill(-1);
    }
    
    void update(float d, size_t i)
    {
        if (d < distance[0]) {
            distance[1] = distance[0]; index[1] = index[0];
            distance[0] = d; index[0] = i;
        }
        else if (d != distance[0] && d < distance[1]) {
            distance[1] = d; index[1] = i;
        }
        assert(distance[0] < distance[1]);
        assert(index[0] != index[1]);
    }
};

}

static int match_one(const Descriptor& d1, const std::vector<Feature>& vb)
{
    best2_accumulator best2;

    for (size_t ib = 0; ib < vb.size(); ++ib) {
        const auto& fb = vb[ib];
        for (int id = 0; id < fb.num_descs; ++id) {
            float d = l2_dist_sq(d1, *fb.desc[id]);
            best2.update(d, ib);
        }
    }

    assert(best2.index[0] != -1);                           // would happen on empty vb
    assert(best2.distance[1] != 0);                         // in that case it should be at index 0
    if (best2.index[1] == -1)                               // happens on vb.size()==1
        return best2.index[0];
    if (best2.distance[0] / best2.distance[1] < 0.8*0.8)    // Threshold from the paper, squared
        return best2.index[0];
    return -1;
}

std::vector<int> Matching_CPU(const Features& ffa, const Features& ffb)
{
    const auto& va = ffa.features();
    std::vector<int> matches;

    if (ffa.features().empty() || ffb.features().empty())
        return matches;

    matches.resize(va.size(), -1);
    const size_t vasz = va.size();

#pragma loop(hint_parallel(8))
    for (size_t ia = 0; ia < vasz; ++ia) {
        const auto& fa = va[ia];
        for (int id = 0; id < fa.num_descs; ++id) {
            int ib = match_one(*fa.desc[id], ffb.features());
            // Match only one orientation.
            if (ib != -1) {
                matches[ia] = ib;
                break;
            }
        }
    }

    return matches;
}

}   // popsift
