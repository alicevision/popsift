/*
* Copyright 2017, Simula Research Laboratory
*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "sift_matching.h"
#include <assert.h>
#include <float.h>
#include <array>

namespace popsift {

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
    static constexpr size_t no_index = size_t(-1);
    std::array<float, 2> distance;
    std::array<size_t, 2> index;

    best2_accumulator()
    {
        distance.fill(FLT_MAX);
        index.fill(no_index);
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

constexpr size_t best2_accumulator::no_index;

}

static size_t match_one(const Descriptor& d1, const std::vector<Feature>& vb)
{
    best2_accumulator best2;

    for (size_t ib = 0; ib < vb.size(); ++ib) {
        const auto& fb = vb[ib];
        for (int id = 0; id < fb.num_descs; ++id) {
            float d = l2_dist_sq(d1, *fb.desc[id]);
            best2.update(d, ib);
        }
    }

    assert(best2.index[0] != best2_accumulator::no_index);  // would happen on empty vb
    assert(best2.distance[1] != 0);                         // in that case it should be at index 0
    if (best2.index[1] == best2_accumulator::no_index)      // happens on vb.size()==1
        return best2.index[0];
    if (best2.distance[0] / best2.distance[1] < 0.8*0.8)    // Threshold from the paper, squared
        return best2.index[0];
    return size_t(-1);
}

PairList cpu_matching(const Features& ffa, const Features& ffb)
{
    const auto& va = ffa.list();
    PairList pairs;

    if (ffa.list().empty() || ffb.list().empty())
        return pairs;

    for (size_t ia = 0; ia < va.size(); ++ia) {
        const auto& fa = va[ia];
        for (int id = 0; id < fa.num_descs; ++id) {
            size_t ib = match_one(*fa.desc[id], ffb.list());
            // Match only one orientation.
            if (ib != size_t(-1)) {
                pairs.push_back(Pair(ia, ib));
                break;
            }
        }
    }

    return pairs;
}

}   // popsift
