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
#include <immintrin.h>
#include <tbb/tbb.h>
#include <array>
#include <stdexcept>

namespace popsift {

Descriptor* FlattenDescriptorsAsyncD(PopSift& ps)
{
    const size_t descriptor_count = ps.getFeatures()->descriptors().size();
    Descriptor* d_descriptors = popsift::cuda::malloc_devT<Descriptor>(descriptor_count, __FILE__, __LINE__);
    Descriptor* d_descriptors_orig = d_descriptors;

    Pyramid& pyramid = ps.pyramid(0);
    for (int o = 0; o < pyramid.getNumOctaves(); ++o) {
        Octave& octave = pyramid.getOctave(o);
        for (int l = 0; l < octave.getLevels(); ++l) {
            size_t count = octave.getFeatVecCountH(l);
            assert(d_descriptors + count <= d_descriptors_orig + descriptor_count);

            // NB: getDescriptors returns device pointer.
            // XXX: popcuda_memcpy_async should allow size==0.
            if (!count) continue;

            popcuda_memcpy_async(
                d_descriptors,
                octave.getDescriptors(l),
                count * sizeof(Descriptor),
                cudaMemcpyDeviceToDevice,
                octave.getStream(l));

            d_descriptors += count;
        }
    }
    return d_descriptors_orig;
}

std::vector<unsigned> CreateFeatureToExtremaMap(PopSift& ps)
{
    const auto& fv = ps.getFeatures()->features();
    const size_t descriptor_count = ps.getFeatures()->descriptors().size();
    
    std::vector<unsigned> map(descriptor_count);
    auto b = map.begin();
    for (size_t i = 0; i < fv.size(); ++i) {
        size_t count = fv[i].num_descs;
        std::fill(b, b + count, i);
        b += count;
    }

    return map;
}

// NON-VECTORIZED MATCHING OF FLOAT DESCRIPTORS /////////////////////////////

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

// Plain implementation for float descriptors.
static float L2DistanceSquared(const Descriptor& a, const Descriptor& b)
{
    float sum = 0;
    for (int i = 0; i < 128; ++i) {
        float d = a.features[i] - b.features[i];
        sum += d*d;
    }
    return sum;
}

static int match_one_scalar(const Descriptor& d1, const std::vector<Descriptor>& db)
{
    const size_t dbsz = db.size();
    best2_accumulator best2;

    for (size_t ib = 0; ib < dbsz; ++ib) {
        float d = L2DistanceSquared(d1, db[ib]);
        best2.update(d, ib);
    }

    assert(best2.index[0] != -1);                           // would happen on empty vb
    assert(best2.distance[1] != 0);                         // in that case it should be at index 0
    if (best2.index[1] == -1)                               // happens on vb.size()==1
        return best2.index[0];
    if (best2.distance[0] / best2.distance[1] < 0.8*0.8)    // Threshold from the paper, squared
        return best2.index[0];
    return -1;
}

std::vector<int> Matching_CPU(const std::vector<Descriptor>& da, const std::vector<Descriptor>& db)
{
    std::vector<int> matches;
    if (da.empty() || db.empty())
        return matches;

    matches.reserve(da.size());
    const size_t dasz = da.size();

    auto t0 = tbb::tick_count::now();
    for (size_t ia = 0; ia < dasz; ++ia)
        matches.push_back(match_one_scalar(da[ia], db));
    auto t1 = tbb::tick_count::now();

    std::clog << "CPU MATCHING, SCALAR: " << (t1 - t0).seconds() << std::endl;
    return matches;
}

// VECTORIZED/PARALLELIZED MATCHING OF U8 DESCRIPTORS ///////////////////////

#ifdef _MSC_VER
#define ALIGNED16 __declspec(align(16))
#else
#define ALIGNED16 __attribute__((aligned(16)))
#endif

// AVX2 implementation for U8 descriptors.
// 128 components fit in 4 AVX2 registers.  Must expand components from 8-bit
// to 16-bit in order to do arithmetic without overflow. Also, AVX2 doesn't
// support vector multiplication of 8-bit elements.
static float L2DistanceSquared(const U8Descriptor& ad, const U8Descriptor& bd) {
    const __m256i* af = reinterpret_cast<const __m256i*>(ad.features);
    const __m256i* bf = reinterpret_cast<const __m256i*>(bd.features);
    __m256i acc = _mm256_setzero_si256();

    // 32 components per iteration.
    for (int i = 0; i < 4; ++i) {
        // Must compute absolute value after subtraction, otherwise we get wrong result
        // after conversion to 16-bit. (E.g. -1 = 0xFF, after squaring we want to get 1).
        // Max value after squaring is 65025.
        //__m256i d = _mm256_abs_epi8(_mm256_sub_epi8(af[i], bf[i]));
        __m256i d = _mm256_abs_epi8(_mm256_sub_epi8(
            _mm256_load_si256(af + i),
            _mm256_load_si256(bf + i)));
        
        // Squared elements, 0..15
        __m256i dl = _mm256_unpacklo_epi8(d, _mm256_setzero_si256());
        dl = _mm256_mullo_epi16(dl, dl);
        
        // Squared elements, 15..31
        __m256i dh = _mm256_unpackhi_epi8(d, _mm256_setzero_si256());
        dh = _mm256_mullo_epi16(dh, dh);

        // Expand the squared elements to 32-bits and add to accumulator.
        acc = _mm256_add_epi32(acc, _mm256_unpacklo_epi16(dl, _mm256_setzero_si256()));
        acc = _mm256_add_epi32(acc, _mm256_unpackhi_epi16(dl, _mm256_setzero_si256()));
        acc = _mm256_add_epi32(acc, _mm256_unpacklo_epi16(dh, _mm256_setzero_si256()));
        acc = _mm256_add_epi32(acc, _mm256_unpackhi_epi16(dh, _mm256_setzero_si256()));
    }

    ALIGNED16 unsigned int buf[8];
    _mm256_store_si256((__m256i*)buf, acc);
    unsigned int sum = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
    return sum;
}

// Anon-namespace, ODR violations.
namespace
{

struct Reducer
{
    const U8Descriptor& da;
    const U8Descriptor* base;

    Reducer(const U8Descriptor& da, const U8Descriptor* base) : da(da), base(base) {}
    
    best2_accumulator operator()(const tbb::blocked_range<const U8Descriptor*>& r, best2_accumulator init) const {
        for (auto it = r.begin(); it != r.end(); ++it) {
            float d = L2DistanceSquared(da, *it);
            init.update(d, &*it - base);
        }
        return init;
    }

};

struct Combiner
{
    best2_accumulator operator()(const best2_accumulator& a1, const best2_accumulator& a2) const {
        best2_accumulator r;

        if (a1.distance[0] < a2.distance[0]) {
            r.distance[0] = a1.distance[0];
            r.index[0] = a1.distance[0];
            if (a2.distance[0] < a1.distance[1]) r.distance[1] = a2.distance[0];
            else r.distance[1] = a1.distance[1];
        }
        else {
            r.distance[0] = a2.distance[0];
            r.index[0] = a2.index[0];
            if (a1.distance[0] < a2.distance[1]) r.distance[1] = a1.distance[0];
            else r.distance[1] = a2.distance[1];
        }

        assert(r.distance[0] < r.distance[1]);
        return r;
    }
};

}

static int match_one_vector(const U8Descriptor& d1, const std::vector<U8Descriptor>& db)
{
    tbb::blocked_range<const U8Descriptor*> range(db.data(), db.data() + db.size());
    best2_accumulator best2 = tbb::parallel_reduce(
        range,
        best2_accumulator(),
        Reducer(d1, db.data()),
        Combiner());
    
    assert(best2.index[0] != -1);                           // would happen on empty vb
    assert(best2.distance[1] != 0);                         // in that case it should be at index 0
    if (best2.index[1] == -1)                               // happens on vb.size()==1
        return best2.index[0];
    if (best2.distance[0] / best2.distance[1] < 0.8*0.8)    // Threshold from the paper, squared
        return best2.index[0];

    return -1;
}

std::vector<int> Matching_CPU(const std::vector<U8Descriptor>& da, const std::vector<U8Descriptor>& db)
{
    std::vector<int> matches;
    if (da.empty() || db.empty())
        return matches;

    matches.reserve(da.size());
    const size_t dasz = da.size();

    auto t0 = tbb::tick_count::now();
    for (size_t ia = 0; ia < dasz; ++ia)
        matches.push_back(match_one_vector(da[ia], db));
    auto t1 = tbb::tick_count::now();

    std::clog << "CPU MATCHING, SCALAR: " << (t1 - t0).seconds() << std::endl;
    return matches;
}


}   // popsift
