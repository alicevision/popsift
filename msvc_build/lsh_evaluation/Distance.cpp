#include "KDTree.h"
#include <stdint.h>
#include <algorithm>
#include <Eigen/Core>
#include <vector>
#include <tuple>

namespace popsift {
namespace kdtree {

static_assert(SPLIT_DIMENSION_COUNT < 128, "Invalid split dimension count");

unsigned L1Distance(const U8Descriptor& ad, const U8Descriptor& bd)
{
    const __m256i* af = ad.features;
    const __m256i* bf = bd.features;
    __m256i acc = _mm256_setzero_si256();

    // 32 components per iteration.
    for (int i = 0; i < 4; ++i) {
        __m256i d = _mm256_sad_epu8(
            _mm256_load_si256(af + i),
            _mm256_load_si256(bf + i));
        acc = _mm256_add_epi64(acc, d);
    }

    ALIGNED32 uint64_t buf[4];
    _mm256_store_si256((__m256i*)buf, acc);
    unsigned int sum = buf[0] + buf[1] + buf[2] + buf[3];
    return sum;
}

// Returns the least possible distance between descriptor and BB.  The code uses the following:
// ----X1----m----X2-------M-----X3--- ; bb.min <= bb.max (componentwise)
// We observe that
//                 / 0, when x >= m                         / 0, when x <= M
// m - min(m,X) = |                         max(M,x) - x = |
//                 \ m-x, when x < m                        \ x-M when x > M
//
unsigned L1Distance(const U8Descriptor& d, const BoundingBox& bb)
{
    __m256i acc = _mm256_setzero_si256();
    
    for (int i = 0; i < 4; ++i) {
        __m256i d1 = _mm256_sad_epu8(bb.min.features[i], _mm256_min_epu8(bb.min.features[i], d.features[i]));
        __m256i d2 = _mm256_sad_epu8(bb.max.features[i], _mm256_max_epu8(bb.max.features[i], d.features[i]));
        acc = _mm256_add_epi64(acc, _mm256_add_epi64(d1, d2));
    }

    ALIGNED32 uint64_t buf[4];
    _mm256_store_si256((__m256i*)buf, acc);
    unsigned int sum = buf[0] + buf[1] + buf[2] + buf[3];
    return sum;
}

// AVX2 implementation for U8 descriptors.
// 128 components fit in 4 AVX2 registers.  Must expand components from 8-bit
// to 16-bit in order to do arithmetic without overflow. Also, AVX2 doesn't
// support vector multiplication of 8-bit elements.
unsigned L2DistanceSquared(const U8Descriptor& ad, const U8Descriptor& bd)
{
    const __m256i* af = ad.features;
    const __m256i* bf = bd.features;
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

    ALIGNED32 unsigned int buf[8];
    _mm256_store_si256((__m256i*)buf, acc);
    unsigned int sum = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
    return sum;

}

SplitDimensions GetSplitDimensions(const U8Descriptor* descriptors, size_t count)
{
    using namespace Eigen;

    using SiftPoint = Array<double, 1, 128>;
    Map<Array<unsigned char, Dynamic, 128, RowMajor>, Aligned32> u8data((unsigned char*)descriptors, count, 128);
    
    SiftPoint mean = u8data.cast<double>().colwise().sum() / count;
    SiftPoint var = SiftPoint::Zero();
    for (size_t i = 0; i < count; ++i) {
        auto v = u8data.row(i).cast<double>() - mean;
        var += v * v;
    }

    using vd_tup = std::tuple<double, unsigned>;
    std::array<vd_tup, 128> vardim;
    for (int i = 0; i < 128; ++i)
        vardim[i] = std::make_tuple(var[i], i);
    std::sort(vardim.begin(), vardim.end(), [](const vd_tup& v1, const vd_tup& v2) {
        return std::get<0>(v1) > std::get<0>(v2);
    });

    SplitDimensions ret;
    std::transform(vardim.data(), vardim.data() + SPLIT_DIMENSION_COUNT, ret.begin(),
        [](const vd_tup& v) { return std::get<1>(v); });
    return ret;
}

//! Compute BB of descriptors referenced by count indexes.
BoundingBox GetBoundingBox(const U8Descriptor* descriptors, const unsigned* indexes, size_t count)
{
    U8Descriptor min, max;

    for (int i = 0; i < 4; i++) {
        min.features[i] = _mm256_set1_epi8(-1);
        max.features[i] = _mm256_setzero_si256();
    }

    for (size_t i = 0; i < count; ++i)
    for (int j = 0; j < 4; ++j) {
        min.features[j] = _mm256_min_epu8(min.features[j], descriptors[i].features[j]);
        max.features[j] = _mm256_max_epu8(max.features[j], descriptors[i].features[j]);
    }

    return BoundingBox{ min, max };
}

BoundingBox Union(const BoundingBox& a, const BoundingBox& b)
{
    BoundingBox r;
    for (int i = 0; i < 4; ++i) {
        r.min.features[i] = _mm256_min_epu8(a.min.features[i], b.min.features[i]);
        r.max.features[i] = _mm256_max_epu8(a.max.features[i], b.max.features[i]);
    }
    return r;
}

}   // kdtree
}   // popsift