#pragma once
#include <immintrin.h>
#include <array>

#ifdef _MSC_VER
#define ALIGNED64 __declspec(align(64))
#else
#define ALIGNED64 __attribute__((aligned(64)))
#endif

namespace popsift {
namespace kdtree {

struct U8Descriptor {
    union {
        __m256i features[4];
        std::array<unsigned char, 128> ufeatures;
    };
};

struct BoundingBox {
    U8Descriptor min;
    U8Descriptor max;
};

struct L1Distance {
    unsigned operator()(const U8Descriptor&, const U8Descriptor&);
};

struct L2DistanceSquared {
    unsigned operator()(const U8Descriptor&, const U8Descriptor&);
};

/////////////////////////////////////////////////////////////////////////////

constexpr int SPLIT_DIMENSION_COUNT = 5;    // Count of dimensions with highest variance to randomly split against

std::array<int, SPLIT_DIMENSION_COUNT> GetSplitDimensions(const U8Descriptor* descriptors, size_t count);
BoundingBox GetBoundingBox(const U8Descriptor* descriptors, unsigned* indexes, size_t count);


}   // kdtree
}   // popsift
