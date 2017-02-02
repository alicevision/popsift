#pragma once
#include <immintrin.h>

#ifdef _MSC_VER
#define ALIGNED64 __declspec(align(64))
#else
#define ALIGNED64 __attribute__((aligned(64)))
#endif

namespace popsift {

struct U8Descriptor {
    __m256i features[4];
};

struct L1Distance {
    unsigned operator()(const U8Descriptor&, const U8Descriptor&);
};

struct L2DistanceSquared {
    unsigned operator()(const U8Descriptor&, const U8Descriptor&);
};



}   // popsift
