#pragma once
#include <immintrin.h>
#include <array>
#include <vector>
#include <random>

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

struct L1Distance {
    unsigned operator()(const U8Descriptor&, const U8Descriptor&);
};

struct L2DistanceSquared {
    unsigned operator()(const U8Descriptor&, const U8Descriptor&);
};

/////////////////////////////////////////////////////////////////////////////

constexpr int SPLIT_DIMENSION_COUNT = 5;    // Count of dimensions with highest variance to randomly split against

struct BoundingBox {
    U8Descriptor min;
    U8Descriptor max;
};

std::array<int, SPLIT_DIMENSION_COUNT> GetSplitDimensions(const U8Descriptor* descriptors, size_t count);
BoundingBox GetBoundingBox(const U8Descriptor* descriptors, unsigned* indexes, size_t count);

class KDTree {
    struct Link {
        unsigned left;      // left link, or begin list index if leaf == 1
        unsigned right;     // right link or end list index if leaf ==1
        unsigned dim : 8;   // splitting dimension
        unsigned leaf : 1;  // 1 for leaf nodes
    };

    const std::uniform_int_distribution<int> _split_dim_gen;
    const std::uniform_int_distribution<unsigned> _split_val_gen;
    const U8Descriptor *_descriptors;   // Descriptor data
    const size_t _dcount;               // Count of descriptors
    std::vector<BoundingBox> _bb;       // BBs of all nodes; packed linearly to not waste cache lines
    std::vector<Link> _links;           // Link nodes
    std::vector<unsigned> _list;        // Elements in leaf nodes; consecutive in range [left,right)

public:
    KDTree(const U8Descriptor* descriptors, size_t dcount);
    KDTree(const KDTree&) = delete;
    KDTree& operator=(const KDTree&) = delete;
    void Build();
};


}   // kdtree
}   // popsift
