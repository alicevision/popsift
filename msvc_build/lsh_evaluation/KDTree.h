#pragma once
#include <immintrin.h>
#include <array>
#include <vector>
#include <random>
#include <utility>

#ifdef _MSC_VER
#define ALIGNED64 __declspec(align(64))
#else
#define ALIGNED64 __attribute__((aligned(64)))
#endif

namespace popsift {
namespace kdtree {

static_assert(sizeof(unsigned) == 4, "Unsupported unsigned int size.");

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

//! KDTree.  Node 0 is the root node.
class KDTree {
    struct Node {
        unsigned left;      // left link, or begin list index if leaf == 1
        unsigned right;     // right link or end list index if leaf ==1
        unsigned dim : 8;   // splitting dimension
        unsigned leaf : 1;  // 1 for leaf nodes
    };

    const std::uniform_int_distribution<int> _split_dim_gen;
    const std::uniform_int_distribution<unsigned> _split_val_gen;
    const U8Descriptor *_descriptors;   // Descriptor data
    const unsigned _dcount;             // Count of descriptors
    std::vector<BoundingBox> _bb;       // BBs of all nodes; packed linearly to not waste cache lines
    std::vector<Node> _nodes;           // Link nodes
    std::vector<unsigned> _list;        // Elements in leaf nodes; consecutive in range [left,right)

public:
    KDTree(const U8Descriptor* descriptors, size_t dcount);
    KDTree(const KDTree&) = delete;
    KDTree& operator=(const KDTree&) = delete;
    
    void Build();
    void Validate();

    const Node& Link(unsigned i) const { return _nodes[i]; }
    const BoundingBox& BB(unsigned i) const { return _bb[i]; }
    std::pair<const unsigned*, const unsigned*> List(const Node& node) const {
        if (node.leaf)
            throw std::logic_error("KDTree::List: node is not a leaf");
        return std::make_pair(_list.data() + node.left, _list.data() + node.right);
    }
};


}   // kdtree
}   // popsift
