#pragma once
#include <immintrin.h>
#include <array>
#include <vector>
#include <random>
#include <utility>
#include <string>

#ifdef _MSC_VER
#define ALIGNED32 __declspec(align(32))
#else
#define ALIGNED32 __attribute__((aligned(32)))
#endif

#define POPSIFT_KDASSERT(x) if (!(x)) ::popsift::kdtree::assert_fail(#x, __FILE__, __LINE__)

namespace popsift {
namespace kdtree {

inline void assert_fail(const char* expr, const char* file, int line) {
    throw std::logic_error(std::string("KDTree assertion failed: ") + expr + " @ " + file + std::to_string(line));
}

ALIGNED32 struct U8Descriptor {
    union {
        __m256i features[4];
        std::array<unsigned char, 128> ufeatures;
    };
};

struct BoundingBox {
    U8Descriptor min;
    U8Descriptor max;
};

// The code crashes unless this is correct.
static_assert(sizeof(unsigned) == 4, "Unsupported unsigned int size.");
static_assert(alignof(U8Descriptor) >= 32 && alignof(BoundingBox) >= 32, "Invalid alignment.");
static_assert(sizeof(U8Descriptor) == 128 && sizeof(BoundingBox) == 256, "Invalid size.");

/////////////////////////////////////////////////////////////////////////////

constexpr int SPLIT_DIMENSION_COUNT = 5;    // Count of dimensions with highest variance to randomly split against

using SplitDimensions = std::array<unsigned char, SPLIT_DIMENSION_COUNT>;

SplitDimensions GetSplitDimensions(const U8Descriptor* descriptors, size_t count);
BoundingBox GetBoundingBox(const U8Descriptor* descriptors, const unsigned* indexes, size_t count);
BoundingBox Union(const BoundingBox& a, const BoundingBox& b);

unsigned L1Distance(const U8Descriptor&, const U8Descriptor&);
unsigned L1Distance(const U8Descriptor&, const BoundingBox&);
unsigned L2DistanceSquared(const U8Descriptor&, const U8Descriptor&);   // Unused ATM

/////////////////////////////////////////////////////////////////////////////

//! KDTree.  Node 0 is the root node.
class KDTree {
public:
    // XXX: Can save space; left node index is always parent_index+1
    struct Node {
        unsigned left;      // left link, or begin list index if leaf == 1
        unsigned right;     // right link or end list index if leaf ==1
        unsigned dim : 8;   // splitting dimension
        unsigned val : 8;   // splitting value
        unsigned leaf : 1;  // 1 for leaf nodes
    };

    KDTree(const U8Descriptor* descriptors, size_t dcount);
    KDTree(const KDTree&) = delete;
    KDTree& operator=(const KDTree&) = delete;
    
    void Build(const SplitDimensions& sdim, unsigned leaf_size);
    void Validate();

    const Node& Link(unsigned i) const { return _nodes[i]; }
    const BoundingBox& BB(unsigned i) const { return _bb[i]; }
    std::pair<const unsigned*, const unsigned*> List(const Node& node) const {
        POPSIFT_KDASSERT(node.leaf);
        return List(node.left, node.right);
    }
    const U8Descriptor* Descriptors() const {
        return _descriptors;
    }

private:
    const std::uniform_int_distribution<int> _split_dim_gen;
    const U8Descriptor *_descriptors;   // Descriptor data
    const unsigned _dcount;             // Count of descriptors
    std::vector<BoundingBox> _bb;       // BBs of all nodes; packed linearly to not waste cache lines
    std::vector<Node> _nodes;           // Link nodes
    std::vector<unsigned> _list;        // Elements in leaf nodes; consecutive in range [left,right)
    
    // Used by Build
    unsigned _leaf_size;
    SplitDimensions _split_dimensions;

    void Build(unsigned node_index);
    unsigned Partition(Node& node);
    
    std::pair<unsigned*, unsigned*> List(unsigned l, unsigned r) const {
        return std::make_pair(
            const_cast<unsigned*>(_list.data() + l),
            const_cast<unsigned*>(_list.data() + r));
    }
};


}   // kdtree
}   // popsift
