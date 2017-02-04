#pragma once
#include <immintrin.h>
#include <array>
#include <vector>
#include <random>
#include <utility>
#include <string>

#ifdef _MSC_VER
#define ALIGNED64 __declspec(align(64))
#else
#define ALIGNED64 __attribute__((aligned(64)))
#endif

#define POPSIFT_KDASSERT(x) if (!(x)) ::popsift::kdtree::assert_fail(#x, __FILE__, __LINE__)

namespace popsift {
namespace kdtree {

inline void assert_fail(const char* expr, const char* file, int line) {
    throw std::logic_error(std::string("KDTree assertion failed: ") + expr + " @ " + file + std::to_string(line));
}

ALIGNED64 struct U8Descriptor {
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

static_assert(sizeof(unsigned) == 4, "Unsupported unsigned int size.");
static_assert(alignof(U8Descriptor) >= 32, "Invalid U8Descriptor alignment.");
static_assert(sizeof(U8Descriptor) == 128, "Invalid U8Descriptor size.");

/////////////////////////////////////////////////////////////////////////////

constexpr int SPLIT_DIMENSION_COUNT = 5;    // Count of dimensions with highest variance to randomly split against

using SplitDimensions = std::array<unsigned char, SPLIT_DIMENSION_COUNT>;

SplitDimensions GetSplitDimensions(const U8Descriptor* descriptors, size_t count);
BoundingBox GetBoundingBox(const U8Descriptor* descriptors, const unsigned* indexes, size_t count);
BoundingBox Union(const BoundingBox& a, const BoundingBox& b);

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

    void Build(Node& node);
    unsigned Partition(Node& node);
    
    std::pair<unsigned*, unsigned*> List(unsigned l, unsigned r) const {
        return std::make_pair(
            const_cast<unsigned*>(_list.data() + l),
            const_cast<unsigned*>(_list.data() + r));
    }

    unsigned Index(Node& node) const {
        POPSIFT_KDASSERT(&node >= &_nodes.front() && &node <= &_nodes.back());
        return static_cast<unsigned>(&node - &_nodes.front());
    }
};


}   // kdtree
}   // popsift
