#include "KDTree.h"
#include <assert.h>
#include <limits>
#include <random>

namespace popsift {
namespace kdtree {

KDTree::KDTree(const U8Descriptor* descriptors, size_t dcount) :
    _split_dim_gen(0, SPLIT_DIMENSION_COUNT-1),
    _split_val_gen(0, 255),
    _descriptors(descriptors),
    _dcount(static_cast<unsigned>(dcount)),
    _list(dcount)
{
    POPSIFT_KDASSERT(dcount < std::numeric_limits<unsigned>::max() / 2);
    for (unsigned i = 0; i < _dcount; ++i)
        _list[i] = i;
}

void KDTree::Build(const SplitDimensions& sdim, unsigned leaf_size)
{
    _leaf_size = leaf_size + 16;    // Don't make too small leafs
    _split_dimensions = sdim;

    // Generate root node as a leaf containing all points.
    _nodes.emplace_back();
    _bb.emplace_back();
 
    _nodes.back().left = 0;
    _nodes.back().right = _dcount;
    _nodes.back().leaf = 1;
    Build(_nodes.back());
}

// On entry, node.left and node.right is read to determine the range; node must be a leaf.
// On exit, node is potentially converted to internal node, and dim,val are filled in
// as well as l/r pointers to children.  BB will also be computed.
void KDTree::Build(Node& node)
{
    POPSIFT_KDASSERT(node.leaf);
    POPSIFT_KDASSERT(node.left < node.right);

    if (node.right - node.left <= _leaf_size) {
        auto list = List(node.left, node.right);
        _bb[Index(node)] = GetBoundingBox(_descriptors, list.first, list.second - list.first);
        return;
    }

    // NB! Partition returns index from [0,n) where 0 maps to left, n maps to right.
    const unsigned m = Partition(node) + node.left;
    const unsigned lc = static_cast<unsigned>(_nodes.size()), rc = lc + 1;

    // Left child to split.
    _nodes.emplace_back();
    _bb.emplace_back();
    _nodes.back().left = node.left;
    _nodes.back().right = m;
    _nodes.back().leaf = 1;
    Build(_nodes.back());

    // Right child to split.
    _nodes.emplace_back();
    _bb.emplace_back();
    _nodes.back().left = m;
    _nodes.back().right = node.right;
    _nodes.back().leaf = 1;
    Build(_nodes.back());

    node.left = lc;     // dim, val, leaf are filled in by successful Partition()
    node.right = rc;
    _bb[Index(node)] = Union(_bb[lc], _bb[rc]);
}

unsigned KDTree::Partition(Node& node)
{
    unsigned m;
    for (unsigned trycount = 0; trycount < 4 * SPLIT_DIMENSION_COUNT; ++trycount)
    if ((m = TryPartition(node)) != _list.size())
        return m;
    // TODO: could create a leaf node in this case instead
    throw std::runtime_error("KDTree: partitioning failed");
}

// Returns _list.size() if the partitioning fails (i.e. all elements have constant value along the dimension)
// Otherwise returns the partition index and fills in partitioning data in node, marking it internal.
unsigned KDTree::TryPartition(Node& node)
{
    static std::mt19937_64 rng_engine;  // XXX! NOT MT-SAFE!

    POPSIFT_KDASSERT(node.leaf);

    const unsigned split_dim = _split_dimensions[_split_dim_gen(rng_engine)];
    const unsigned split_val = _split_val_gen(rng_engine);
    const auto list = List(node.left, node.right);
    const unsigned* mit = std::partition(list.first, list.second, [&, this](unsigned di) {
        return _descriptors[di].ufeatures[split_dim] < split_val;
    });

    if (mit == list.first || mit == list.second)
        return static_cast<unsigned>(_list.size());

    node.dim = split_dim;
    node.val = split_val;
    node.leaf = 0;
    return static_cast<unsigned>(mit - list.first);
}

// We also check that every element is referenced exactly once by some leaf node.
// This is done in somewhat hacky way by summing element indices withing a leaf
// node and comparing the total sum with the expected sum of 0 + 1 + ... + dcount-1.
void KDTree::Validate()
{
    size_t sum = 0;

    for (const Node& n : _nodes) {
        const size_t lim = n.leaf ? _list.size() : _nodes.size();

        POPSIFT_KDASSERT(n.left < n.right);
        POPSIFT_KDASSERT(n.left < lim && n.right <= lim);
        POPSIFT_KDASSERT(n.dim < 128);

        if (n.leaf)
        for (auto range = List(n); range.first != range.second; ++range.first) {
            POPSIFT_KDASSERT(*range.first < _dcount);
            sum += *range.first;
        }
    }

    // Constructor limits count to 2^31, so multiplication won't overflow here.
    POPSIFT_KDASSERT(sum == (size_t(_dcount) - 1) * size_t(_dcount) / 2);
}

}   // kdtree
}   // popsift
