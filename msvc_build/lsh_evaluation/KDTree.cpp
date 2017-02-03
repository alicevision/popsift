#include "KDTree.h"
#include <limits>

namespace popsift {
namespace kdtree {

KDTree::KDTree(const U8Descriptor* descriptors, size_t dcount) :
    _split_dim_gen(0, SPLIT_DIMENSION_COUNT-1),
    _split_val_gen(0, 255),
    _descriptors(descriptors),
    _dcount(dcount),
    _list(dcount)
{
    if (dcount > std::numeric_limits<unsigned>::max() / 2)
        throw std::length_error("KDTree: too many descriptors");
    for (unsigned i = 0; i < _dcount; ++i)
        _list[i] = i;
}

void KDTree::Build()
{

}

// We also check that every element is referenced exactly once by some leaf node.
// This is done in somewhat hacky way by summing element indices withing a leaf
// node and comparing the total sum with the expected sum of 0 + 1 + ... + dcount-1.
void KDTree::Validate()
{
    size_t sum = 0;

    for (const Node& n : _nodes) {
        const size_t lim = n.leaf ? _list.size() : _nodes.size();
        if (n.left <= n.right)
            throw std::logic_error("KDTree: unordered links");
        if (n.left >= lim || n.right >= lim)
            throw std::logic_error("KDTree: links out of range");
        if (n.dim >= 128)
            throw std::logic_error("KDTree: invalid split dimension");

        if (n.leaf)
        for (auto range = List(n); range.first != range.second; ++range.first) {
            if (*range.first >= _dcount)
                throw std::logic_error("KDTree: list element out of range");
            sum += *range.first;
        }
    }

    // Constructor limits count to 2^31, so multiplication won't overflow here.
    if (sum != (size_t(_dcount) - 1) * size_t(_dcount) / 2)
        throw std::logic_error("KDTree: not all elements referenced");
}

}   // kdtree
}   // popsift