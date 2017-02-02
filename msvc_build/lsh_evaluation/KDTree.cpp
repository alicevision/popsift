#include "KDTree.h"

namespace popsift {
namespace kdtree {

KDTree::KDTree(const U8Descriptor* descriptors, size_t dcount) :
    _split_dim_gen(0, SPLIT_DIMENSION_COUNT-1),
    _split_val_gen(0, 255),
    _descriptors(descriptors),
    _dcount(dcount),
    _list(dcount)
{
    for (size_t i = 0; i < _dcount; ++i)
        _list[i] = i;
}

void KDTree::Build()
{

}

}   // kdtree
}   // popsift