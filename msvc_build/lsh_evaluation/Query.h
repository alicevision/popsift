#pragma once


#include "KDTree.h"

#include <memory>

namespace popsift {
namespace kdtree {

struct {

};

class Query {
public:
    Query(const U8Descriptor* qDescriptors, size_t dcount,
        std::vector<std::unique_ptr<KDTree>> trees);

private:
    std::vector<unsigned> candidates(
        const U8Descriptor* qDescriptors,
        size_t dcount,
        std::unique_ptr<KDTree> tree);
};

}
}