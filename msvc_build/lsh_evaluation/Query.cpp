#include "Query.h"

popsift::kdtree::Query::Query(const U8Descriptor * qDescriptors, size_t dcount, 
                                std::vector<std::unique_ptr<KDTree>> trees)
{

}

std::vector<unsigned> popsift::kdtree::Query::candidates(const U8Descriptor * qDescriptors, size_t dcount, std::unique_ptr<KDTree> tree)
{

    return std::vector<unsigned>();
}
