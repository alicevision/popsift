#pragma once


#include "KDTree.h"

#include <memory>
#include <vector>
#include <queue>
#include <functional>
#include <mutex>

namespace popsift {
namespace kdtree {

class Query;

class TreeQuery {
public:
    TreeQuery(const U8Descriptor* qDescriptors, size_t dcount,
        unsigned treeIndex,
        Query* query);

    void FindCandidates();

    const std::vector<unsigned>& Candidates() { return _candidates; }

private:
    const U8Descriptor* _qDescriptors;
    const size_t _dcount;
    const KDTree* _tree;
    const unsigned _initialTreeIndex;
    Query* _query;

    unsigned _maxCandidates = 1000;
    std::vector<unsigned> _candidates;

    void traverse(const U8Descriptor& q,
        unsigned nodeIndex,
        unsigned treeIndex);

    unsigned BBDistance(const BoundingBox& bb,
        const U8Descriptor& q);
};

class Query {
public:
    Query(const U8Descriptor* qDescriptors, size_t dcount,
        std::vector<std::unique_ptr<KDTree>> trees, unsigned num_threads);

    struct PC {
        unsigned treeIndex;
        unsigned nodeIndex;
        unsigned bbDistance;
        friend bool operator> (PC const& a, PC const& b) {
            return a.bbDistance > b.bbDistance;
        }
    };

    std::mutex pq_mutex;
    std::priority_queue<PC,std::vector<PC>, std::greater<PC>> priority_queue;

    KDTree* Tree(size_t t) { return _trees.at(t).get(); }

private:
    std::vector<std::unique_ptr<KDTree>> _trees;
};

}
}