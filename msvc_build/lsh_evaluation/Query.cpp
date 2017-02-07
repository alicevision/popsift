#include "Query.h"
#include "KDTree.h"
#include <boost/container/flat_set.hpp>
#include <tbb/tbb.h>
#undef min
#undef max

#define DISTANCE_CHECK(a, b) L2DistanceSquared(a, b);

namespace popsift {
namespace kdtree {

Q2NNAccumulator Q2NNAccumulator::Combine(const Q2NNAccumulator& other) const
{
    Q2NNAccumulator r;

    if (distance[0] == other.distance[0]) {
        r.distance[0] = distance[0];
        r.index[0] = index[0];

        if (distance[1] < other.distance[1]) {
            r.distance[1] = distance[1];
            r.index[1] = index[1];
        }
        else {
            r.distance[1] = other.distance[1];
            r.index[1] = other.index[1];
        }
    }
    else if (distance[0] < other.distance[0]) {
        r.distance[0] = distance[0];
        r.index[0] = index[0];

        if (other.distance[0] < distance[1]) {
            r.distance[1] = other.distance[0];
            r.index[1] = other.index[0];
        }
        else {
            r.distance[1] = distance[1];
            r.index[1] = index[1];
        }
    }
    else {
        r.distance[0] = other.distance[0];
        r.index[0] = other.index[0];

        if (distance[0] < other.distance[1]) {
            r.distance[1] = distance[0];
            r.index[1] = index[0];
        }
        else {
            r.distance[1] = other.distance[1];
            r.index[1] = other.index[1];
        }
    }

    r.Validate();
    return r;
}

class Q2NNpq    // std::priority_queue doesn't support preallocation
{
public:
    struct Entry {
        unsigned short distance;    // max L1 distance is 255*128 = 32640
        unsigned short tree;
        unsigned node;
        friend bool operator<(const Entry& e1, const Entry& e2) {
            return e1.distance > e2.distance;   // Reverse heap ordering; smallest on top
        }
    };

    Q2NNpq()
    {
        _pq.reserve(4096);  // Should be more than #trees * #levels to avoid allocations on Push/Pop
    }

    template<typename Mutex>
    void Push(const Entry& e, Mutex& mtx)
    {
        Mutex::scoped_lock lk(mtx);
        Push(e);
    }

    template<typename Mutex>
    bool Pop(Entry& e, Mutex& mtx)
    {
        Mutex::scoped_lock lk(mtx);
        return Pop(e);
    }

private:
    void Push(const Entry& e)
    {
        _pq.push_back(e);
        std::push_heap(_pq.begin(), _pq.end());
    }

    bool Pop(Entry& e)
    {
        if (_pq.empty())
            return false;
        e = _pq.front();
        std::pop_heap(_pq.begin(), _pq.end());
        _pq.pop_back();
        return true;
    }

    std::vector<Entry> _pq;
};

class Q2NNquery
{
    const std::vector<KDTreePtr>& _trees;
    const U8Descriptor& _descriptor;
    const size_t _max_descriptors;

    Q2NNpq _pq;
    tbb::null_mutex _pqmtx;
    boost::container::flat_set<unsigned> _found_descriptors;
    std::vector<unsigned> _leaf_new_descriptors;
    Q2NNAccumulator _result;

    void TraverseToLeaf(Q2NNpq::Entry pqe);
    void ProcessLeaf(const KDTree& tree, unsigned node);

public:
    Q2NNquery(const std::vector<KDTreePtr>& trees, const U8Descriptor& descriptor, size_t max_descriptors);
    std::pair<unsigned, unsigned> operator()();
};

Q2NNquery::Q2NNquery(const std::vector<KDTreePtr>& trees, const U8Descriptor& descriptor, size_t max_descriptors) :
    _trees(trees), _descriptor(descriptor), _max_descriptors(max_descriptors)
{
    _found_descriptors.reserve(_max_descriptors + 16);
    _leaf_new_descriptors.reserve(2048);
}

std::pair<unsigned, unsigned> Q2NNquery::operator()()
{
    for (unsigned short i = 0; i < _trees.size(); ++i) {
        unsigned short d = DISTANCE_CHECK(_descriptor, _trees[i]->BB(0));
        _pq.Push(Q2NNpq::Entry{ d, i, 0 }, _pqmtx);
    }

    Q2NNpq::Entry pqe;
    while (_found_descriptors.size() < _max_descriptors && _pq.Pop(pqe, _pqmtx))
    if (pqe.distance <= _result.distance[1])    // We're searching 2NN, so test 2nd-best distance
        TraverseToLeaf(pqe);

    return std::make_pair(_result.index[0], _result.index[1]);
}

void Q2NNquery::TraverseToLeaf(Q2NNpq::Entry pqe)
{
    const KDTree& tree = *_trees[pqe.tree];
    unsigned node = pqe.node;

    while (!tree.IsLeaf(node)) {
        unsigned l = tree.Left(node), dl = DISTANCE_CHECK(_descriptor, tree.BB(l));
        unsigned r = tree.Right(node), dr = DISTANCE_CHECK(_descriptor, tree.BB(r));

        if (dl <= dr) {
            node = l;
            pqe.node = r; pqe.distance = dr;
            _pq.Push(pqe, _pqmtx);
        }
        else {
            node = r;
            pqe.node = l; pqe.distance = dl;
            _pq.Push(pqe, _pqmtx);
        }
    }
    ProcessLeaf(tree, node);
}

void Q2NNquery::ProcessLeaf(const KDTree& tree, unsigned node)
{
    _leaf_new_descriptors.clear();
    auto list = tree.List(node);
    std::set_difference(list.first, list.second, _found_descriptors.begin(), _found_descriptors.end(),
        std::back_inserter(_leaf_new_descriptors));
    
    // TODO: The two can run in parallel.

    for (unsigned di : _leaf_new_descriptors) {
        unsigned d = DISTANCE_CHECK(_descriptor, tree.Descriptors()[di]);
        _result.Update(d, di);
    }

    _found_descriptors.insert(boost::container::ordered_unique_range,
        _leaf_new_descriptors.begin(), _leaf_new_descriptors.end());
}

std::pair<unsigned, unsigned> Query2NN(const std::vector<KDTreePtr>& trees, const U8Descriptor& descriptor, size_t max_descriptors)
{
    const U8Descriptor* descriptors = trees.front()->Descriptors();
    for (const auto& t : trees) POPSIFT_KDASSERT(t->Descriptors() == descriptors);
    Q2NNquery q(trees, descriptor, max_descriptors);
    return q();
}

/////////////////////////////////////////////////////////////////////////////

TreeQuery::TreeQuery(const U8Descriptor * qDescriptors, size_t dcount, 
                    unsigned treeIndex, Query* query)
    :_qDescriptors(qDescriptors),
    _dcount(dcount),
    _initialTreeIndex(treeIndex),
    _query(query)
{
}

void TreeQuery::FindCandidates()
{
    for (int i = 0; i < _dcount; i++) {
        const U8Descriptor& desc = _qDescriptors[i];

        //initial traverse from root-node
        traverse(desc, 0, _initialTreeIndex);

        //followup-traversal based on priority queue
        while (_candidates.size() < _maxCandidates && _query->priority_queue.size() > 0) {
            
            int nextIndex = -1;
            unsigned nextTreeIndex;
            {
                std::lock_guard<std::mutex>(_query->pq_mutex);
                if (_query->priority_queue.size() > 0) {
                    nextIndex = _query->priority_queue.top().nodeIndex;
                    nextTreeIndex = _query->priority_queue.top().treeIndex;
                    _query->priority_queue.pop();
                }
            }
            if (nextIndex >= 0)
                traverse(desc, nextIndex, nextTreeIndex);
        }

        //Moved the priority-queue traversal from leaf-node block 
        //in TreeQuery::traverse to avoid huge stacks
    }
}

void TreeQuery::traverse(const U8Descriptor & q, unsigned nodeIndex, unsigned treeIndex)
{
    const KDTree& tree = _query->Tree(treeIndex);

    if (tree.IsLeaf(nodeIndex)) {
        auto candidates = _tree->List(nodeIndex);
        _candidates.insert(_candidates.end(), candidates.first, candidates.second);
        //todo: can potentially calc dist between q and tree-desc here.
    }
    else {
        if (tree.Val(nodeIndex) < q.ufeatures[tree.Dim(nodeIndex)]) {
            const BoundingBox& rightBB = _tree->BB(_tree->Right(nodeIndex));
            unsigned right_dist = BBDistance(rightBB, q);
            {
                std::lock_guard<std::mutex>(_query->pq_mutex);
                _query->priority_queue.push(Query::PC{ treeIndex, _tree->Right(nodeIndex), right_dist });
            }
            traverse(q, _tree->Left(nodeIndex), treeIndex);
        }
        else {
            const BoundingBox& leftBB = _tree->BB(_tree->Left(nodeIndex));
            unsigned left_dist = BBDistance(leftBB, q);
            
            {
                std::lock_guard<std::mutex>(_query->pq_mutex);
                _query->priority_queue.push(Query::PC{ treeIndex, _tree->Right(nodeIndex), left_dist });
            }
            traverse(q, _tree->Right(nodeIndex), treeIndex);
        }
    }
}

unsigned TreeQuery::BBDistance(const BoundingBox& bb, const U8Descriptor & q)
{
    unsigned sum = 0;
    for (int i = 0; i < 128; i++) {
        if (q.ufeatures[i] < bb.min.ufeatures[i]) {
            sum += bb.min.ufeatures[i] - q.ufeatures[i];
        }
        else if (q.ufeatures[i] > bb.max.ufeatures[i]) {
            sum += q.ufeatures[i] - bb.max.ufeatures[i];
        }
    }
    return sum;
}

Query::Query(const U8Descriptor * qDescriptors, size_t dcount,
    std::vector<std::unique_ptr<KDTree>> trees, unsigned num_threads)
{
    for (int i = 0; i < trees.size(); i++) {
        TreeQuery q(qDescriptors, dcount, i, this);

    }
}


}
}
