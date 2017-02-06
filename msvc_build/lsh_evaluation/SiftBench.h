#pragma once

#include "KDTree.h"

#include <Eigen/Core>

#include <vector>
#include <utility>
#include <string>
#include <map>

namespace popsift {
namespace kdtree {

class SiftBench {
public:
    SiftBench(const std::vector<std::pair<unsigned, unsigned>>& groundTruth,
        const std::vector<U8Descriptor>& database,
        const std::vector<U8Descriptor>& queries) :
        groundTruth(groundTruth),
        database(database),
        query(queries)
    { }
    
    void BuildKDTree(unsigned leafSize, unsigned treeCount);
    void Bench(unsigned maxCandidates);

private:
    std::vector<std::pair<unsigned, unsigned>> groundTruth;
    std::vector<U8Descriptor> database;
    std::vector<U8Descriptor> query;
    std::vector<KDTreePtr> trees;
    
    unsigned leafSize;
    unsigned treeCount;
    unsigned maxCandidates;
    

    void EvaluateQuery(const U8Descriptor& q, const std::pair<unsigned, unsigned>& gt);
    
    bool SiftMatch(const U8Descriptor& dq, const U8Descriptor& dn1, const U8Descriptor& dn2);

    enum Counters {
        GT_ACCEPTS,
        FALSE_ACCEPTS,
        FALSE_REJECTS,
        TRUE_ACCEPTS,
        TRUE_REJECTS,
        CORRECT_1_MATCHES,
        CORRECT_2_MATCHES,
        DISTANCE_SWAPS,
        COUNTERS_COUNT
    };
    size_t G_Counters[COUNTERS_COUNT];

};

}
}