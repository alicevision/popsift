#include "SiftBench.h"
#include "KDTree.h"
#include "dataio.h"

#include <chrono>
#include <iostream>

namespace popsift {
namespace kdtree {

using std::clog;
using std::endl;

void SiftBench::Bench(unsigned maxCandidates)
{
    this->maxCandidates = maxCandidates;
    
    {
        clog << "\nQUERYING; #VECTORS=" << query.size() << " " << std::flush;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int qi = 0; qi < query.size(); ++qi)
            EvaluateQuery(query[qi], groundTruth[qi]);
        auto t1 = std::chrono::high_resolution_clock::now();
        clog << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << endl;
        ReportMemoryUsage();
    }

    clog << "\n\nSTATISTICS:\n";
    clog << "GT ACCEPTS: " << G_Counters[GT_ACCEPTS] << endl;
    clog << "TRUE ACCEPTS: " << G_Counters[TRUE_ACCEPTS] << endl;
    clog << "TRUE REJECTS: " << G_Counters[TRUE_REJECTS] << endl;
    clog << "FALSE ACCEPTS : " << G_Counters[FALSE_ACCEPTS] << endl;
    clog << "FALSE REJECTS: " << G_Counters[FALSE_REJECTS] << endl;
    clog << "DISTANCE SWAPS: " << G_Counters[DISTANCE_SWAPS] << endl;
    clog << "CORRECT 1-MATCHES: " << G_Counters[CORRECT_1_MATCHES] << endl;
    clog << "CORRECT 2-MATCHES: " << G_Counters[CORRECT_2_MATCHES] << endl;
}



void SiftBench::BuildKDTree(unsigned leafSize, unsigned treeCount)
{
    this->leafSize = leafSize;
    this->treeCount = treeCount;

    auto sdim = GetSplitDimensions(database.data(), database.size());
    clog << "\nBUILDING KDTREE: " << std::flush;
    auto t0 = std::chrono::high_resolution_clock::now();
    trees = Build(database.data(), groundTruth.size(), treeCount, leafSize);
    auto t1 = std::chrono::high_resolution_clock::now();
    clog << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << endl;
    
    ReportMemoryUsage();
}

void SiftBench::EvaluateQuery(const U8Descriptor& q, 
    const std::pair<unsigned, unsigned>& gt)
{

    auto knn = Query2NN(trees, q, maxCandidates);
    bool gt_sift_accept = SiftMatch(q, database[gt.first], database[gt.second]);
    bool q_sift_accept = SiftMatch(q, database[knn.first], database[knn.second]);

    if (gt_sift_accept)
        ++G_Counters[GT_ACCEPTS];

    if (gt_sift_accept == q_sift_accept) {
        if (gt_sift_accept) ++G_Counters[TRUE_ACCEPTS];
        else ++G_Counters[TRUE_REJECTS];
    }
    else {
        if (q_sift_accept) ++G_Counters[FALSE_ACCEPTS];
        else ++G_Counters[FALSE_REJECTS];
    }

    if (gt.first == knn.first) {
        ++G_Counters[CORRECT_1_MATCHES];
        if (gt.second == knn.second)
            ++G_Counters[CORRECT_2_MATCHES];
    }

}

bool SiftBench::SiftMatch(const U8Descriptor & dq, const U8Descriptor & dn1, const U8Descriptor & dn2)
{
    unsigned d1 = L2DistanceSquared_scalar(dq, dn1);
    unsigned d2 = L2DistanceSquared_scalar(dq, dn2);
    if (d1 > d2) {  // Search is with L1-norm, so this can happen
        ++G_Counters[DISTANCE_SWAPS];
        std::swap(d1, d2);
    }

    return (float)d1 / (float)d2 < 0.8;
}

}
}
