#include "dataio.h"
#include "KDTree.h"
#include <Eigen/Core>
#include <iostream>
#include <chrono>
#include <memory>

using namespace popsift::kdtree;

using Eigen::Map;
using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::RowMajor;

using std::clog;
using std::endl;

// http://corpus-texmex.irisa.fr/ ANN_SIFT1M set
using SiftPoint = Matrix<unsigned char, 128, 1>;
using SiftMatrix = Map<Matrix<unsigned char, Dynamic, 128, RowMajor>, Eigen::Aligned32>;
using GroundTruthMatrix = Map<Matrix<int, Dynamic, 100, RowMajor>>;

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


static std::string G_DataPrefix = "C:/LOCAL/texmex_sift1M_corpus";
static std::vector<U8Descriptor> G_Base; // 1million descriptors database
static std::vector<U8Descriptor> G_Query; // query descriptors
static std::vector<int> G_GroundTruth; // 100x10k, 100 nearest neighbours for each desc in G_Query
static std::vector<KDTreePtr> G_trees;
static size_t G_Counters[COUNTERS_COUNT];

static void ReadData();
static void BuildKDTree();
static void EvaluateQuery(const U8Descriptor& q, const std::pair<unsigned, unsigned>& gt);
static bool SiftMatch(const U8Descriptor& dq, const U8Descriptor& dn1, const U8Descriptor& dn2);

static constexpr unsigned LEAF_SIZE = 50;
static constexpr unsigned TREE_COUNT = 20;
static constexpr unsigned QUERY_DESCRIPTOR_LIMIT = 1000;

void TexMexBench()
{
    ReadData();
    BuildKDTree();

    GroundTruthMatrix gt_vectors(G_GroundTruth.data(), G_GroundTruth.size() / 100, 100);

    {
        clog << "\nQUERYING; #VECTORS=" << G_Query.size() << " " << std::flush;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int qi = 0; qi < G_Query.size(); ++qi)
            EvaluateQuery(G_Query[qi], std::make_pair(gt_vectors(qi, 0), gt_vectors(qi, 1)));
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

static void ReadData()
{
    std::chrono::high_resolution_clock::time_point t0, t1;

    clog << "\nREADING DATA: " << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    auto base = ReadTexMex<float>(G_DataPrefix + "/sift_base.fvecs");
    auto query = ReadTexMex<float>(G_DataPrefix + "/sift_query.fvecs");
    G_GroundTruth = ReadTexMex<int>(G_DataPrefix + "/sift_groundtruth.ivecs");
    t1 = std::chrono::high_resolution_clock::now();
    clog << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << endl;
    ReportMemoryUsage();

    G_Base.resize(base.size() / 128);
    G_Query.resize(query.size() / 128);

    clog << "\nCONVERTING TO U8: " << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    auto conv_fn = [](float x) { return (unsigned char)x; };
    std::transform(base.begin(), base.end(), reinterpret_cast<unsigned char*>(G_Base.data()), conv_fn);
    std::transform(query.begin(), query.end(), reinterpret_cast<unsigned char*>(G_Query.data()), conv_fn);
    t1 = std::chrono::high_resolution_clock::now();
    clog << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << endl;
}

static void BuildKDTree()
{
    clog << "\nBUILDING KDTREE: " << std::flush;
    auto t0 = std::chrono::high_resolution_clock::now();
    G_trees = Build(G_Base.data(), G_Base.size(), TREE_COUNT, LEAF_SIZE);
    auto t1 = std::chrono::high_resolution_clock::now();
    clog << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << endl;
    ReportMemoryUsage();
}

static void EvaluateQuery(const U8Descriptor& q, const std::pair<unsigned, unsigned>& gt)
{
    auto knn = Query2NN(G_trees, q, QUERY_DESCRIPTOR_LIMIT);
    bool gt_sift_accept = SiftMatch(q, G_Base[gt.first], G_Base[gt.second]);
    bool q_sift_accept = SiftMatch(q, G_Base[knn.first], G_Base[knn.second]);

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

static bool SiftMatch(const U8Descriptor& dq, const U8Descriptor& dn1, const U8Descriptor& dn2)
{
    unsigned d1 = L2DistanceSquared(dq, dn1);
    unsigned d2 = L2DistanceSquared(dq, dn2);
    if (d1 > d2) {  // Search is with L1-norm, so this can happen
        ++G_Counters[DISTANCE_SWAPS];
        std::swap(d1, d2);
    }

    return (float)d1 / (float)d2 < 0.8;
}
