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
using SiftMatrix = Map<Matrix<unsigned char, Dynamic, 128, RowMajor>, Eigen::Aligned64>;
using GroundTruthMatrix = Map<Matrix<int, Dynamic, 100, RowMajor>>;

enum Counters {
    FALSE_ACCEPTS,
    FALSE_REJECTS,
    TRUE_ACCEPTS,
    TRUE_REJECTS,
    CORRECT_1_MATCHES,
    CORRECT_2_MATCHES,
    COUNTERS_COUNT
};


static std::string G_DataPrefix = "E:/texmex_sift1M_corpus";
static std::vector<U8Descriptor> G_Base; // 1million descriptors database
static std::vector<U8Descriptor> G_Query; // query descriptors
static std::vector<int> G_GroundTruth; // 100x10k, 100 nearest neighbours for each desc in G_Query
static std::unique_ptr<KDTree> G_kdtree;
static size_t G_Counters[COUNTERS_COUNT];

static void ReadData();
static void BuildKDTree(unsigned leaf_size);
static void EvaluateQuery(size_t qi, const SiftMatrix& base_vectors, const SiftMatrix& query_vectors, const GroundTruthMatrix gt_vectors);

void TexMexBench()
{
    ReadData();
    BuildKDTree(50);    // XXX: guess for leaf size.

    SiftMatrix base_vectors(reinterpret_cast<unsigned char*>(G_Base.data()), G_Base.size() / 128, 128);
    SiftMatrix query_vectors(reinterpret_cast<unsigned char*>(G_Query.data()), G_Query.size() / 128, 128);
    GroundTruthMatrix gt_vectors(G_GroundTruth.data(), G_GroundTruth.size() / 100, 100);

    {
        clog << "\nQUERYING; #VECTORS=" << query_vectors.rows() << " " << std::flush;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int qi = 0; qi < query_vectors.rows(); ++qi)
            EvaluateQuery(qi, base_vectors, query_vectors, gt_vectors);
        auto t1 = std::chrono::high_resolution_clock::now();
        clog << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << endl;
        ReportMemoryUsage();
    }

    clog << "\n\nSTATISTICS:\nFALSE ACCEPTS: " << G_Counters[FALSE_ACCEPTS] << endl;
    clog << "FALSE REJECTS: " << G_Counters[FALSE_REJECTS] << endl;
    clog << "TRUE REJECTS: " << G_Counters[TRUE_REJECTS] << endl;
    clog << "TRUE ACCEPTS: " << G_Counters[TRUE_ACCEPTS] << endl;
    clog << "CORRECT 1-MATCHES: " << G_Counters[CORRECT_1_MATCHES] << endl;
    clog << "CORRECT 2-MATCHES: " << G_Counters[CORRECT_2_MATCHES] << endl;
}

static void ReadData()
{
    using namespace Eigen;

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

static void BuildKDTree(unsigned leaf_size)
{
    auto sdim = GetSplitDimensions(G_Base.data(), G_Base.size());
    G_kdtree.reset(new KDTree(G_Base.data(), G_Base.size()));

    clog << "\nBUILDING KDTREE: " << std::flush;
    auto t0 = std::chrono::high_resolution_clock::now();
    G_kdtree->Build(sdim, leaf_size);
    auto t1 = std::chrono::high_resolution_clock::now();
    clog << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << endl;
    ReportMemoryUsage();
}

static void EvaluateQuery(size_t qi, const SiftMatrix& base_vectors, const SiftMatrix& query_vectors, const GroundTruthMatrix gt_vectors)
{
#if 0
    static std::vector<int_fast32_t> knn;
    auto q_v = query_vectors.row(qi);

    auto gt0_v = base_vectors.row(gt_vectors(qi, 0));
    auto gt1_v = base_vectors.row(gt_vectors(qi, 1));
    float gt0_d = (gt0_v - q_v).norm();
    float gt1_d = (gt1_v - q_v).norm();
    bool gt_sift_accept = gt0_d / gt1_d < 0.8f;

    G_LSH->find_k_nearest_neighbors(q_v, 2, &knn);
    auto lsh0_v = base_vectors.row(knn[0]);
    auto lsh1_v = base_vectors.row(knn[1]);
    float lsh0_d = (lsh0_v - q_v).norm();
    float lsh1_d = (lsh1_v - q_v).norm();
    bool lsh_sift_accept = lsh0_d / lsh1_d < 0.8f;

    if (gt_sift_accept == lsh_sift_accept) {
        if (gt_sift_accept) ++G_Counters[TRUE_ACCEPTS];
        else ++G_Counters[TRUE_REJECTS];
    }
    else {
        if (lsh_sift_accept) ++G_Counters[FALSE_ACCEPTS];
        else ++G_Counters[FALSE_REJECTS];
    }

    if (gt_vectors(qi, 0) == knn[0]) {
        ++G_Counters[CORRECT_1_MATCHES];
        if (gt_vectors(qi, 1) == knn[1])
            ++G_Counters[CORRECT_2_MATCHES];
    }
#endif
}
