#include "dataio.h"
#include <iostream>
#include <chrono>
#include <falconn/lsh_nn_table.h>

using namespace falconn;
using std::clog;
using std::endl;
using SiftPoint = falconn::DenseVector<float>;
using SiftMatrix = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using GroundTruthMatrix = Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

enum Counters {
    FALSE_ACCEPTS,
    FALSE_REJECTS,
    TRUE_ACCEPTS,
    TRUE_REJECTS,
    CORRECT_1_MATCHES,
    CORRECT_2_MATCHES,
    COUNTERS_COUNT
};

static const int NUM_HASH_TABLES = 8;  // 10-ish is a good start
static const int NUM_HASH_BITS = 22;   // should be about log2(n_points); but memory usage is proportional with that!

//http://corpus-texmex.irisa.fr/ ANN_SIFT1M set
static std::string G_DataPrefix = "C:/LOCAL/texmex_sift1M_corpus";
static std::vector<float> G_Base; // 1million descriptors database
static std::vector<float> G_Query; // query descriptors
static std::vector<int> G_GroundTruth; // 100x10k, 100 nearest neighbours for each desc in G_Query
static std::unique_ptr<LSHNearestNeighborTable<SiftPoint>> G_LSH;
static size_t G_Counters[COUNTERS_COUNT];

static void ReadData();
static void NormalizeDataSet(SiftMatrix&);
static void BuildLSH();
static void EvaluateQuery(size_t qi, const SiftMatrix& base_vectors, const SiftMatrix& query_vectors, const GroundTruthMatrix gt_vectors);

void TexMexBench()
{
    ReadData();
    BuildLSH();
    G_LSH->set_num_probes(NUM_HASH_TABLES); // This is minimal acceptable value

    SiftMatrix base_vectors(G_Base.data(), G_Base.size() / 128, 128);
    SiftMatrix query_vectors(G_Query.data(), G_Query.size() / 128, 128);
    GroundTruthMatrix gt_vectors(G_GroundTruth.data(), G_GroundTruth.size() / 100, 100);

    {
        clog << "\nQUERYING; #VECTORS=" << query_vectors.rows() << " " << std::flush;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (size_t qi = 0; qi < query_vectors.rows(); ++qi)
            EvaluateQuery(qi, base_vectors, query_vectors, gt_vectors);
        auto t1 = std::chrono::high_resolution_clock::now();
        clog << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << endl;
        ReportMemoryUsage();
    }

    {
        auto st = G_LSH->get_query_statistics();
        clog << "\nLSH AVERAGES:" << endl;
        clog << "total query time: " << st.average_total_query_time << endl;
        clog << "lsh time: " << st.average_lsh_time << endl;
        clog << "hash table time: " << st.average_hash_table_time << endl;
        clog << "distance time: " << st.average_distance_time << endl;
        clog << "#candidates: " << st.average_num_candidates << endl;
        clog << "#unique_cand: " << st.average_num_unique_candidates << endl;
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
    G_Base = ReadTexMex<float>(G_DataPrefix + "/sift_base.fvecs");
    G_Query = ReadTexMex<float>(G_DataPrefix + "/sift_query.fvecs");
    G_GroundTruth = ReadTexMex<int>(G_DataPrefix + "/sift_groundtruth.ivecs");
    t1 = std::chrono::high_resolution_clock::now();
    clog << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << endl;
    ReportMemoryUsage();

    clog << "NORMALIZING DATA (base set and queries): " << std::flush;
    t0 = std::chrono::high_resolution_clock::now();
    NormalizeDataSet(SiftMatrix(G_Base.data(), G_Base.size() / 128, 128));
    NormalizeDataSet(SiftMatrix(G_Query.data(), G_Query.size() / 128, 128));
    t1 = std::chrono::high_resolution_clock::now();
    clog << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << endl;
}

static void NormalizeDataSet(SiftMatrix& siftm)
{
    auto norms = siftm.rowwise().norm().array();
    auto sifta = siftm.array();
    sifta.colwise() /= norms;
}

static void BuildLSH()
{
    LSHConstructionParameters params;
    params.dimension = 128;
    params.lsh_family = LSHFamily::CrossPolytope;
    params.l = NUM_HASH_TABLES;
    params.distance_function = DistanceFunction::NegativeInnerProduct;
    compute_number_of_hash_functions<SiftPoint>(NUM_HASH_BITS, &params);
    params.num_rotations = 2;       // 1 for dense, 2 for sparse data
    params.num_setup_threads = 0;   // use all available
    params.storage_hash_table = StorageHashTable::BitPackedFlatHashTable;

    clog << "\nBUILDING TABLE: " << std::flush;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto pas = PlainArrayPointSet<float>{ G_Base.data(), static_cast<int>(G_Base.size() / 128), 128 };
    G_LSH = construct_table<SiftPoint>(pas, params);
    auto t1 = std::chrono::high_resolution_clock::now();
    clog << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << endl;
    ReportMemoryUsage();
}

static void EvaluateQuery(size_t qi, const SiftMatrix& base_vectors, const SiftMatrix& query_vectors, const GroundTruthMatrix gt_vectors)
{
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
}


