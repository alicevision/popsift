#include "dataio.h"
#include "KDTree.h"
#include <fstream>
#include <random>
#include <vector>
#include <limits>
#include <algorithm>
#include <iterator>
#include <tbb/parallel_for.h>

static unsigned GetItemCount(std::ifstream& ifs, size_t item_size)
{
    if (ifs.tellg())
        throw std::logic_error("Not at beginning of file");

    ifs.seekg(0, std::ios::end);
    size_t fsz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    if (fsz % item_size)
        throw std::runtime_error("invalid file size");

    size_t ret = fsz / item_size;
    if (ret >= std::numeric_limits<unsigned>::max())
        throw std::length_error("Too many descriptors in file");

    return static_cast<unsigned>(ret);
}

// Returns a sorted sample of k random records of n.
static std::vector<unsigned> ReservoirSample(unsigned n, unsigned k, size_t seed)
{
    std::mt19937_64 engine(seed);
    std::vector<unsigned> sample(k);
    
    for (unsigned i = 0; i < k; ++i) sample[i] = i;
    for (unsigned i = k; i < n; ++i) {
        std::uniform_int_distribution<unsigned> dist(0, i-1);
        unsigned j = dist(engine);
        if (j < k) sample[j] = i;
    }
    std::sort(sample.begin(), sample.end());
    return sample;
}

// Select a sample of k descriptors from the CD covers data set and write it out to a file.
void SampleCDCoversDataSet(const std::string& in_fname, const std::string& out_fname, unsigned k, size_t seed)
{
    using namespace std;

    ifstream ifs;
    ifs.exceptions(ios::failbit | ios::badbit | ios::eofbit);
    ifs.open(in_fname, ios::binary);

    unsigned dcount = GetItemCount(ifs, 128);
    if (dcount >= std::numeric_limits<unsigned>::max())
        throw std::length_error("Too many descriptors in file");
    if (k >= dcount)
        throw std::out_of_range("k >= descriptor count");
    
    auto sample = ReservoirSample(dcount, k, seed);

    ofstream ofs;
    ofs.exceptions(ios::failbit | ios::badbit | ios::eofbit);
    ofs.open(in_fname, ios::binary);

    std::vector<unsigned char> descriptor(128);

    for (unsigned di : sample) {
        ifs.seekg(di * 128);
        std::copy_n(std::istream_iterator<unsigned char>(ifs), 128, std::ostream_iterator<unsigned char>(ofs));
    }
}

template<typename T>
static std::vector<T> SlurpBinaryFile(const std::string& fname)
{
    using namespace std;

    ifstream ifs;
    ifs.exceptions(ios::failbit | ios::badbit | ios::eofbit);
    ifs.open(fname, ios::binary);

    unsigned count = GetItemCount(ifs, sizeof(T));
    std::vector<T> ret(count);
    ifs.read(reinterpret_cast<char*>(ret.data()), sizeof(T) * count);
    return ret;
}

std::vector<popsift::kdtree::U8Descriptor> ReadCDCoversVectors(const std::string& fname)
{
    return SlurpBinaryFile<popsift::kdtree::U8Descriptor>(fname);
}

std::vector<std::pair<unsigned, unsigned>> ReadCDCoversGT(const std::string& fname)
{
    return SlurpBinaryFile<std::pair<unsigned, unsigned>>(fname);
}

//! Outputs a binary file containing 2 unsigned ints with nearest neighbors in the database for each test vector.
void CalculateGroundTruth(const std::string& db_fname, const std::string& test_fname, const std::string& out_fname)
{
    using namespace std;

    auto dbv = ReadCDCoversVectors(db_fname);
    auto qv = ReadCDCoversVectors(test_fname);

    ofstream ofs;
    ofs.exceptions(ios::failbit | ios::badbit | ios::eofbit);
    ofs.open(out_fname, ios::binary);

    std::vector<std::pair<unsigned, unsigned>> gt(qv.size());
    
    using namespace popsift::kdtree;
    tbb::parallel_for((size_t)0, qv.size(), [&](size_t i) {
        Q2NNAccumulator acc;
        for (int x = 0; x < dbv.size(); x++) {
            unsigned dist = L2DistanceSquared(qv[i], dbv[x]);
            acc.Update(dist, x);
        }
        gt[i].first = acc.index[0];
        gt[i].second = acc.index[1];
    });
    ofs.write(reinterpret_cast<const char*>(gt.data()), gt.size() * sizeof(gt.front()));
}
