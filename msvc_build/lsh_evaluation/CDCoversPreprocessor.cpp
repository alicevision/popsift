#include <fstream>
#include <random>
#include <vector>
#include <limits>
#include <algorithm>
#include <iterator>

namespace popsift {
namespace kdtree {

static unsigned GetDescriptorCount(std::ifstream& ifs)
{
    if (ifs.tellg())
        throw std::logic_error("Not at beginning of file");

    ifs.seekg(0, std::ios::end);
    size_t fsz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    if (fsz % 128)
        throw std::runtime_error("invalid file size");

    size_t ret = fsz / 128;
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

void SampleCDCoversDataSet(const std::string& in_fname, const std::string& out_fname, unsigned k, size_t seed)
{
    using namespace std;

    ifstream ifs;
    ifs.exceptions(ios::failbit | ios::badbit | ios::eofbit);
    ifs.open(in_fname, ios::binary);

    unsigned dcount = GetDescriptorCount(ifs);
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

}   // kdtree
}   // popsift
