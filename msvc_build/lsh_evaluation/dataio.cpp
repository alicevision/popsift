#include "dataio.h"
#include <boost/filesystem.hpp>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <tuple>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <Psapi.h>
#endif // _WIN32

// Return vector count and dimension.
static std::tuple<size_t, size_t> GetVectorCountAndDim(std::ifstream& ifs, size_t component_size)
{
    static_assert(sizeof(unsigned) == 4, "unsigned int has invalid size");

    if (ifs.tellg())
        throw std::logic_error("Not at beginning of file");

    unsigned v_dim;
    ifs.read(reinterpret_cast<char*>(&v_dim), 4);

    ifs.seekg(0, std::ios::end);
    size_t fsz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    size_t row_size = 4 + v_dim*component_size;

    if (fsz % row_size)
        throw std::runtime_error("invalid file size");

    return std::make_tuple(fsz / row_size, v_dim);
}

// TexMex descriptors are floats, but each integer in byte range.
static void ReadTexMex(const std::string& fname, std::vector<popsift::kdtree::U8Descriptor>& out_desc)
{
    using namespace std;
        
    ifstream ifs;
    ifs.exceptions(ios::failbit | ios::badbit | ios::eofbit);
    ifs.open(fname, ios::binary);

    size_t v_count, v_dim;
    std::tie(v_count, v_dim) = GetVectorCountAndDim(ifs, sizeof(float));
    if (v_dim != 128)
        throw std::runtime_error("ReadTexMex: dim != 128");

    std::vector<float> row(v_dim);
    out_desc.resize(v_count);

    for (size_t i = 0; i < v_count; ++i) {
        {
            unsigned tmp_dim;
            ifs.read(reinterpret_cast<char*>(&tmp_dim), 4);
            if (tmp_dim != v_dim)
                throw std::runtime_error("non-constant dimension");
        }
        ifs.read(reinterpret_cast<char*>(row.data()), 4*128);
        std::transform(row.begin(), row.end(), out_desc[i].ufeatures.begin(),
            [](float x) { return (unsigned char)x; });
    }
}

// Extract only 2NNs from the GT.
static void ReadTexMex(const std::string& fname, std::vector<std::pair<unsigned, unsigned>>& out_gt)
{
    using namespace std;

    ifstream ifs;
    ifs.exceptions(ios::failbit | ios::badbit | ios::eofbit);
    ifs.open(fname, ios::binary);

    size_t v_count, v_dim;
    std::tie(v_count, v_dim) = GetVectorCountAndDim(ifs, sizeof(unsigned));

    std::vector<unsigned> row(v_dim);
    out_gt.resize(v_count);

    for (size_t i = 0; i < v_count; ++i) {
        {
            unsigned tmp_dim;
            ifs.read(reinterpret_cast<char*>(&tmp_dim), 4);
            if (tmp_dim != v_dim)
                throw std::runtime_error("non-constant dimension");
        }
        ifs.read(reinterpret_cast<char*>(row.data()), 4 * v_dim);
        out_gt[i] = std::make_pair(row[0], row[1]);
    }
}

void ReadTexMex(std::vector<popsift::kdtree::U8Descriptor>& database,
    std::vector<popsift::kdtree::U8Descriptor>& queries,
    std::vector<std::pair<unsigned, unsigned>>& gt)
{
    static std::string G_DataPrefix = "C:/LOCAL/texmex_sift1M_corpus";

    ReadTexMex(G_DataPrefix + "/sift_base.fvecs", database);
    ReadTexMex(G_DataPrefix + "/sift_query.fvecs", queries);
    ReadTexMex(G_DataPrefix + "/sift_groundtruth.ivecs", gt);
}


/////////////////////////////////////////////////////////////////////////////

// Format of the descriptor file format: 8-byte count, followed by flat U8 list.
// See saveDescsToBinFile here: https://github.com/alicevision/openMVG/blob/popart_develop/src/openMVG/features/descriptor.hpp
std::tuple<std::vector<unsigned>, std::vector<float>>
ReadOpenMVGU8Database(const std::string& dir)
{
    static_assert(sizeof(size_t) == 8, "Must compile in 64-bit mode");

    std::vector<unsigned> counts;
    std::vector<float> data;

    {
        boost::filesystem::directory_iterator it(dir), end;
        size_t count;

        while (it != end) {
            auto de = *it++;
            if (de.path().extension() != ".desc")
                continue;
            size_t fsz = file_size(de.path());

            std::ifstream ifs;
            ifs.exceptions(std::ios::badbit | std::ios::failbit | std::ios::eofbit);
            ifs.open(de.path().string(), std::ios::binary);
            ifs.read(reinterpret_cast<char*>(&count), 8);
            if (fsz != 8 + count * 128)
                throw std::runtime_error("ReadOpenMVGU8Database: invalid file size");
            
            data.reserve(128 * count);
            counts.push_back(static_cast<unsigned>(count));

            std::transform(
                std::istreambuf_iterator<char>(ifs),
                std::istreambuf_iterator<char>(),
                std::back_inserter(data),
                [](unsigned char ch) { return (float)ch; }
            );
        }
    }

    for (size_t i = 1; i < counts.size(); ++i)
        counts[i] += counts[i - 1];

    return std::make_tuple(counts, data);
}

unsigned MapDescriptorToImage(const std::vector<unsigned>& image_d_counts, unsigned di)
{
    auto it = std::lower_bound(image_d_counts.begin(), image_d_counts.end(), di);
    auto i = it - image_d_counts.begin();
    return static_cast<unsigned>(i);
}

/////////////////////////////////////////////////////////////////////////////

void ReportMemoryUsage()
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS mc;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &mc, sizeof(mc)))
        throw std::runtime_error("GetProcessMemoryInfo failed");
    std::clog << "MEMORY PEAKS (MB): " << "WSZ: " << (mc.PeakWorkingSetSize >> 20);
    std::clog << "; CCHARGE: " << (mc.PagefileUsage >> 20) << std::endl;
#endif
}
