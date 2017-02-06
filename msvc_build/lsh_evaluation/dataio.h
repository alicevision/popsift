#pragma once

#include "KDTree.h"
#include <vector>
#include <string>
#include <tuple>

template<typename T>
std::vector<T> ReadTexMex(const std::string& fname);

// Reads all descriptors from OpenMVG reconstruction directory; assumes uchar
// SIFT descriptors.  Returns two vectors:
// - the first has as many elements as there are images; it contains cumulative counts
//   of descriptors in all images up to and including the image
// - the second contains a flat list of descriptors
std::tuple<std::vector<unsigned>, std::vector<float>>
ReadOpenMVGU8Database(const std::string& dir);

unsigned MapDescriptorToImage(const std::vector<unsigned>& image_d_counts, unsigned di);

void SampleCDCoversDataSet(const std::string& in_fname, const std::string& out_fname, unsigned k, size_t seed);
std::vector<popsift::kdtree::U8Descriptor> ReadCDCoversVectors(const std::string& fname);
std::vector<std::pair<unsigned, unsigned>> ReadCDCoversGT(const std::string& fname);
void CalculateGroundTruth(const std::string& db_fname, const std::string& test_fname, const std::string& out_fname);
void PrepareCDCoversDataSet();

void ReportMemoryUsage();
