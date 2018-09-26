#include <iostream>
#include <string>
#include <fstream>
#include <sys/stat.h>

#include "pqt/VectorQuantization.hh"
#include "pqt/ProductQuantization.hh"
#include "pqt/ProQuantization.hh"
#include "pqt/ProTree.hh"
#include "pqt/PerturbationProTree.hh"


#include "utils/filereader.hpp"
#include "utils/arr.hpp"
#include <gflags/gflags.h>


using namespace pqt;
using namespace pqtPQ;


DEFINE_int32(device     , 0                                                           , "selected cuda device");
DEFINE_int32(c1         , 4                                                           , "number of clusters in first level");
DEFINE_int32(c2         , 4                                                           , "number of refinements in second level");
DEFINE_int32(p          , 2                                                           , "parts per vector");
DEFINE_int32(dim        , 128                                                         , "expected dimension for each vector");
DEFINE_int32(lineparts  , 32                                                          , "vectorparts for reranking informations");
DEFINE_uint64(chunksize , 100000                                                      , "number of vectors per chunk");
DEFINE_uint64(hashsize  , 400000000                                                   , "maximal number of bins");
DEFINE_string(basename  , "tmp"                                                       , "prefix for generated data");
DEFINE_string(dataset   , "/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1M/base.umem" , "patch to vector dataset");
DEFINE_string(queryset   , "/graphics/projects/scratch/ANNSIFTDB/ANN_SIFT1M/query.umem" , "patch to vector dataset");


bool file_exists(const std::string & _name) {
  struct stat buffer;
  return (stat(_name.c_str(), &buffer) == 0);
}

uint locate(uint _baseNum, const uint* _prefix, const uint* _counts,
            const uint* _dbIdx, uint _idx) {

  int pos;
  for (pos = 0; pos < _baseNum; pos++) {
    if (_dbIdx[pos] == _idx)
      break;
  }

  int bin;
  for (bin = 0; bin < HASH_SIZE - 1; bin++) {
    if (_prefix[bin + 1] > pos)
      break;
  }


  return bin;
}

int main(int argc, char* argv[]) {

  // parse flags
  gflags::SetUsageMessage("This script builds a database for a given dataset of vectors\n"
                          "Usage:\n"
                          "    tool_createdb --c1 4 --c2 4 --p 2 --basename \"tmp\""
                          " You should convert the fvecs beforehand using the accompanying convert script\n");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // select cuda device
  cudaSetDevice(FLAGS_device);
  cudaSetDeviceFlags (cudaDeviceMapHost);

  const string preName  = FLAGS_basename + "_" + std::to_string(FLAGS_dim) + "_" + std::to_string(FLAGS_p)
                          + "_" + std::to_string(FLAGS_c1) + "_" + std::to_string(FLAGS_c2);


  FileReader<float> DataReader = FileReader<float>(FLAGS_dataset);

  FileReader<float> QueryReader = FileReader<float>(FLAGS_queryset);
  arr<float> query = arr<float>(FLAGS_chunksize * FLAGS_dim);
  query.mallocHost();
  query.mallocDevice();
  query.host = QueryReader.data(FLAGS_chunksize);
  query.toDevice();


  PerturbationProTree ppt(FLAGS_dim, FLAGS_p, FLAGS_p);
  const string codebook_file = preName + ".ppqt";

  if (!file_exists(codebook_file)) {
    cout << "you need to generate a codebook first. No codebook found in " << codebook_file << endl;
    return 1;
  } else {
    cout << "codebook exists, reading from " << codebook_file << endl;
    ppt.readTreeFromFile(codebook_file);
  }

  const uint base_num = DataReader.num();

  const string lineName   = preName + "_" + std::to_string(FLAGS_lineparts) + ".lines";
  const string prefixName = preName + ".prefix";
  const string countsName = preName + ".count";
  const string dbIdxName  = preName + ".dbIdx";

  const uint chunkMax = base_num / FLAGS_chunksize;
  const uint data_num = chunkMax * FLAGS_chunksize;

  uint* binPrefix = new uint[FLAGS_hashsize];
  uint* binCounts = new uint[FLAGS_hashsize];
  uint* dbIdx     = new uint[data_num];

  // read data base
  ifstream fprefix(prefixName.c_str(), std::ifstream::in | std::ofstream::binary);
  fprefix.read((char*) binPrefix, FLAGS_hashsize * sizeof(uint));
  fprefix.close();
  cout << "read " << prefixName << endl;

  ifstream fcounts(countsName.c_str(), std::ofstream::in | std::ofstream::binary);
  fcounts.read((char*) binCounts, HASH_SIZE * sizeof(uint));
  fcounts.close();
  cout << "read " << countsName << endl;

  size_t nfloats = DataReader.num();
  nfloats *= FLAGS_lineparts;

  float* hLines = nullptr;
  float* dLines;

  cudaHostAlloc((void **) &hLines, nfloats * sizeof(float), cudaHostAllocMapped);
  cudaHostGetDevicePointer((void **) &dLines, (void *) hLines, 0);
  if (!hLines) {
    cerr << " did not get hLine memory " << endl;
    exit(1);
  }

  ifstream fdb(dbIdxName.c_str(), std::ifstream::in | std::ofstream::binary);
  fdb.read((char*) dbIdx, base_num * sizeof(uint));
  fdb.close();
  cout << "read " << dbIdxName << endl;


  ppt.setDB(base_num, binPrefix, binCounts, dbIdx);

  // query
  vector<uint> resIdx;
  vector<float> resDist;
  
  for (int idxA = 0; idxA < FLAGS_chunksize; idxA += 4096) {
    const int len = min(4096, (int)(FLAGS_chunksize - idxA));
    ppt.queryKNN(resIdx, resDist, query.device + 4096 * idxA * FLAGS_dim, len, 4096);
    for (int r = 0; r < len; ++r) {
      const int queryVectorId = idxA*4096 + r;
      const int bestfoundBaseVectorId = resIdx[4096*r];
      const int secondbestfoundBaseVectorId = resIdx[4096*r+1];
    }
  }

  gflags::ShutDownCommandLineFlags();
  return 0;

}
