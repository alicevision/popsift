#include <iostream>
#include <string>
#include <string.h>
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


DEFINE_int32(device     , 0                                , "selected cuda device");
DEFINE_int32(c1         , 4                                , "number of clusters in first level");
DEFINE_int32(c2         , 4                                , "number of refinements in second level");
DEFINE_int32(p          , 2                                , "parts per vector");
DEFINE_int32(dim        , 128                              , "expected dimension for each vector");
DEFINE_int32(lineparts  , 32                               , "vectorparts for reranking informations");
DEFINE_uint64(chunksize , 10000000                         , "number of vectors per chunk");
DEFINE_uint64(hashsize  , 400000000                        , "maximal number of bins");
DEFINE_string(basename  , "tmp"                            , "prefix for generated data");
DEFINE_string(dataset   , "/home/griff/GIT/popsift-samples/annsift/bigann_query.umem" , "patch to vector dataset");


bool file_exists(const std::string & _name) {
  struct stat buffer;
  return (stat(_name.c_str(), &buffer) == 0);
}

int main(int argc, char* argv[])
{

  // parse flags
  gflags::SetUsageMessage("This script builds a database for a given dataset of vectors\n"
        "Usage:\n"
        "    tool_createdb --c1 4 --c2 4 --p 2 --basename \"tmp\"\n"
        " You should convert the fvecs beforehand using the accompanying convert script\n");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // select cuda device
  cudaSetDevice(FLAGS_device);
  cudaSetDeviceFlags (cudaDeviceMapHost);

  const string preName  = FLAGS_basename + "_" + std::to_string(FLAGS_dim) + "_" + std::to_string(FLAGS_p)
                        + "_" + std::to_string(FLAGS_c1) + "_" + std::to_string(FLAGS_c2);


  // read in dataset
  FileReader<float> DataReader = FileReader<float>(FLAGS_dataset);
  arr<float> data((int) FLAGS_chunksize * FLAGS_dim);
  data.mallocHost();
  data.mallocDevice();
  data.host = DataReader.data(FLAGS_chunksize);
  data.toDevice();

  // building the codebook
  // ==============================================================================================
  int k = 16;
  PerturbationProTree ppt(FLAGS_dim, FLAGS_p, FLAGS_p);
  const string codebook_file = preName + ".ppqt";

  ppt.createTree(FLAGS_c1, FLAGS_c2, data.device, 20000);
  ppt.writeTreeToFile(codebook_file);

  

  const string lineName   = preName + "_" + std::to_string(FLAGS_lineparts) + ".lines";
  const string prefixName = preName + ".prefix";
  const string countsName = preName + ".count";
  const string dbIdxName  = preName + ".dbIdx";

  const uint chunkMax = DataReader.num() / FLAGS_chunksize;
  const uint data_num = chunkMax * FLAGS_chunksize;

  uint* binPrefix = new uint[FLAGS_hashsize];
  uint* binCounts = new uint[FLAGS_hashsize];
  uint* dbIdx = new uint[data_num];

  // building the data base
  // ==============================================================================================
  uint* dbIdxSave = new uint[data_num];

  memset(binPrefix, 0, FLAGS_hashsize * sizeof(uint));
  memset(binCounts, 0, FLAGS_hashsize * sizeof(uint));
  memset(dbIdx, 0, data_num * sizeof(uint));

  uint* chBinPrefix = new uint[FLAGS_hashsize];
  uint* chBinCounts = new uint[FLAGS_hashsize];
  uint* chDBIdx     = new uint[FLAGS_chunksize];
  float* chLines    = new float[FLAGS_chunksize * FLAGS_lineparts];


  ppt.buildKBestDB(data.device, FLAGS_chunksize);
  ppt.lineDist(data.device, FLAGS_chunksize);

  // GPU -> CPU memory
  SAFE_CUDA_CALL(cudaMemcpy(chBinPrefix, ppt.getBinPrefix(), FLAGS_hashsize * sizeof(uint),  cudaMemcpyDeviceToHost));
  SAFE_CUDA_CALL(cudaMemcpy(chBinCounts, ppt.getBinCounts(), FLAGS_hashsize * sizeof(uint),  cudaMemcpyDeviceToHost));
  SAFE_CUDA_CALL(cudaMemcpy(chDBIdx,     ppt.getDBIdx(),     FLAGS_chunksize * sizeof(uint), cudaMemcpyDeviceToHost));
  SAFE_CUDA_CALL(cudaMemcpy(chLines,     ppt.getLine(),      FLAGS_chunksize * FLAGS_lineparts * sizeof(float), cudaMemcpyDeviceToHost));

  ofstream fLines(lineName.c_str(), std::ofstream::out | std::ofstream::binary);
  fLines.write((char*) chLines, FLAGS_chunksize * FLAGS_lineparts * sizeof(float));
  fLines.close();
  cout << "written " << lineName << endl;

  // prefixSum for bin-idx
  ofstream fprefix(prefixName.c_str(), std::ofstream::out | std::ofstream::binary);
  fprefix.write((char*) binPrefix, FLAGS_hashsize * sizeof(uint));
  fprefix.close();
  cout << "written " << prefixName << endl;

  // size of non-empty bins
  ofstream fcounts(countsName.c_str(), std::ofstream::out | std::ofstream::binary);
  fcounts.write((char*) binCounts, FLAGS_hashsize * sizeof(uint));
  fcounts.close();
  cout << "written " << countsName << endl;
  cout << "size: " << (FLAGS_hashsize * sizeof(uint)) << endl;

  // for each bin the ids of containing vectors
  ofstream fdb(dbIdxName.c_str(), std::ofstream::out | std::ofstream::binary);
  fdb.write((char*) dbIdx, data_num * sizeof(uint));
  fdb.close();
  cout << "written " << dbIdxName << endl;

  if (data.device)
    cudaFree(data.device);
  delete[] data.host;


  gflags::ShutDownCommandLineFlags();
  return 0;

}
