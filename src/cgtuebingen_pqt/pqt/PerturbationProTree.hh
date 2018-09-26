#ifndef NEARESTNEIGHBOR_PERTURBATIONPROTREE_H
#define NEARESTNEIGHBOR_PERTURBATIONPROTREE_H

/*! \file  PerturbationProTree.hh
 \brief implements a product quantization tree with perturbation, i.e. various permuations on the input vectors to boost the recogntion rate
 */
#include "helper.hh"
#include "ProTree.hh"
#include "ProQuantization.hh"

#define USE_HASH 1
#define HASH_SIZE 400000000


namespace pqt {


/** \class PerturbationProTree PerturbationProTree.hh
 implements a product quantization tree with perturbation, i.e. various permuations on the input vectors to boost the recogntion rate */

typedef struct {
  char p1;
  char p2;
  ushort lambda;
} lineDescr;


class PerturbationProTree: public ProTree {

  __device__ uint pertIdx(uint _i, uint _dimBits, uint _cb);




public:

  PerturbationProTree(uint _dim, uint _p, uint _p2);

  ~PerturbationProTree();

  void writeTreeToFile(const std::string& _name);
  void readTreeFromFile(const std::string& _name);

  /** produces a two-layer product quantization tree with _k basis vectors on each layer */
  void createTree(uint _k, uint _k2, const float* _A, uint _N);


  void createTreeSplitSparse(uint _nClusters1, uint _nClusters2,
      const float* _A, uint _N, bool _sparse);


  /** takes all vectors an _A and sorts their index into the corresponding leaf */
  void buildDB(const float* _A, uint _N);

  /** as above but tries k1 best clusters on the first level */
  void buildKBestDB(const float* _A, uint _N);

  /** only consider the elements that fall into sparse / coarse bins on the first level */
  void buildKBestDBSparse(const float* _A, uint _N,
      bool _sparse);

  /** performs product tree quantization and calculates the line quantizatoin to level 1 */
  void buildKBestLineDB(const float* _A, uint _N);

  /** upload a previously stored db. pointers should be host pointers */
  void setDB(uint _N, const uint* _prefix, const uint* _counts, const uint* _dbIdx);


  /** looks for some nearest neighbors */
  void testKNN(const float* _Q, uint _QN);

  void queryKNN( std::vector<uint>& _resIdx, std::vector<float>& _resDist, const float* _Q, uint _QN, uint _nVec);

  void queryBIGKNN( std::vector<uint>& _resIdx, std::vector<float>& _resDist, const float* _Q, uint _QN, uint _nVec, const std::vector<uint>& _gtBins, uint _offset);

  /** performs line reranking, fetching line data from host memory */
  void queryBIGKNNRerank( std::vector<uint>& _resIdx, std::vector<float>& _resDist, const float* _Q, uint _QN, uint _nVec, const float* _hlines);


  /** performs line reranking, fetching line data from host memory, assumes all line data is stored in pinned memory such that the GPU can do a direct access */
  void queryBIGKNNRerank2( std::vector<uint>& _resIdx, std::vector<float>& _resDist, const float* _Q, uint _QN, uint _nVec, const float* _hlines);

  /** performs line reranking, fetching original vectors from host memory, assumes all vector data is stored in pinned memory such that the GPU can do a direct access */
  void queryBIGKNNRerankPerfect( std::vector<uint>& _resIdx, std::vector<float>& _resDist, const float* _Q, uint _QN, uint _nVec, const float* _hlines);

  uint getNPerturbations() const { return d_nDBs; }

  void testLineDist(const float* _assignVal, const uint* assignIdx, uint _k1, uint _N);
  void testSortLarge();




  void lineDist(const float* _DB, uint _N);


  uint* getBinPrefix() { return d_binPrefix; }
  uint* getBinCounts() { return d_binCounts; }
  uint* getDBIdx() { return d_dbIdx; }

  float* getLine() { return d_lineLambda; }

  void prepareEmptyLambda( uint _N, uint _lParts = 16) { d_lineParts = _lParts; cudaMalloc(&d_lineLambda, _N * _lParts * sizeof(float) ); }

  void getAssignment(uint *_assign, const float* _A, const float* _B,
      uint _Arows, uint _Brows) const;

protected:

  void perturbVectors(float* _pertA, const float* _A, uint _N, uint _pert);

  void getAssignment2(uint *_assign2, const float* _A, const float* _B,
      uint _Arows, uint _Brows, const uint *_assign1,
      uint _nClusters1) const;

  /** returns the k best level 1 matches for each part and each DB */
  void getKBestAssignment(uint *_assign, const float* _A,
      const float* _B, uint _Arows, uint _Brows, uint _k) const;

  /** returns the k best level 1 matches for each part and each DB, stores the distances to all level 1 of the first DB */
  void getKBestLineAssignment(uint *_assign, float* _l1Dist, const float* _A,
        const float* _B, uint _Arows, uint _Brows, uint _k) const;

  void getKBestAssignment2(float *_assignVal,
      uint *_assignIdx, const float* _A, const float* _B, uint _Arows,
      uint _Brows, const uint *_assign1, uint _nClusters1, uint _k1) const;

  void countBins(uint* _bins, const uint* _assign, uint* _assign2, uint _N);

  void getBins(uint *_bins, uint *_nBins, const float *_assignVal,
        const uint *_assignIdx, uint _N, uint _k1, uint _k2, uint _maxBins);


  void getBIGBins(uint *_bins, uint *_nBins, const float *_assignVal,
          const uint *_assignIdx, uint _N, uint _k1, uint _k2, uint _maxBins);

  void getBIGBinsSorted(uint *_bins, uint *_nBins, const float *_assignVal,
            const uint *_assignIdx, uint _N, uint _k1, uint _k2, uint _maxBins);


  void getBIGBins2D(uint *_bins, uint *_nBins, const float *_assignVal,
            const uint *_assignIdx, uint _N, uint _k1, uint _k2, uint _maxBins);

  /** assuming an array of bin indices */
  void countBins(uint* _bins, const int* _assignedBins,
      uint _N);

  /** assuming an array of bin indices */
  void sortIdx(uint* _dbIdx, const int* _assignedBins,
      uint _N);

  void getKBestVectors(float *_bestDist, uint *_bestIdx,
      const uint *_bins, const uint *_nBins, uint _maxBins, const float* _Q,
      uint _QN, uint _k);

  void getKBestVectorsLarge(float *_bestDist, uint *_bestIdx,
      const uint *_bins, const uint *_nBins, uint _maxBins, const float* _Q,
      uint _QN, uint _k);


  /** computes the best bin given the k1-best assignments on the first level */
  void getBestBinAssignment2(int *_assignBin,
      const float* _cb2, const float* _B, uint _nClusters2, uint _Brows,
      const uint *_assign1, uint _k1, uint _nClusters1) const;

  void getBestBinAssignment2Sparse(int *_assignBin,
      const float* _cb2, const float* _B, uint _nClusters2, uint _Brows,
      const uint *_assign1, uint _k1, uint _nClusters1, bool _sparse) const;

  /** as above but stores distance to selected cluster */
  void getBestBinLineAssignment2(int *_assignBin, uint *_l2Idx, float *_l2Dist,
        const float* _cb2, const float* _B, uint _nClusters2, uint _Brows,
        const uint *_assign1, uint _k1, uint _nClusters1) const;

  /** for each query vector compute and store the distances to all cluster centers */
  void getLineAssignment(float* _queryL1Dist, const float* _Q, uint _QN);

  void computeCBDist();

  void computeCBL1L2Dist();

  /** compute the distances between all clusters centers for _nParts */
  void computeCBL1L1Dist(uint _nParts);

  void assembleLines( const float* _l1Dist, uint* _l2Idx, const float* _l2Dist, uint _N);

  void rerankKBestVectors(float *_bestDist, uint *_bestIdx, const float* _queryL1,
        const uint *_bins, const uint *_nBins, uint _maxBins, const float* _Q,
        uint _QN, uint _k);


  void rerankBIGKBestVectors(float *_bestDist, uint *_bestIdx, const float* _queryL1,
        const uint *_bins, const uint *_nBins, uint _maxBins, const float* _Q,
        uint _QN, uint _k);

  void rerankBIGKBestVectors(vector<uint>& _resIdx, float *_bestDist, uint *_bestIdx, const float* _queryL1,
        const uint *_bins, const uint *_nBins, uint _maxBins, const float* _Q,
        uint _QN, uint _k, const float* _hLines);

  // with pinned memory
  void rerankBIGKBestVectors2(float *_bestDist, uint *_bestIdx, const float* _queryL1,
        const uint *_bins, const uint *_nBins, uint _maxBins, const float* _Q,
        uint _QN, uint _k, const float* _hLines);

  void rerankBIGKBestVectorsPerfect(float *_bestDist, uint *_bestIdx,
          const uint *_bins, const uint *_nBins, uint _maxBins, const float* _Q,
          uint _QN, uint _k, const float* _hLines);


  /* combines geBins und reranking in one method  --- did not work well  */
  void rerankKBestBinVectors(float *_bestDist,
      uint *_bestIdx, const float* _queryL1, const float* _assignVal,
      const uint* _assignIdx, uint _maxBins, uint _k1, uint k2,
      const float* _Q, uint _QN, uint _k);


  uint d_dimBits;            // number of bits of the vector dimension
  float* d_multiCodeBook;    // concatenation of the codebooks of each db
  float* d_multiCodeBook2;   // concatenation of the codebooks2 of each db

  float* d_codeBookDistL1L2; // distances between all L1 and all L2 cluster centers
  float* d_codeBookDistL2;   // distances between all L2 cluster centers

  float* d_lineLambda;       // array of lambdas obtained by line projection, d_p for each db point
  uint*  d_lineIdx;          // array of L1 indices, d_p for each db point
  uint*  d_l2Idx;            // array of L2 indices, d_p for each db point

  uint*  d_lineP1;           // array of L1 indices, d_p for each db point
  uint*  d_lineP2;           // array of L2 indices, d_p for each db point
  uint   d_lineParts;


  void sortIdx(uint* _dbIdx, const uint* _assign, const uint* _assign2,
      uint _N);
};

} /* namespace */

#endif /* NEARESTNEIGHBOR_PERTURBATIONPROTREE_H */
