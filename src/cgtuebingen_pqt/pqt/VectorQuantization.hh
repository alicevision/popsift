#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

#include <vector>
#include <math.h>


/*! \file  VectorQuantization.hh
    \brief implements a GPU-based vector quantization
 */

namespace pqt {

  /** \class VectorQuantization VectorQuantization.hh
      implements a GPU-based vector quantization */
  
  class VectorQuantization {

  public:

    /** default constructor */
    VectorQuantization(uint _dim);

    ~VectorQuantization();

    /** computes the squared distance between all pairs of A and B */
    void calcDist(float *_res, const float* _A, const float* _B, uint _Arows, uint _Brows, uint _dim ) const;

    /** produces a codebook with _k vectors from _N samples */
    void createCodeBook( uint _k, const float* _A, uint _N);

    /** given the distance matrix produces for each vector the index with the smallest distance */
    void getAssignment(uint* _assignd, const float* _distMat, uint _N, uint _nClusters) const;

    /** takes the current vectors in the codebook and doubles their number by creating (1+eps)*v and (1-eps)*v */
    void splitCodeBook(uint &_nClusters, float _epsilon);

    /** estimates the maximum radius for each cluster */
    void getMaxRad(float *_maxRad, uint _nCluster, const uint* _assign, uint _N, const float* _distMat) const;

  protected:

    uint d_dim;

    float* d_codeBook;


  };


} /* namespace */

