#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

#include <vector>
#include <math.h>

/*! \file  ProductQuantization.hh
    \brief implements product quantization following the paper by Cordula Schmid
 */

namespace pqtPQ {

  /** \class ProductQuantization ProductQuantization.hh
      implements product quantization following the paper by Cordula Schmid
      Each vector is separated into p sections, a vector quantization is performed for each of them. */
  
  class ProductQuantization {

  public:
	  /** default constructor */
	     ProductQuantization(uint _dim, uint _p);

	     ~ProductQuantization();

	     /** computes the squared distance between all pairs of A and B */
	     void calcDist(float *_res, const float* _A, const float* _B, uint _Arows, uint _Brows, uint _dim ) const;

	     /** produces a codebook with _k vectors from _N samples */
	     void createCodeBook( uint _k, const float* _A, uint _N);

	     /** given the distance matrix produces for each vector the index with the smallest distance */
	     void getAssignment(uint* _assignd, const float* _distMat, uint _N, uint _nClusters) const;

	     /** takes the current vectors in the codebook and doubles their number by creating (1+eps)*v and (1-eps)*v */
	     void splitCodeBook(uint &_nClusters, float _epsilon);

	   protected:

	     uint d_dim;

	     uint d_p;

	     float* d_codeBook;


  };


} /* namespace */

