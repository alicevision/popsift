#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

#include <string>
#include <vector>
#include <math.h>

/*! \file  ProQuantization.hh
 \brief a more flexible implementation
 */

namespace pqt {

/** \class ProQuantization ProQuantization.hh
 a more flexible implementation */

class ProQuantization {

public:

	/** default constructor */
	ProQuantization(uint _dim, uint _p);

	~ProQuantization();

	void writeCodebookToFile(const std::string& _name);

	void readCodebookFromFile(const std::string& _name);

	/** computes the squared distance between all pairs of A and B looking at product quantization vectors */
	void calcDist(float *_res, const float* _A, const float* _B, uint _Arows,
			uint _Brows) const;

	void calcDist(float *_res, const float* _A, const float* _B, uint _Arows,
				uint _Brows, uint _dim, uint _p) const;

	void testDist(float *_B, uint _Brows);

	void parallelSort(float *_vec, uint _lower, uint _upper);

	void testDistReal(float* _Mh, float *_Md, uint _N);

	/** produces a codebook with _k vectors from _N samples */
	void createCodeBook(uint _k, const float* _A, uint _N);

	/** computes the distance to the codebook aqnd selects for each segment the vector with the smallest distance */
	void getAssignment(uint* _assignd, const float* _A, const float* _B,
			uint _Arows, uint _Brows) const;

	/** computes the distance to the codebook and returns the _k best matches for each segment */
	void getKBestAssignment(uint* _assignd, const float* _A, const float* _B,
				uint _Arows, uint _Brows, uint _k) const;

	void testAssignment(float *_B, uint _Brows);

	void testKBestAssignment(float* _Qh, float *_Qd, uint _QN);

	void getClusterAverage(float *_codebook, float *_count,
			uint* _retirementCount, uint _nClusters, const float *_B, uint _Brows,
			uint* _assign);

	void testAvg(float *_B, uint _Brows);

	/** takes the current vectors in the codebook and doubles their number by creating (1+eps)*v and (1-eps)*v */
	void splitCodeBook(uint &_nClusters, float _epsilon);


	float* getCodeBook() { return d_codeBook; }

	void testCodeBook();


	float calcStatistics(std::vector<float>& _histogram, float* _Q, uint _QN, float *_dbVec,
			uint _nDB, std::vector< std::pair< uint, uint> >& _distSeq);

protected:

	uint d_dim;

	uint d_p;

	uint d_vl;

	uint d_nClusters;


	float* d_codeBook;

};

} /* namespace */

