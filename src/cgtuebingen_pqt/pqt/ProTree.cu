#ifndef NEARESTNEIGHBOR_PROTREE_C
#define NEARESTNEIGHBOR_PROTREE_C

#include <algorithm>
#include <fstream>
#include "ProTree.hh"

#include <sys/stat.h>

using namespace std;


#define OUTPUT
#include "helper.hh"

#include <assert.h>

namespace pqt {

/** default constructor */

ProTree::ProTree(uint _dim, uint _p, uint _p2, uint _groupParts) :
		ProQuantization(_dim, _p), d_p2(_p2), d_groupParts(_groupParts), d_vl2(
				_dim / _p2), d_codeBook2(NULL), d_binCounts(NULL), d_binPrefix(
		NULL), d_distSeq(NULL), d_dbVec(NULL), d_NdbVec(0), d_sparseBin(NULL) {
	cudaMalloc(&d_count, sizeof(uint));
}

ProTree::ProTree(uint _dim, uint _p, uint _p2) :
		ProQuantization(_dim, _p), d_p2(_p2), d_groupParts(_p2), d_vl2(
				_dim / _p2), d_codeBook2(NULL), d_binCounts(NULL), d_binPrefix(
		NULL), d_distSeq(NULL), d_dbVec(NULL), d_NdbVec(0), d_sparseBin(NULL) {
	cudaMalloc(&d_count, sizeof(uint));
}

ProTree::~ProTree() {
	if (d_codeBook2)
		cudaFree(d_codeBook2);
	if (d_binPrefix)
		cudaFree(d_binPrefix);
	if (d_binCounts)
		cudaFree(d_binCounts);
	if (d_distSeq)
		cudaFree(d_distSeq);
	if (d_sparseBin)
		cudaFree(d_sparseBin);
	cudaFree(d_count);

}

void ProTree::prepare2DDistSequence(int _maxCluster) {

	if (d_distSeq)
		cudaFree(d_distSeq);

	uint nVec = pow(_maxCluster, 2);

	int copyVec = (nVec < NUM_DISTSEQ) ? nVec : NUM_DISTSEQ;

	d_numDistSeq = nVec;
	d_distCluster = _maxCluster;

	cout << "prepareDist2D, nVec: " << nVec << endl;

	uint* distseq = new uint[NUM_DISTSEQ * NUM_ANISO_DIR];

	for (int slope = 0; slope < NUM_ANISO_DIR; slope++) {

		float s = pow(0.9 * ANISO_BASE, slope - (NUM_ANISO_DIR / 2));
		cout << "slope: " << s << endl;

		vector<pair<float, uint> > dists;

		dists.clear();

		for (int i = 0; i < nVec; i++) {

			float x = i % _maxCluster;
			float y = i / _maxCluster;

			float dist;

			dist = x + s * y;

//			dist = sqrt(x) + s * sqrt(y);
//
//			if ((x < 2) || (y < 2))
//				dist *= 0.3;

			float n = 0.8; // 93% 16 8

			dist = pow(x, n) + s * pow(y, n);

//			float dist;
//			if (x < y)
//				dist = 5 * y + x;
//			else
//				dist = 5 * x + y;

//			dist = x + y;

			dists.push_back(pair<float, uint>(dist, i));

		}

		std::sort(dists.begin(), dists.end());

		for (int i = 0; i < 100; i++) {
			cout << "\t" << dists[i].second;
		}
		cout << endl;

		// copy to big array
		for (int i = 0; i < NUM_DISTSEQ; i++)
			distseq[slope * NUM_DISTSEQ + i] = 0;

		for (int i = 0; i < copyVec; i++)
			distseq[slope * NUM_DISTSEQ + i] = dists[i].second;
	}

	cudaMalloc(&d_distSeq, NUM_DISTSEQ * NUM_ANISO_DIR * sizeof(float));

	cudaMemcpy(d_distSeq, distseq, NUM_DISTSEQ * NUM_ANISO_DIR * sizeof(float),
			cudaMemcpyHostToDevice);

	delete[] distseq;
}

void ProTree::prepareDistSequence(int _maxCluster, int _groupParts) {

	// creates all integer vectors from 0 to _dim^d_p
	// sorts them by distance to the origin
	// uploads the sequence to the GPU

	if (d_distSeq)
		cudaFree(d_distSeq);

	_maxCluster = (_maxCluster > 16) ? 16 : _maxCluster;

	uint nVec = pow(_maxCluster, _groupParts);

	cout << "prepareDist, nVec: " << nVec << endl;

//	vector<pair<uint, uint> > dists;
	vector<pair<float, uint> > dists;

	vector<uint> denom;

	denom.resize(_groupParts);

	denom[0] = 1;
	for (int p = 1; p < _groupParts; p++) {
		denom[p] = denom[p - 1] * _maxCluster;
	}

	for (int i = 0; i < nVec; i++) {

//		uint dist = 0;

		float dist = 0;
		for (int p = 0; p < _groupParts; p++) {
			uint val = (i / denom[p]) % _maxCluster;

//			dist += val;
			dist += sqrt((float) val);
//			dist += val * val;
		}

//		dists.push_back(pair<uint, uint>(dist, i));

		dists.push_back(pair<float, uint>(dist, i));

	}

	std::sort(dists.begin(), dists.end());

//	d_distSeqH = dists;

//	for (int i = 0; i < dists.size(); i++) {
//
//		cout << i << " == " << dists[i].second << " : ";
//		for (int p = 0; p < d_p; p++) {
//			uint val = (dists[i].second / denom[p]) % _maxCluster;
//			cout << val << " ";
//		}
//		cout << endl;
//	}

	uint* distseq = new uint[NUM_DISTSEQ];

	for (int i = 0; i < NUM_DISTSEQ; i++)
		distseq[i] = 0;

	nVec = (nVec < NUM_DISTSEQ) ? nVec : NUM_DISTSEQ;

	d_numDistSeq = nVec;
	d_distCluster = _maxCluster;

	for (int i = 0; i < nVec; i++)
		distseq[i] = dists[i].second;

	cudaMalloc(&d_distSeq, NUM_DISTSEQ * sizeof(float));

	cudaMemcpy(d_distSeq, distseq, NUM_DISTSEQ * sizeof(float),
			cudaMemcpyHostToDevice);

	delete[] distseq;
}

uint ProTree::locateInSequence(uint _code) {

	uint i = 0;
	for (; i < d_distSeqH.size(); i++) {
		if (d_distSeqH[i].second == _code)
			break;
	}

	return i;

}

void ProTree::writeTreeToFile(const std::string& _name) {

	std::ofstream f(_name.c_str(), std::ofstream::out | std::ofstream::binary);

	f << d_dim << endl;
	f << d_p << endl;
	f << d_p2 << endl;
	f << d_nClusters << endl;
	f << d_nClusters2 << endl;

	float * cb1Host = new float[d_nClusters * d_dim];
	float * cb2Host = new float[d_nClusters * d_nClusters2 * d_dim];

	cudaMemcpy(cb1Host, d_codeBook, d_nClusters * d_dim * sizeof(float),
			cudaMemcpyDeviceToHost);
	cudaMemcpy(cb2Host, d_codeBook2,
			d_nClusters * d_nClusters2 * d_dim * sizeof(float),
			cudaMemcpyDeviceToHost);

	char* cc = (char*) cb1Host;
	for (int i = 0; i < 10; i++) {
		cout << int(cc[i]) << " ";
	}
	cout << endl;

	cout << "cb1[12]: " << cb1Host[12] << endl;
	cout << "cb2[12]: " << cb2Host[12] << endl;

	f.write((char*) cb1Host, d_nClusters * d_dim * sizeof(float));
	f.write((char*) cb2Host,
			d_nClusters * d_nClusters2 * d_dim * sizeof(float));

	f.close();

	delete[] cb2Host;
	delete[] cb1Host;

	if (d_sparseBin) {
		uint nBins = pow(d_nClusters, d_p);
		ofstream fs((_name + string("_sparse")).c_str(),
				std::ofstream::out | std::ofstream::binary);

		uint* sparseHost = new uint[nBins];

		cudaMemcpy(sparseHost, d_sparseBin, nBins * sizeof(uint),
				cudaMemcpyDeviceToHost);

		fs.write((char*) sparseHost, nBins * sizeof(uint));

		fs.close();

		delete[] sparseHost;
	}

}

void ProTree::readTreeFromFile(const std::string& _name) {

	ifstream f(_name.c_str(), std::ofstream::in | std::ofstream::binary);

	f >> d_dim;
	f >> d_p;
	f >> d_p2;
	f >> d_nClusters;
	f >> d_nClusters2;

	f.ignore(1);

	cout << d_dim << endl;
	cout << d_p << endl;
	cout << d_p2 << endl;
	cout << d_nClusters << endl;
	cout << d_nClusters2 << endl;

	d_vl = d_dim / d_p;
	d_vl2 = d_dim / d_p2;

	if (d_codeBook)
		cudaFree(d_codeBook);

	if (d_codeBook2)
		cudaFree(d_codeBook2);

	if (d_distSeq)
		cudaFree(d_distSeq);

	float * cb1Host = new float[d_nClusters * d_dim];
	float * cb2Host = new float[d_nClusters * d_nClusters2 * d_dim];

	cudaMalloc(&d_codeBook, d_nClusters * d_dim * sizeof(float));
	cudaMalloc(&d_codeBook2,
			d_nClusters * d_nClusters2 * d_dim * sizeof(float));

	f.read((char*) cb1Host, d_nClusters * d_dim * sizeof(float));
	f.read((char*) cb2Host, d_nClusters * d_nClusters2 * d_dim * sizeof(float));

	char* cc = (char*) cb1Host;
	for (int i = 0; i < 10; i++) {
		cout << int(cc[i]) << " ";
	}
	cout << endl;

	cout << "cb1[12]: " << cb1Host[12] << endl;
	cout << "cb2[12]: " << cb2Host[12] << endl;

	cudaMemcpy(d_codeBook, cb1Host, d_nClusters * d_dim * sizeof(float),
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_codeBook2, cb2Host,
			d_nClusters * d_nClusters2 * d_dim * sizeof(float),
			cudaMemcpyHostToDevice);

	checkCudaErrors(cudaDeviceSynchronize());

	f.close();

	prepareDistSequence(d_nClusters2 * NUM_NEIGHBORS, d_groupParts);

	delete[] cb2Host;
	delete[] cb1Host;

	string sparseName(_name + string("_sparse"));

	struct stat buffer;

	if (stat(sparseName.c_str(), &buffer) == 0) {
		uint nBins = pow(d_nClusters, d_p);
		ifstream fs((_name + string("_sparse")).c_str(),
				std::ifstream::in | std::ifstream::binary);

		uint* sparseHost = new uint[nBins];

		fs.read((char*) sparseHost, nBins * sizeof(uint));

		cudaMalloc(&d_sparseBin, nBins * sizeof(uint));
		cudaMemcpy(d_sparseBin, sparseHost, nBins * sizeof(uint),
				cudaMemcpyHostToDevice);

		fs.close();

		delete[] sparseHost;
	}

}

__global__ void groupVectorSegementsKernel(uint* _count, float* _seg, uint _dim,
		uint _vl, const float* _A, uint _N, uint _np, uint _p, uint _c,
		uint* _assign) {

	__shared__ uint pos;

	for (int iter = blockIdx.x; iter < _N; iter += gridDim.x) {
		__syncthreads();

		uint bid = _assign[iter * _np + _p];

		if (bid == _c) {

			if (threadIdx.x == 0) {
				pos = atomicInc(_count, _N);
				//printf( "%d-%d  ", bid, pos );
			}
			__syncthreads();

			_seg[pos * _vl + threadIdx.x] = _A[iter * _dim + _p * _vl
					+ threadIdx.x];
		}
	}
}

uint ProTree::groupVectorSegments(float* _seg, const float* _A, uint _N,
		uint _p, uint _c, uint* _assign) {

	uint count;
	dim3 block(d_vl, 1, 1);

	uint nblocks = idiv(_N, 8);
	dim3 grid((nblocks > 1024) ? 1024 : nblocks, 1, 1);

	cudaMemset(d_count, 0, sizeof(uint));

	groupVectorSegementsKernel<<<block, grid>>>(d_count, _seg, d_dim, d_vl, _A,
			_N, d_p, _p, _c, _assign);

	checkCudaErrors(cudaDeviceSynchronize());
	cudaMemcpy(&count, d_count, sizeof(uint), cudaMemcpyDeviceToHost);

	cout << "count: " << count << endl;
	return count;
}

__global__ void groupVectorSegementsSparseKernel(uint* _count, float* _seg,
		uint _dim, uint _vl, const float* _A, uint _N, uint _np, uint _p,
		uint _c, uint* _assign, const uint*_sparseVec, bool _sparse) {

	__shared__ uint pos;

	for (int iter = blockIdx.x; iter < _N; iter += gridDim.x) {
		__syncthreads();

		uint bid = _assign[iter * _np + _p];

		if ((bid == _c) && (_sparseVec[iter] == _sparse)) {

			if (threadIdx.x == 0) {
				pos = atomicInc(_count, _N);
				//printf( "%d-%d  ", bid, pos );
			}
			__syncthreads();

			_seg[pos * _vl + threadIdx.x] = _A[iter * _dim + _p * _vl
					+ threadIdx.x];
		}
	}
}

uint ProTree::groupVectorSegmentsSparse(float* _seg, const float* _A, uint _N,
		uint _p, uint _c, uint* _assign, const uint *_sparseVec, bool _sparse) {

	uint count;
	dim3 block(d_vl, 1, 1);

	uint nblocks = idiv(_N, 8);
	dim3 grid((nblocks > 1024) ? 1024 : nblocks, 1, 1);

	cudaMemset(d_count, 0, sizeof(uint));

	groupVectorSegementsSparseKernel<<<block, grid>>>(d_count, _seg, d_dim,
			d_vl, _A, _N, d_p, _p, _c, _assign, _sparseVec, _sparse);

	checkCudaErrors(cudaDeviceSynchronize());
	cudaMemcpy(&count, d_count, sizeof(uint), cudaMemcpyDeviceToHost);

	cout << "count: " << count << endl;
	return count;
}

void ProTree::createTree(uint _nClusters1, uint _nClusters2, const float* _A,
		uint _N) {

// compute the first level product quantization
	createCodeBook(_nClusters1, _A, _N);

	if (d_codeBook2)
		cudaFree(d_codeBook2);

	cudaMalloc(&d_codeBook2,
			d_p * _nClusters1 * _nClusters2 * d_vl * sizeof(float));

// prepare product quantization of the second level
	uint* assignd;

	cudaMalloc(&assignd, _N * d_p * sizeof(uint));

	getAssignment(assignd, d_codeBook, _A, d_nClusters, _N);

	float* segments;

	cudaMalloc(&segments, _N * d_vl * sizeof(float));

// each bin will get its one codebook
	ProQuantization pq(d_vl, 1);

// loop over all bins (segments times clusters)
	for (int p = 0; p < d_p; p++) {
		for (int c = 0; c < d_nClusters; c++) {

			// extract those segments that are sorted into this bin
			uint count = groupVectorSegments(segments, _A, _N, p, c, assignd);

			pq.createCodeBook(_nClusters2, segments, count);

			// copy the created codebook
			cudaMemcpy(
					d_codeBook2
							+ getCBIdx(p, c, d_nClusters, d_vl, _nClusters2),
					pq.getCodeBook(), d_vl * _nClusters2 * sizeof(float),
					cudaMemcpyDeviceToDevice);

			//(p * d_nClusters + c) * d_vl * _k2, pq.getCodeBook(), d_vl * _k2, cudaMemcpyDeviceToDevice);

		}
	}

	d_nClusters2 = _nClusters2;

	prepareDistSequence(d_nClusters2 * NUM_NEIGHBORS, d_groupParts);

	cudaFree(segments);
	cudaFree(assignd);
}

__global__ void calcL1HistogramKernel(uint* _sparseBin, uint _N, uint _np,
		uint _nClusters, const uint* _assign) {

	for (int iter = blockIdx.x * blockDim.x + threadIdx.x; iter < _N;
			iter += gridDim.x * blockDim.x) {
		__syncthreads();

		uint binIdx;
		binIdx = _assign[iter * _np + 0];
		binIdx = _assign[iter * _np + 1] + binIdx * _nClusters;
		binIdx = _assign[iter * _np + 2] + binIdx * _nClusters;
		binIdx = _assign[iter * _np + 3] + binIdx * _nClusters;

		atomicInc(_sparseBin + binIdx, _N);
	}
}

void ProTree::calcL1Histogram(uint* _sparseBin, const uint* _assignd, uint _N,
		float _percent) {

	uint nThreads = 1024;
	dim3 block(nThreads, 1, 1);

	uint nblocks = idiv(_N, nThreads);
	dim3 grid((nblocks > 1024) ? 1024 : nblocks, 1, 1);

	uint nBins = pow(d_nClusters, d_p);
	cudaMemset(_sparseBin, 0, nBins * sizeof(uint));

	// compute the histogram of L1 bins
	calcL1HistogramKernel<<<block, grid>>>(_sparseBin, _N, d_p, d_nClusters,
			_assignd);

	checkCudaErrors(cudaDeviceSynchronize());

	outputVecUint("L1Hist: ", _sparseBin, nBins);

	uint* hist = new uint[nBins];
	cudaMemcpy(hist, _sparseBin, nBins * sizeof(uint), cudaMemcpyDeviceToHost);

	/////////////////////////////////////////
	// mark the bins that hold the densest _percent samples
	uint total = 0;
	for (int i = 0; i < nBins; i++)
		total += hist[i];

	vector<pair<uint, uint> > shist;
	shist.resize(nBins);

	for (int i = 0; i < nBins; i++)
		shist[i] = pair<uint, uint>(hist[i], i);

	// sort the bins ascending
	std::sort(shist.begin(), shist.end());

	// mark all bins sparse enough as valid
	uint thresh = (1. - _percent) * total;
	uint accu = 0;
	for (int i = 0; i < nBins; i++) {
		if (accu < thresh) {
			hist[shist[i].second] = true;
		} else
			hist[shist[i].second] = false;

		accu += shist[i].first;

		if ((i % 100) == 0)
			cout << accu << "  " << shist[i].first << "   "
					<< hist[shist[i].second] << endl;

	}

	cudaMemcpy(_sparseBin, hist, nBins * sizeof(uint), cudaMemcpyHostToDevice);

	delete[] hist;
}

__global__ void markSparseVecKernel(uint* _sparseVec, const uint* _sparseBin,
		uint _N, uint _np, uint _nClusters, const uint* _assign, bool _sparse) {

	for (int iter = blockIdx.x * blockDim.x + threadIdx.x; iter < _N;
			iter += gridDim.x * blockDim.x) {
		__syncthreads();

		uint binIdx;
		binIdx = _assign[iter * _np + 0];
		binIdx = _assign[iter * _np + 1] + binIdx * _nClusters;
		binIdx = _assign[iter * _np + 2] + binIdx * _nClusters;
		binIdx = _assign[iter * _np + 3] + binIdx * _nClusters;

		_sparseVec[iter] = (_sparseBin[binIdx] == _sparse);
	}
}

void ProTree::markSparseVectors(uint* _sparseVec, const uint* _sparseBin,
		const uint* _assignd, uint _N, bool _sparse) const {

	uint nThreads = 1024;
	dim3 block(nThreads, 1, 1);

	uint nblocks = idiv(_N, nThreads);
	dim3 grid((nblocks > 1024) ? 1024 : nblocks, 1, 1);

	// mark all bins that fall into a sparse bin
	markSparseVecKernel<<<block, grid>>>(_sparseVec, _sparseBin, _N, d_p,
			d_nClusters, _assignd, _sparse);

	checkCudaErrors(cudaDeviceSynchronize());

	uint* sparseHost = new uint[_N];

	cudaMemcpy(sparseHost, _sparseVec, _N * sizeof(uint),
			cudaMemcpyDeviceToHost);

	uint nActive = 0;
	for (int i = 0; i < _N; i++) {
		if (sparseHost[i])
			nActive++;
	}

	cout << "active Samples: " << nActive << " out of " << _N << endl;

	delete[] sparseHost;
}

void ProTree::createTreeSplitSparse(uint _nClusters1, uint _nClusters2,
		const float* _A, uint _N, float _percent, bool _sparse) {

// compute the first level product quantization
	if (!d_codeBook)
		// keep the codebook of the other iteration if available
		createCodeBook(_nClusters1, _A, _N);

	if (d_codeBook2)
		cudaFree(d_codeBook2);

	cudaMalloc(&d_codeBook2,
			d_p * _nClusters1 * _nClusters2 * d_vl * sizeof(float));

// prepare product quantization of the second level
	uint* assignd;

	cudaMalloc(&assignd, _N * d_p * sizeof(uint));

	uint* sparseSample;

	cudaMalloc(&sparseSample, _N * sizeof(uint));
	cudaMalloc(&d_sparseBin, pow(_nClusters1, d_p) * sizeof(uint));

	getAssignment(assignd, d_codeBook, _A, d_nClusters, _N);

	// estimate which bins are highly occupied.

	calcL1Histogram(d_sparseBin, assignd, _N, 0.3);

	markSparseVectors(sparseSample, d_sparseBin, assignd, _N, _sparse);

	float* segments;

	cudaMalloc(&segments, _N * d_vl * sizeof(float));

// each bin will get its one codebook
	ProQuantization pq(d_vl, 1);

// loop over all bins (segments times clusters)
	for (int p = 0; p < d_p; p++) {
		for (int c = 0; c < d_nClusters; c++) {

			// extract those segments that are sorted into this bin
			uint count = groupVectorSegmentsSparse(segments, _A, _N, p, c,
					assignd, sparseSample, _sparse);

			pq.createCodeBook(_nClusters2, segments, count);

			// copy the created codebook
			cudaMemcpy(
					d_codeBook2
							+ getCBIdx(p, c, d_nClusters, d_vl, _nClusters2),
					pq.getCodeBook(), d_vl * _nClusters2 * sizeof(float),
					cudaMemcpyDeviceToDevice);

			//(p * d_nClusters + c) * d_vl * _k2, pq.getCodeBook(), d_vl * _k2, cudaMemcpyDeviceToDevice);

		}
	}

	d_nClusters2 = _nClusters2;

	prepareDistSequence(d_nClusters2 * NUM_NEIGHBORS, d_groupParts);

	cudaFree(segments);
	cudaFree(assignd);

	cudaFree(sparseSample);
}

// each block is responsible for one vector, blockDim.x should be _dim
// requires _dim * float shared memory
// looping multiple times to process all B vectors
// _vl is the length of the _p vector segments (should be 2^n)
__global__ void assignClusterKernel2(uint *_assign, const float* _A,
		const float* _B, uint _Arows, uint _Brows, uint _dim, uint _p, uint _vl,
		const uint* _assign1, uint _nClusters1) {

	extern __shared__ float shm[];

	uint* code1 = (uint*) shm + _dim;

	float minVal;
	uint minIdx;

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		__syncthreads();

		if (threadIdx.x < _p) {
			minVal = 1000000.;
			minIdx = 0;

			code1[threadIdx.x] = _assign1[iter * _p + threadIdx.x];
		}
		__syncthreads();

		// each segment needs a different codebook
		uint p = threadIdx.x / _vl;
		const float* cb = _A + getCBIdx(p, code1[p], _nClusters1, _vl, _Arows)
				+ (threadIdx.x % _vl);

		// load vector
		float b = _B[iter * _dim + threadIdx.x];

		// loop over all vectors of A
		for (int a = 0; a < _Arows; a++) {

			shm[threadIdx.x] = sqr(b - cb[a * _vl]);

//			if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
//				printf("aaa %f ", cb[a*_vl]);
//			}

// compute the sum of differences for the vector segments
// i.e. all _p segments in parallel
			for (uint stride = _vl >> 1; stride > 0; stride >>= 1) {
				__syncthreads();
//				if ((threadIdx.x % _vl) < stride) {
//					shm[threadIdx.x] += shm[threadIdx.x + stride];
//				}
				if ((threadIdx.x) < _p * stride) {
					uint p = threadIdx.x / stride * _vl;
					uint bias = threadIdx.x % stride;
					shm[p + bias] += shm[p + bias + stride];
				}
			}
			__syncthreads();

			if (threadIdx.x < _p) {

				// select the minimum of each segment
				float val = shm[threadIdx.x * _vl];
				if (val < minVal) {
					minVal = val;
					minIdx = a;
				}
			}

			__syncthreads();
		}

		// write out decision
		if (threadIdx.x < _p) {
			_assign[iter * _p + threadIdx.x] = minIdx;
		}

	} // iter
}

void ProTree::getAssignment2(uint *_assign2, const float* _A, const float* _B,
		uint _Arows, uint _Brows, const uint *_assign1,
		uint _nClusters1) const {

	dim3 block(d_dim, 1, 1);
	dim3 grid((_Brows < 1024) ? _Brows : 1024, 1, 1);

	uint shm = (d_dim + d_p) * sizeof(float);
	assignClusterKernel2<<<grid, block, shm>>>(_assign2, _A, _B, _Arows, _Brows,
			d_dim, d_p, d_vl, _assign1, _nClusters1);

	checkCudaErrors(cudaDeviceSynchronize());
}

#if 0
__device__ void calcIdx(volatile uint* _shm, volatile uint* _factor,
		const uint* _assign, const uint* _assign2, uint _p, uint _nClusters,
		uint _nClusters2, uint _iter) {

	// load assignment vector into shm;
	if (threadIdx.x < _p) {
		uint offs = _iter * _p + threadIdx.x;
		_shm[threadIdx.x] = _assign[offs];
		_factor[threadIdx.x] = _nClusters;
	} else {
		uint offs = _iter * _p + threadIdx.x - _p;
		_shm[threadIdx.x] = _assign2[offs];
		_factor[threadIdx.x] = _nClusters2;
	}

	// calculate index
	for (uint stride = 1; stride <= _p; stride <<= 1) {
		__syncthreads();
		if (threadIdx.x % (2 * stride) == 0) {
			_shm[threadIdx.x] = _shm[threadIdx.x]
			* _factor[threadIdx.x + stride]
			+ _shm[threadIdx.x + stride];
			_factor[threadIdx.x] *= _factor[threadIdx.x + stride];
		}
	}

	__syncthreads();
}
#endif

__device__ void calcIdx(volatile uint* _shm, volatile uint* _factor,
		const uint* _assign, const uint* _assign2, uint _p, uint _nClusters,
		uint _nClusters2, uint _iter) {

	// load assignment vector into shm;
	if (threadIdx.x < _p) {
		uint offs = _iter * _p + threadIdx.x;
		_shm[threadIdx.x] = _assign[offs] * _nClusters2 + _assign2[offs];
	}

	// assume implicit synchronization as num threads is smaller than
	if (threadIdx.x == 0) {
		for (int p = 1; p < _p; p++)
			_shm[0] = _shm[0] * _nClusters * _nClusters2 + _shm[p];

//		if (_iter < 20 )
//			printf("%d, %d - %d, %d = %d\n",_assign[_iter * _p],_assign2[_iter * _p],_assign[_iter * _p+1],_assign2[_iter * _p+1],  _shm[0] );
//
//		_shm[0] = _nClusters + 2;
//
//		if (_assign[_iter * _p +1] == 2)
//			_shm[0] = _assign[_iter * _p];
	}

	__syncthreads();
}

#if 0
// TODO _p1 + _p2
__device__ uint calcIdxSequential(uint *_idx, uint _p, uint _nClusters,
		uint _nClusters2, uint _c1scale) {

	uint idx = 0;

	for (int k = 0; k < _p; k++) {
		idx = (idx * _nClusters) + _idx[k] / _c1scale;
	}

	for (int k = 0; k < _p; k++) {
		idx = (idx * _nClusters2) + _idx[k] % _c1scale;
	}
	s
	return idx;

}
#endif

#if 0
__device__ void calcIdx(volatile uint* _shm, volatile uint* _factor,
		const uint* _assign, const uint* _idx, uint _nIdx, uint _p,
		uint _nClusters, uint _nClusters2, uint _c1scale) {

	if (threadIdx.x == 0) {
//		printf("calcIdx ");
//		for (int p = 0; p < _p; p++)
//			printf("%d ", _assign[p]);
//		printf("\n");
		_shm[0] = 0;
	}
	__syncthreads();
//	return;

	// load assignment vector into shm;
	if (threadIdx.x < _p) {
		uint offs = threadIdx.x;
		_shm[threadIdx.x] = _idx[_assign[offs] + offs * _nIdx] / _c1scale;
		_factor[threadIdx.x] = _nClusters;
	} else if (threadIdx.x < 2 * _p) {
		uint offs = threadIdx.x - _p;
		_shm[threadIdx.x] = _idx[_assign[offs] + offs * _nIdx] % _c1scale;
		_factor[threadIdx.x] = _nClusters2;
	}

	// calculate index
	for (uint stride = 1; stride <= _p; stride <<= 1) {
		__syncthreads();
		if (threadIdx.x % (2 * stride) == 0) {
			_shm[threadIdx.x] = _shm[threadIdx.x]
			* _factor[threadIdx.x + stride]
			+ _shm[threadIdx.x + stride];
			_factor[threadIdx.x] *= _factor[threadIdx.x + stride];
		}
	}

	__syncthreads();

}
#endif

__global__ void countBinsKernel(uint* _bins, const uint* _assign,
		uint* _assign2, uint _N, uint _p, uint _nClusters, uint _nClusters2) {

	extern __shared__ float shmf[];

	uint* shm = (uint*) shmf;

	uint *factor = shm + 2 * _p;

	for (int iter = blockIdx.x; iter < _N; iter += gridDim.x) {
		__syncthreads();
#if 0
		// load assignment vector into shm;
		if (threadIdx.x < _p) {
			uint offs = iter * _p + threadIdx.x;
			shm[threadIdx.x] = _assign[offs];
			factor[threadIdx.x] = _nClusters;
		} else {
			uint offs = iter * _p + threadIdx.x - _p;
			shm[threadIdx.x] = _assign2[offs];
			factor[threadIdx.x] = _nClusters2;
		}

		// calculate index
		for (uint stride = 1; stride <= _p; stride <<= 1) {
			__syncthreads();
			if (threadIdx.x % (2 * stride) == 0) {
				shm[threadIdx.x] = shm[threadIdx.x]
				* factor[threadIdx.x + stride]
				+ shm[threadIdx.x + stride];
				factor[threadIdx.x] *= factor[threadIdx.x + stride];
			}
		}

		__syncthreads();
#endif
		calcIdx(shm, factor, _assign, _assign2, _p, _nClusters, _nClusters2,
				iter);

		if (threadIdx.x == 0) {

//			if (iter < 20)
//				printf( "%d .. %d\n", shm[0], _N);
			atomicInc(_bins + shm[0], _N);

//			_assign2[iter * _p] = shm[0];
		}
	}
}

void ProTree::countBins(uint* _bins, const uint* _assign, uint* _assign2,
		uint _N) {

	dim3 block(d_p + d_p, 1, 1);

	dim3 grid((_N < 1024) ? _N : 1024, 1, 1);

	uint shmsize = (d_p + d_p) * sizeof(uint);

	cudaMemset(d_binCounts, 0, d_nBins * sizeof(uint));

	countBinsKernel<<<grid, block, shmsize>>>(_bins, _assign, _assign2, _N, d_p,
			d_nClusters, d_nClusters2);

	checkCudaErrors(cudaDeviceSynchronize());
}

__device__ void calcMultiIdx(uint& _idx, uint *_shm, const uint* _assign,
		const uint* _assign2, uint _p, uint _nClusters, uint _nClusters2,
		uint _iter, uint _groupParts, uint _nDBs) {

	// losd the assignments into shared memory
	if (threadIdx.x < _p) {

		uint offs = _iter * _p + threadIdx.x;
		_shm[threadIdx.x] = _assign[offs];
		_shm[threadIdx.x + _p] = _assign2[offs];
	}
	__syncthreads();

	if (threadIdx.x < _nDBs) {
		_idx = 0;

		for (int i = 0; i < _groupParts; i++) {
			uint offs = threadIdx.x * _groupParts + i;
			_idx = _shm[offs] + _nClusters * _idx;
		}

		for (int i = 0; i < _groupParts; i++) {
			uint offs = threadIdx.x * _groupParts + i + _p;
			_idx = _shm[offs] + _nClusters2 * _idx;
		}
	}

	__syncthreads();
}

__global__ void countMultiBinsKernel(uint* _bins, const uint* _assign,
		const uint* _assign2, uint _N, uint _p, uint _nClusters,
		uint _nClusters2, uint _groupParts, uint _nBins, uint _nDBs) {

	extern __shared__ float shmf[];

	uint* shm = (uint*) shmf;

	uint idx;

	for (int iter = blockIdx.x; iter < _N; iter += gridDim.x) {

		__syncthreads();

		calcMultiIdx(idx, shm, _assign, _assign2, _p, _nClusters, _nClusters2,
				iter, _groupParts, _nDBs);

		if (threadIdx.x < _nDBs) {
			atomicInc(_bins + threadIdx.x * _nBins + idx, _N);
		}
	}
}

void ProTree::countMultiBins(uint* _bins, const uint* _assign,
		const uint* _assign2, uint _N, uint _groupParts, uint _nDBs) {

	dim3 block(d_p + d_p, 1, 1);

	dim3 grid((_N < 1024) ? _N : 1024, 1, 1);

	uint shmsize = (d_p + d_p) * sizeof(uint);

	cudaMemset(d_binCounts, 0, _nDBs * d_nBins * sizeof(uint));

	countMultiBinsKernel<<<grid, block, shmsize>>>(_bins, _assign, _assign2, _N,
			d_p, d_nClusters, d_nClusters2, _groupParts, d_nBins, _nDBs);

	checkCudaErrors(cudaDeviceSynchronize());
}

template<typename scalar>
__device__ scalar scan_warp(volatile scalar * ptr, bool _inclusive,
		const uint idx = threadIdx.x) {
	const uint lane = idx & 31;

	if (lane >= 1)
		ptr[idx] = ptr[idx - 1] + ptr[idx];
	if (lane >= 2)
		ptr[idx] = ptr[idx - 2] + ptr[idx];
	if (lane >= 4)
		ptr[idx] = ptr[idx - 4] + ptr[idx];
	if (lane >= 8)
		ptr[idx] = ptr[idx - 8] + ptr[idx];
	if (lane >= 16)
		ptr[idx] = ptr[idx - 16] + ptr[idx];

	if (_inclusive)
		return ptr[idx];
	else
		return (lane > 0) ? ptr[idx - 1] : 0.;
}

__device__ float scan_block(volatile uint *ptr, bool _inclusive,
		const uint idx = threadIdx.x) {
	const uint lane = idx & 31;
	const uint warpid = idx >> 5;

	// Step 1: Intra-warp scan in each warp
	float val = scan_warp(ptr, _inclusive, idx);
	__syncthreads();

	// Step 2: Collect per-warp partial results
	if (lane == 31)
		ptr[warpid] = ptr[idx];
	__syncthreads();

	// Step 3: Use 1st warp to scan per-warp results
	if (warpid == 0)
		scan_warp(ptr, true, idx);
	__syncthreads();

	// Step 4: Accumulate results from Steps 1 and 3
	if (warpid > 0)
		val = ptr[warpid - 1] + val;
	__syncthreads();

	// Step 5: Write and return the final result
	ptr[idx] = val;
	__syncthreads();

	return val;
}

__global__ void scan_block(uint *_ofield, uint *_block_res, const uint *_field,
		uint _n, uint _nBlocks, bool _inclusive) {

	extern __shared__ float shmf[];

	uint* shm = (uint*) shmf;

	uint idx = threadIdx.x;
	uint bid = blockIdx.y * gridDim.x + blockIdx.x;
	uint pos = idx + bid * blockDim.x;

	// fill shm;
	shm[idx] = (pos < _n) ? _field[pos] : 0.;
	__syncthreads();

	float val = scan_block(shm, _inclusive, idx);
	__syncthreads();
	if (pos < _n)
		_ofield[pos] = val;

	if (_block_res != NULL) {
		if (threadIdx.x == blockDim.x - 1) {
//			_block_res[blockIdx.x] = shm[blockDim.x - 1];
			if (bid < _nBlocks)
				_block_res[bid] = shm[blockDim.x - 1];
			//printf("blockres: %i %i \n", blockIdx.x, _block_res[blockIdx.x]);
		}
	}

}

__global__ void addBlock(uint *_ofield, uint *_block_res, uint _n,
		bool _inclusive) {
	uint idx = threadIdx.x;
	uint pos = idx + blockIdx.x * blockDim.x;

//	if (idx == 0) {
//		printf("block: %i, %f \n", blockIdx.x, _block_res[blockIdx.x]);
//	}

	if (pos < _n) {
		if (blockIdx.x > 0) {

			if (_inclusive)
				_ofield[pos] += _block_res[blockIdx.x - 1];
			else
				_ofield[pos] = ((idx == 0) ? 0. : _ofield[pos - 1])
						+ _block_res[blockIdx.x - 1];
		} else {
			if (!_inclusive)
				_ofield[pos] = (idx == 0) ? 0. : _ofield[pos - 1];

		}
	}

}

__global__ void addBlockInclusive(uint *_ofield, uint *_block_res, uint _n) {
	uint idx = threadIdx.x;
	uint bid = blockIdx.y * gridDim.x + blockIdx.x;
	uint pos = idx + bid * blockDim.x;

	if (bid > 0) {
		if (pos < _n) {
			_ofield[pos] += _block_res[bid - 1];
		}
	}

}

__global__ void addBlockExclusive(uint *_ofield, uint *_ifield,
		uint *_block_res, uint _n) {
	uint idx = threadIdx.x;
	uint pos = idx + blockIdx.x * blockDim.x;

//	if (idx == 0) {
//		printf("block: %i, %f \n", blockIdx.x, _block_res[blockIdx.x]);
//	}

	if (pos < _n) {
		if (blockIdx.x > 0) {
			_ofield[pos] = ((idx == 0) ? 0. : _ifield[pos - 1])
					+ _block_res[blockIdx.x - 1];
		} else {
			_ofield[pos] = (idx == 0) ? 0. : _ifield[pos - 1];

		}
	}

}

__global__ void saveBlockBoundaries(uint *_boundaries, uint *_field, uint _n) {

	uint idx = threadIdx.x;
	uint pos = idx + blockIdx.x * blockDim.x;

	// save allways the last item of each block as this will be needed in the next block
	if (pos < _n) {
		if (threadIdx.x == blockDim.x - 1)
			_boundaries[blockIdx.x] = _field[pos];
	}

}

__global__ void addBlockExclusiveBoundaries(uint *_field, uint *_block_res,
		uint _n) {

	extern __shared__ float shmf[];

	uint* shm = (uint*) shmf;

	uint idx = threadIdx.x;
	uint bid = (blockIdx.y * gridDim.x + blockIdx.x);
	uint pos = idx + bid * blockDim.x;

	// save entire block to shm
	if (pos < _n) {
		shm[threadIdx.x] = _field[pos];
	}

	__syncthreads();

	if (pos < _n) {
		if (bid > 0) {
			if (threadIdx.x == 0) {
				// take the first entry from the previous block, which is saved in boundaries
				_field[pos] = _block_res[bid - 1];
			} else
				_field[pos] = shm[threadIdx.x - 1] + _block_res[bid - 1];
		} else {
			_field[pos] = (idx == 0) ? 0. : shm[pos - 1];

		}
	}

}

void ProTree::scan(uint *_prefixSum, const uint *_ptr, uint _n,
		bool _inclusive) {
	dim3 grid, blocks;

	uint nWarps = idiv(_n, 32);
	uint nBlocks = idiv(_n, MAX_THREADS);

	cout << "nBlocks: " << nBlocks << endl;

	if (_n <= MAX_THREADS) {
		grid = dim3(1, 1, 1);
		blocks = dim3(nWarps * 32, 1, 1);
		scan_block<<<grid, blocks, nWarps * 32 * sizeof(uint)>>>(_prefixSum,
		NULL, _ptr, _n, 10, _inclusive);
		checkCudaErrors(cudaDeviceSynchronize());
	} else {
		uint *blockRes;
		uint *blockRes2;
		cudaMalloc((void**) &blockRes, nBlocks * sizeof(uint));
		cudaMalloc((void**) &blockRes2, nBlocks * sizeof(uint));

		grid = dim3(nBlocks, 1, 1);
		if (nBlocks > MAX_BLOCKS) {
			grid = dim3(MAX_BLOCKS, idiv(nBlocks, MAX_BLOCKS), 1);
		}
		blocks = dim3(MAX_THREADS, 1, 1);

		cout << "grid: " << grid.x << " " << grid.y << " " << grid.z << endl;

		scan_block<<<grid, blocks, 2 * MAX_THREADS * sizeof(uint)>>>(_prefixSum,
				blockRes2, _ptr, _n, nBlocks, true);
		checkCudaErrors(cudaDeviceSynchronize());

		scan(blockRes, blockRes2, nBlocks, true);

		if (_inclusive) {

			addBlockInclusive<<<grid, blocks>>>(_prefixSum, blockRes, _n);

		} else {

			addBlockExclusiveBoundaries<<<grid, blocks,
			MAX_THREADS * sizeof(uint)>>>(_prefixSum, blockRes, _n);
		}

		checkCudaErrors(cudaDeviceSynchronize());
		cudaFree(blockRes2);
		cudaFree(blockRes);
	}
}

__global__ void sortIdxKernel(uint* _dbIdx, uint* _binCount,
		const uint* _prefix, const uint* _assign, const uint* _assign2, uint _N,
		uint _p, uint _nClusters, uint _nClusters2) {

	extern __shared__ float shmf[];

	uint* shm = (uint*) shmf;

	uint *factor = shm + 2 * _p;

	for (int iter = blockIdx.x; iter < _N; iter += gridDim.x) {

		__syncthreads();

		calcIdx(shm, factor, _assign, _assign2, _p, _nClusters, _nClusters2,
				iter);

		if (threadIdx.x == 0) {
			uint pos = atomicInc(_binCount + shm[0], _N);
			if ((_prefix[shm[0]] + pos) > _N) {
				printf("out of range!: %d, %d, %d \n", _prefix[shm[0]], pos,
						shm[0]);
			}
//			if //(iter == 932085)
//			(((_prefix[shm[0]] + pos) >= 789966)
//					&& ((_prefix[shm[0]] + pos) < 789975)) {
//				printf("found %d, %d, %d, %d \n", iter, _prefix[shm[0]], pos,
//						shm[0]);
//			}
			_dbIdx[_prefix[shm[0]] + pos] = iter;

		}
	}
}

void ProTree::sortIdx(uint* _dbIdx, const uint* _assign, const uint* _assign2,
		uint _N) {

	dim3 block(d_p + d_p, 1, 1);

	dim3 grid((_N < 1024) ? _N : 1024, 1, 1);

	uint shmsize = (d_p + d_p) * sizeof(uint);

	cudaMemset(d_binCounts, 0, d_nBins * sizeof(uint));
	sortIdxKernel<<<grid, block, shmsize>>>(_dbIdx, d_binCounts, d_binPrefix,
			_assign, _assign2, _N, d_p, d_nClusters, d_nClusters2);

	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void sortMultiIdxKernel(uint* _dbIdx, uint* _binCount,
		const uint* _prefix, const uint* _assign, const uint* _assign2, uint _N,
		uint _p, uint _nClusters, uint _nClusters2, uint _groupParts,
		uint _nBins, uint _nDBs) {

	extern __shared__ float shmf[];

	uint* shm = (uint*) shmf;

	uint idx;

	for (int iter = blockIdx.x; iter < _N; iter += gridDim.x) {

		__syncthreads();
		calcMultiIdx(idx, shm, _assign, _assign2, _p, _nClusters, _nClusters2,
				iter, _groupParts, _nDBs);

		if (threadIdx.x < _nDBs) {
			uint pos = atomicInc(_binCount + threadIdx.x * _nBins + idx, _N);

			if (iter == 0) {
				printf("final bins: db: %d : %d \n", threadIdx.x,
						threadIdx.x * _nBins + idx);
				printf("%d prefix: %d, pos: %d \n", threadIdx.x,
						*(_prefix + threadIdx.x * _nBins + idx), pos);
			}

			_dbIdx[*(_prefix + threadIdx.x * _nBins + idx) + pos] = iter;
		}
	}
}

void ProTree::sortMultiIdx(uint* _dbIdx, const uint* _assign,
		const uint* _assign2, uint _N, uint _groupParts, uint _nDBs) {

	dim3 block(d_p + d_p, 1, 1);

	dim3 grid((_N < 1024) ? _N : 1024, 1, 1);

	uint shmsize = (d_p + d_p) * sizeof(uint);

	cudaMemset(d_binCounts, 0, _nDBs * d_nBins * sizeof(uint));
	sortMultiIdxKernel<<<grid, block, shmsize>>>(_dbIdx, d_binCounts,
			d_binPrefix, _assign, _assign2, _N, d_p, d_nClusters, d_nClusters2,
			_groupParts, d_nBins, _nDBs);

	checkCudaErrors(cudaDeviceSynchronize());
}

void ProTree::histogram(uint _nBins) const {
	uint* binCounts = new uint[_nBins];

	cudaMemcpy(binCounts, d_binCounts, _nBins * sizeof(uint),
			cudaMemcpyDeviceToHost);

	vector<uint> binHist;
	binHist.resize(8);
	uint total = 0;

	uint maxVal = 0;
	uint maxIdx = 0;
	for (int i = 0; i < _nBins; i++) {

		if (binCounts[i] > maxVal) {
			maxVal = binCounts[i];
			maxIdx = i;
		}

		if (binCounts[i] == 0)
			binHist[0]++;
		else if (binCounts[i] < 5)
			binHist[1]++;
		else if (binCounts[i] < 10)
			binHist[2]++;
		else if (binCounts[i] < 20)
			binHist[3]++;
		else if (binCounts[i] < 50)
			binHist[4]++;
		else if (binCounts[i] < 100)
			binHist[5]++;
		else if (binCounts[i] < 500)
			binHist[6]++;
		else
			binHist[7]++;

		total += binCounts[i];

	}

	cout << "total entries: " << total << endl;

	cout << "histogram: " << endl;
	cout << "0 \t" << binHist[0] << endl;
	cout << "<5 \t" << binHist[1] << endl;
	cout << "<10 \t" << binHist[2] << endl;
	cout << "<20 \t" << binHist[3] << endl;
	cout << "<50\t" << binHist[4] << endl;
	cout << "<100 \t" << binHist[5] << endl;
	cout << "<500 \t" << binHist[6] << endl;
	cout << ">500 \t" << binHist[7] << endl;

	cout << "maxbin: " << maxIdx << "  entries: " << maxVal << endl;

	delete[] binCounts;
}

void ProTree::buildDB(const float* _A, uint _N) {

	cout << "build data base" << endl;

	uint* assignd = NULL;
	uint* assignd2 = NULL;

	cudaMalloc(&assignd, _N * d_p * sizeof(uint));
	cudaMalloc(&assignd2, _N * d_p * sizeof(uint));

	getAssignment(assignd, d_codeBook, _A, d_nClusters, _N);

	getAssignment2(assignd2, d_codeBook2, _A, d_nClusters2, _N, assignd,
			d_nClusters);

	outputVecUint("assign", assignd, 256);
	outputVecUint("assign2", assignd2, 256);

	uint* assh = new uint[_N * d_p];

	cudaMemcpy(assh, assignd, _N * d_p * sizeof(uint), cudaMemcpyDeviceToHost);

	uint countOdd = 0;
	for (int i = 0; i < _N * d_p; i++) {
		if (assh[i] % 2 != 0)
			countOdd++;
	}

	cout << "countEven " << (_N * d_p - countOdd) << " countOdd " << countOdd
			<< endl;

	cout << "clusters: " << d_nClusters << "  " << d_nClusters2 << endl;

	d_nBins = pow(d_nClusters, d_p) * pow(d_nClusters2, d_p);
	cout << "number of bins: " << d_nBins << endl;

	cudaMalloc(&d_binPrefix, d_nBins * sizeof(uint));

	cudaMalloc(&d_binCounts, d_nBins * sizeof(uint));

	cudaMalloc(&d_dbIdx, _N * sizeof(uint));

	if ((d_binPrefix == NULL) || (d_binCounts == NULL)) {
		cout << "not enough memory - exiting" << endl;
		exit(1);
	}

	cudaMemset(d_binCounts, 0, d_nBins * sizeof(uint));

	countBins(d_binCounts, assignd, assignd2, _N);

	cudaMemcpy(assh, assignd2, _N * d_p * sizeof(uint), cudaMemcpyDeviceToHost);

	countOdd = 0;
	for (int i = 0; i < _N * d_p; i += 2) {
		if (assh[i] % 2 != 0)
			countOdd++;
	}

	cout << "bin: countEven " << (_N - countOdd) << " countOdd " << countOdd
			<< endl;

//	outputVecUint("binCounts", d_binCounts, 256);

	histogram(d_nBins);

	cudaMemset(d_binPrefix, 0, d_nBins * sizeof(uint));

//	scan(d_binPrefix, d_binCounts, d_nBins, true);
	scan(d_binPrefix, d_binCounts, d_nBins, false);

	uint *prefh = new uint[d_nBins];

	cudaMemcpy(prefh, d_binPrefix, d_nBins * sizeof(uint),
			cudaMemcpyDeviceToHost);

//	uint sum = 0;
//	bool test = true;
//	for (int i = 0; i < d_nBins; i++) {
////		sum += binCounts[i];
//		if (sum != prefh[i]) {
//			test = false;
//			cout << i << ":  " << sum << " vs. " << prefh[i] << endl;
//		}
//		sum += binCounts[i];
//	}
//
//	cout << "test (exclusive) " << ((test) ? "passed" : "not passed!!") << endl;

	delete[] prefh;
//	delete[] binCounts;

//	outputVecUint("prefix", d_binPrefix, 256);

//	uint prf[10000];
//
//	cout << "binCounts " << endl;
//	cudaMemcpy(prf, d_binCounts, 10000 * sizeof(uint), cudaMemcpyDeviceToHost);
//	for (int i = 0; i < 2000; i++)
//		cout << prf[i] << "  ";
//	cout << endl;
//
//	cout << "prefix " << endl;
//	cudaMemcpy(prf, d_binPrefix, 10000 * sizeof(uint), cudaMemcpyDeviceToHost);
//	for (int i = 0; i < 2000; i++)
//		cout << prf[i] << "  ";
//	cout << endl;

	cudaMemset(d_dbIdx, 0, _N * sizeof(uint));

	sortIdx(d_dbIdx, assignd, assignd2, _N);

//	outputVecUint("Assign", assignd, d_p * _N);
//
//	outputVecUint("Assign2", assignd2, d_p * _N);

	d_dbVec = _A;
	d_NdbVec = _N;

	delete[] assh;

	cudaFree(assignd2);
	cudaFree(assignd);

}

void ProTree::buildMultiDB(const float* _A, uint _N) {

	uint* assignd;
	uint* assignd2;

	cudaMalloc(&assignd, _N * d_p * sizeof(uint));
	cudaMalloc(&assignd2, _N * d_p * sizeof(uint));

	getAssignment(assignd, d_codeBook, _A, d_nClusters, _N);

	getAssignment2(assignd2, d_codeBook2, _A, d_nClusters2, _N, assignd,
			d_nClusters);

	cout << "clusters: " << d_nClusters << "  " << d_nClusters2 << endl;

	outputVecUint("assign1: ", assignd, d_p);
	outputVecUint("assign2: ", assignd2, d_p);

	d_nDBs = d_p / d_groupParts;
	cout << "number of data bases " << d_nDBs << endl;

	d_nBins = pow(d_nClusters, d_groupParts) * pow(d_nClusters2, d_groupParts);
	cout << "number of bins: " << d_nBins << endl;

	cudaMalloc(&d_binPrefix, d_nDBs * d_nBins * sizeof(uint));

	cudaMalloc(&d_binCounts, d_nDBs * d_nBins * sizeof(uint));

	cudaMalloc(&d_dbIdx, d_nDBs * _N * sizeof(uint));

	countMultiBins(d_binCounts, assignd, assignd2, _N, d_groupParts, d_nDBs);

	cudaMemset(d_binPrefix, 0, d_nDBs * d_nBins * sizeof(uint));

//	for (int i = 0; i < d_nDBs; i++)
//		scan(d_binPrefix + i * d_nBins, d_binCounts + i * d_nBins, d_nBins,
//				true);

	scan(d_binPrefix, d_binCounts, d_nDBs * d_nBins, true);

#if 0
	uint prf[10000];

	cout << "binCounts " << endl;
	cudaMemcpy(prf, d_binCounts, 10000 * sizeof(uint), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 2000; i++)
	cout << prf[i] << "  ";
	cout << endl;

	cout << "prefix " << endl;
	cudaMemcpy(prf, d_binPrefix, 10000 * sizeof(uint), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 2000; i++)
	cout << prf[i] << "  ";
	cout << endl;

#endif

	sortMultiIdx(d_dbIdx, assignd, assignd2, _N, d_groupParts, d_nDBs);

//	outputVecUint("Assign", assignd, d_p * _N);
//
//	outputVecUint("Assign2", assignd2, d_p * _N);

// store references to original vectors
	d_dbVec = _A;
	d_NdbVec = _N; // store number o orginal vectors

	cudaFree(assignd2);
	cudaFree(assignd);

}

void ProTree::fakeDB(const float* _A, uint _N) {
// store references to original vectors
	d_dbVec = _A;
	d_NdbVec = _N; // store number o orginal vectors
}

template<class T>
__device__ void swap(T& _a, T&_b) {
	T h = _a;
	_a = _b;
	_b = h;
}

// parallel bitonic sort
__device__ void bitonic2(volatile float _val[], volatile uint _idx[], uint _N) {

	for (int k = 2; k <= _N; k <<= 1) {

		// bitonic merge
		for (int j = k / 2; j > 0; j /= 2) {
			int ixj = threadIdx.x ^ j;  // XOR
			if ((ixj > threadIdx.x) && (ixj < _N)) {
				if ((threadIdx.x & k) == 0) // ascending - descending
						{
					if (_val[threadIdx.x] > _val[ixj]) {

						swap(_val[threadIdx.x], _val[ixj]);
						swap(_idx[threadIdx.x], _idx[ixj]);
					}
				} else {
					if (_val[threadIdx.x] < _val[ixj]) {

						swap(_val[threadIdx.x], _val[ixj]);
						swap(_idx[threadIdx.x], _idx[ixj]);
					}

				}
			}
			__syncthreads();
		}
	}
}

// parallel bitonic sort (descending)
__device__ void bitonic2Descending(volatile float _val[], volatile uint _idx[],
		uint _N) {

	for (int k = 2; k <= _N; k <<= 1) {

		// bitonic merge
		for (int j = k / 2; j > 0; j /= 2) {
			int ixj = threadIdx.x ^ j;  // XOR
			if ((ixj > threadIdx.x) && (ixj < _N)) {
				if ((threadIdx.x & k) != 0) // ascending - descending
						{
					if (_val[threadIdx.x] > _val[ixj]) {

						swap(_val[threadIdx.x], _val[ixj]);
						swap(_idx[threadIdx.x], _idx[ixj]);
					}
				} else {
					if (_val[threadIdx.x] < _val[ixj]) {

						swap(_val[threadIdx.x], _val[ixj]);
						swap(_idx[threadIdx.x], _idx[ixj]);
					}

				}
			}
			__syncthreads();
		}
	}
}

__global__ void sortTest(float *_vals, uint* _idx, uint _N, uint _NP2) {

	extern __shared__ float shm[];

	uint* shmIdx = (uint*) (shm + _NP2);

	if (threadIdx.x < _NP2) {
		shm[threadIdx.x] = 100.;
	}

	if (threadIdx.x < _N) {
		shm[threadIdx.x] = _vals[threadIdx.x];
		shmIdx[threadIdx.x] = _idx[threadIdx.x];
	}

	__syncthreads();
	bitonic2(shm, shmIdx, _NP2);

	if (threadIdx.x < _N) {
		_vals[threadIdx.x] = shm[threadIdx.x];
		_idx[threadIdx.x] = shmIdx[threadIdx.x];
	}

}

void ProTree::testScan() {

	cout << "starting testScan " << endl;

	uint N = 128 * 64 * 64 * 64;

	uint* values = new uint[N];
	uint* prefix = new uint[N];
	uint* valued;
	uint* prefixd;

	for (int i = 0; i < N; i++)
//		values[i] = 10 * drand48();
		values[i] = 1; // i % 3;

	cudaMalloc(&valued, N * sizeof(float));
	cudaMalloc(&prefixd, N * sizeof(uint));

	if ((valued == NULL) || (prefixd == NULL)) {
		cout << "did not get memory - exiting" << endl;
		exit(1);
	}

	cudaMemcpy(valued, values, N * sizeof(uint), cudaMemcpyHostToDevice);

	scan(prefixd, valued, N, false);

	cudaMemcpy(prefix, prefixd, N * sizeof(uint), cudaMemcpyDeviceToHost);

	uint sum = 0;
	bool test = true;
	for (int i = 0; i < N; i++) {
		if (sum != prefix[i]) {
			test = false;
			cout << i << "  " << sum << " vs. " << prefix[i] << endl;
		}
		sum += values[i];
	}

	cout << "test (exclusive) " << ((test) ? "passed" : "not passed!!") << endl;

	scan(prefixd, valued, N, true);

	cudaMemcpy(prefix, prefixd, N * sizeof(uint), cudaMemcpyDeviceToHost);

	sum = 0;
	test = true;
	for (int i = 0; i < N; i++) {
		sum += values[i];
		if (sum != prefix[i]) {
			test = false;
			cout << i << "  " << sum << " vs. " << prefix[i] << endl;
		}
	}

	cout << "test (inclusive) " << ((test) ? "passed" : "not passed!!") << endl;

	cudaFree(prefixd);
	cudaFree(valued);
	delete[] prefix;
	delete[] values;
}

void ProTree::testSort() {

	cout << "starting testSort " << endl;

	const int N = 24;

	float val[N];
	uint idx[N];

	for (int i = 0; i < N; i++) {
		val[i] = (uint) (N * drand48());
		idx[i] = i;
	}

	for (int i = 0; i < N; i++) {
		printf("%3.0f", val[i]);
	}
	cout << endl;
	for (int i = 0; i < N; i++) {
		printf("%3i", idx[i]);
	}
	cout << endl;

	float* vald;
	uint* idxd;

	cudaMalloc(&vald, N * sizeof(float));
	cudaMalloc(&idxd, N * sizeof(uint));

	cudaMemcpy(vald, val, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(idxd, idx, N * sizeof(uint), cudaMemcpyHostToDevice);

	uint nthreads = log2(N);
	cout << "nthreads: " << nthreads << endl;
	dim3 block(nthreads, 1, 1);
	dim3 grid(1, 1, 1);

	uint shmsize = nthreads * 2 * sizeof(float);

	sortTest<<<grid, block, shmsize>>>(vald, idxd, N, nthreads);

	checkCudaErrors(cudaDeviceSynchronize());

	cudaMemcpy(val, vald, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(idx, idxd, N * sizeof(uint), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {
		printf("%3.0f", val[i]);
	}
	cout << endl;
	for (int i = 0; i < N; i++) {
		printf("%3i", idx[i]);
	}
	cout << endl;

	cout << "done test sort" << endl;

}

// each block is responsible for one vector, blockDim.x should be _dim
// requires _dim * float shared memory
// looping multiple times to process all B vectors
// _vl is the length of the _p vector segments (should be 2^n)
__global__ void assignKBestClusterKernel2(float *_assignVal, uint* _assignIdx,
		const float* _A, const float* _B, uint _Arows, uint _Brows, uint _dim,
		uint _p, uint _vl, const uint* _assign1, uint _nClusters1, uint _k1,
		uint _NP2, uint _c1scale) {

	extern __shared__ float shm[];

//	uint* code1 = (uint*) shm + _dim;
//
//	float* val = shm + _dim + _p;
//	uint* idx = (uint*) (val + _p * _k1 * _Arows);
//
//	uint* shmIdx = (uint*) (shm + _NP2);

	float* shmIter = shm + blockDim.x;

	uint* shmIdx = (uint*) shmIter;
	shmIter += blockDim.x;

	uint* binL1 = (uint*) shmIter;
	shmIter += _p;

	float* val = shmIter;
	shmIter += _p * _k1 * _Arows;

	uint* idx = (uint*) shmIter;
	shmIter += _p * _k1 * _Arows;

//	float minVal;
//	uint minIdx;

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		__syncthreads();
		// load vector
		float b;
		if (threadIdx.x < _dim)
			b = _B[iter * _dim + threadIdx.x];

		// loop over the best k1 first-level bins
		for (int k = 0; k < _k1; k++) {

			if (threadIdx.x < _p) {
				binL1[threadIdx.x] = _assign1[iter * _k1 * _p + k * _p
						+ threadIdx.x];
			}
			__syncthreads();

			// each segment needs a different codebook
			const float* cb;
			if (threadIdx.x < _dim) {
				uint p = threadIdx.x / _vl;
				cb = _A + getCBIdx(p, binL1[p], _nClusters1, _vl, _Arows)
						+ (threadIdx.x % _vl);
			}
			// loop over all vectors of A
			for (int binL2 = 0; binL2 < _Arows; binL2++) {

				if (threadIdx.x < _dim)
					shm[threadIdx.x] = sqr(b - cb[binL2 * _vl]);

//			if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
//				printf("aaa %f ", cb[a*_vl]);
//			}

// compute the sum of differences for the vector segments
// i.e. all _p segments in parallel
				for (uint stride = _vl >> 1; stride > 0; stride >>= 1) {
					__syncthreads();
//				if ((threadIdx.x % _vl) < stride) {
//					shm[threadIdx.x] += shm[threadIdx.x + stride];
//				}
					if ((threadIdx.x) < _p * stride) {
						uint p = threadIdx.x / stride * _vl;
						uint bias = threadIdx.x % stride;
						shm[p + bias] += shm[p + bias + stride];
					}
				}
				__syncthreads();

				// store the result
				if (threadIdx.x < _p) {

					val[binL2 + k * _Arows + threadIdx.x * _k1 * _Arows] =
							shm[threadIdx.x * _vl];
					idx[binL2 + k * _Arows + threadIdx.x * _k1 * _Arows] = binL2
							+ binL1[threadIdx.x] * _c1scale;

				}

				__syncthreads();
			}
		}

		// sort the results
#if 1
		for (int i = 0; i < _p; i++) {

			if (threadIdx.x < _NP2)
				shm[threadIdx.x] = 1000000000.;

			// copy to original shm
			if (threadIdx.x < _k1 * _Arows) {
				shm[threadIdx.x] = val[threadIdx.x + i * _k1 * _Arows];
				shmIdx[threadIdx.x] = idx[threadIdx.x + i * _k1 * _Arows];
			}
			__syncthreads();

			bitonic2(shm, shmIdx, _NP2);

			if (threadIdx.x < _k1 * _Arows) {
				val[threadIdx.x + i * _k1 * _Arows] = shm[threadIdx.x];
				idx[threadIdx.x + i * _k1 * _Arows] = shmIdx[threadIdx.x];
			}

			__syncthreads();

		}
#endif

		// write out the sorted bins
		for (int p = 0; p < _p; p++) {
			if (threadIdx.x < _k1 * _Arows) {
				_assignVal[iter * _k1 * _p * _Arows + p * _k1 * _Arows
						+ threadIdx.x] = val[threadIdx.x + p * _k1 * _Arows];
				_assignIdx[iter * _k1 * _p * _Arows + p * _k1 * _Arows
						+ threadIdx.x] = idx[threadIdx.x + p * _k1 * _Arows];
			}
		}

	}
}

// returns the product of the _num[0]*... _num[_n-1];
__device__ int numCombis(uint *_numbers, uint _n) {
	float val = _numbers[0];

	for (int i = 1; i < _n; i++)
		val *= _numbers[i];

	return val;

}

__global__ void selectBinKernel(uint* _assign, uint* _nBins,
		const float *_assignVal, const uint* _assignIdx,
		const uint* _nElemPerBin, uint _Arows, uint _Brows, uint _p, uint _vl,
		uint _nClusters1, uint _nClusters2, uint _k1, uint _k, uint _maxTrials,
		uint _maxOutBin, uint _c1scale, const uint *_distSeq, uint _numDistSeq,
		uint _distCluster) {

// instead of the Dijkstra do the brute-force thing
	extern __shared__ float shm[];

	float* shmIter = shm;

	float* val = shmIter;
	shmIter += _p * _k1 * _Arows;
	uint* idx = (uint*) shmIter;
	shmIter += _p * _k1 * _Arows;
	uint* numbers = (uint*) shmIter;
	shmIter += _p;
	uint* denom = (uint*) shmIter;
	shmIter += _p;

	uint* outIdx = (uint*) shmIter;
	shmIter += blockDim.x;

	float* dist = (float*) shmIter;
	shmIter += blockDim.x;

	uint* nElem = (uint*) shmIter;
	shmIter += blockDim.x;

	uint* outBin = (uint*) shmIter;
	shmIter += _maxOutBin;

	uint& nOutBins = *(uint*) shmIter;
	shmIter += 1;

	uint& nElements = *(uint*) shmIter;
	shmIter += 1;

	uint& nIter = *(uint*) shmIter;
	shmIter += 1;

//
//	denom[0] = 1;
//	numbers[0] = _nClusters2 * NUM_NEIGHBORS;
//	for (int i = 1; i < _p; i++) {
//		numbers[i] = _nClusters2 * NUM_NEIGHBORS;
//		denom[i] = denom[i - 1] * numbers[i - 1];
//	}

	if (threadIdx.x < _p) {
		numbers[threadIdx.x] = _distCluster;
	}

	__syncthreads();

	if (threadIdx.x == 0) {
		denom[0] = 1;
		for (int i = 1; i < _p; i++) {
			denom[i] = denom[i - 1] * numbers[i - 1];
		}
	}

	__syncthreads();

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		__syncthreads();
		// read the sorted assignment
		for (int p = 0; p < _p; p++) {
			if (threadIdx.x < _k1 * _Arows) {
				val[threadIdx.x + p * _k1 * _Arows] = _assignVal[iter * _k1 * _p
						* _Arows + p * _k1 * _Arows + threadIdx.x];
				idx[threadIdx.x + p * _k1 * _Arows] = _assignIdx[iter * _k1 * _p
						* _Arows + p * _k1 * _Arows + threadIdx.x];

			}
		}

		// TODO loop multiple times to include sufficiently many bins at the end

		if (threadIdx.x == 0) {
			nOutBins = 0;
			nElements = 0;
			nIter = 0;

		}

		__syncthreads();

		while ((nElements < _k) && (nIter < _maxTrials)) {

			// generate all possible bins within the bounds given by numbers[]
			// calc the corresponding binIdx in the DB and the distance to the cluster center
			dist[threadIdx.x] = 0.;

			// TODO fix 4
			uint bin[4]; // maximum number for p
			for (int p = 0; p < _p; p++) {
				bin[p] = (_distSeq[nIter * blockDim.x + threadIdx.x] / denom[p])
						% numbers[p];
				dist[threadIdx.x] += val[bin[p] + p * _k1 * _Arows];
				bin[p] = idx[bin[p] + p * _k1 * _Arows];

			}

//			printf("%d: %d = %d %d \n", threadIdx.x,
//					_distSeq[nIter * blockDim.x + threadIdx.x], bin[0], bin[1]);

			if (threadIdx.x >= _numDistSeq) {
				dist[threadIdx.x] = 99999999.;
			}

//			if (threadIdx.x < 10)
//			printf("%d: %d %d === %f \n", threadIdx.x, bin[0], bin[1], dist[threadIdx.x]);

//			printf("%d: %d %d %d %d === %f \n", threadIdx.x, bin[0], bin[1],
//					bin[2], bin[3], dist[threadIdx.x]);

			__syncthreads();

			// TODO _p1 + _p2
			outIdx[threadIdx.x] = calcIdxSequential(bin, _p, _nClusters1,
					_nClusters2, _c1scale);

			//	printf("%d --- %d \n", threadIdx.x, outIdx[threadIdx.x]);

			if (threadIdx.x < 100)
				printf("%d: %d %d === %f  -- %d \n", threadIdx.x, bin[0],
						bin[1], dist[threadIdx.x], outIdx[threadIdx.x]);

			__syncthreads();

			// sort all cluster centers based on the distance
			bitonic2(dist, outIdx, blockDim.x);

			//	if (outIdx[threadIdx.x] < )
			nElem[threadIdx.x] = _nElemPerBin[outIdx[threadIdx.x]];

			__syncthreads();

			// collect the number of vectors in all the bins
			// prepare output of bins with one or more vectors until the maximum number of vectors is reached
			// (performs a sequential reduction)
			if (threadIdx.x == 0) {

				for (int i = 0; i < blockDim.x; i++) {
					if ((nElements > _k) || (nOutBins > _maxOutBin))
						break;

					printf("outIdx: %d, %f, %d \n", outIdx[i], dist[i],
							nElem[i]);

					int n = nElem[i];
					if (n > 0) {
						outBin[nOutBins] = outIdx[i];
						nOutBins++;
						nElements += n;

					}

				}

				nIter++;
			}

			__syncthreads();
		}

		// write out result
		for (int b = threadIdx.x; b < nOutBins; b += blockDim.x)
			_assign[iter * _maxOutBin + b] = outBin[b];

		if (threadIdx.x == 0) {
			_nBins[iter] = nOutBins;
		}

	}
}

#if 0
__global__ void selectBinKernel(uint* _assign, uint* _nBins,
		const float *_assignVal, const uint* _assignIdx,
		const uint* _nElemPerBin, uint _Arows, uint _Brows, uint _p, uint _vl,
		uint _nClusters1, uint _nClusters2, uint _k1, uint _k,
		uint _maxTrialBin, uint _maxOutBin, uint _c1scale) {

// now do the crazy select the best bin thing
// corresponds to an n-dimensional Dijkstra stopping after having sufficiently many bins for _k output vectors
// for each of the p best segments we have to identify the next best possible combination

	extern __shared__ float shm[];

	float* shmIter = shm;

	bool* state = (bool*) shmIter;
	shmIter += _maxTrialBin * _p;

	float* val = shmIter;
	shmIter += _p * _k1 * _Arows;
	uint* idx = (uint*) shmIter;
	shmIter += _p * _k1 * _Arows;

	uint* shmTmp = (uint*) shmIter;
	shmIter += blockDim.x;
	uint* factor = (uint*) shmIter;
	shmIter += blockDim.x;

	float* bestDist = shmIter;
	shmIter += _p;
	uint &selected(*(uint*) shmIter);
	shmIter += 1;
	uint &nBins(*(uint*) shmIter);
	shmIter += 1;
	uint &nOutBins(*(uint*) shmIter);
	shmIter += 1;
	uint &nElements(*(uint*) shmIter);
	shmIter += 1;
	uint &bestThread(*(uint*) shmIter);
	shmIter += 1;
	bool &same(*(bool*) shmIter);
	shmIter += 1;

	uint* outIdx = (uint*) shmIter;
	shmIter += _maxTrialBin * _p;

	uint* outBin = (uint*) shmIter;
	shmIter += _maxOutBin;

// local memory
// assert(_p <= 16) ;
	uint v[16];
	uint bestV[16];
	uint bestIdx;
	float dist;
	uint counter;

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		__syncthreads();
		if (threadIdx.x == 0) {
			nBins = 0;
			nOutBins = 0;
			selected = 11111111.;
			nElements = 0;
		}
		__syncthreads();

		// read the sorted assignment
		for (int p = 0; p < _p; p++) {
			if (threadIdx.x < _k1 * _Arows) {
				val[threadIdx.x + p * _k1 * _Arows] = _assignVal[iter * _k1 * _p
				* _Arows + p * _k1 * _Arows + threadIdx.x];
				idx[threadIdx.x + p * _k1 * _Arows] = _assignIdx[iter * _k1 * _p
				* _Arows + p * _k1 * _Arows + threadIdx.x];

			}
		}

		// reset the state of all possible output vectors
		for (int i = threadIdx.x; i < _maxTrialBin * _p; i += blockDim.x) {
			state[i] = true;
		}

		__syncthreads();

		if (threadIdx.x == 0) {
			nBins = 0;
			selected = 1111111;

			for (int i = 0; i < _maxTrialBin * _p; i++) {
				if (state[i])
				nBins++;
			}

			//	printf("ntrues: %d out of %d \n", nBins, _maxTrialBin * _p);

			nBins = 0;
		}

		// add the first bin
		if (threadIdx.x < _p) {
			outIdx[nBins * _p + threadIdx.x] = 0;
			bestDist[threadIdx.x] = 11111111.;
			bestIdx = 0;

//			printf("val %f", val[threadIdx.x * _k1 * _Arows]);
		}

		__syncthreads();
		calcIdx(shmTmp, factor, outIdx + nBins * _p, idx, _k1 * _Arows, _p,
				_nClusters1, _nClusters2, _c1scale);

		if (threadIdx.x == 0) {
			outBin[nOutBins] = shmTmp[0];
			uint n = _nElemPerBin[outBin[nOutBins]];
//			printf("------------ firstBin: shmTmp: %d, nElem: %d    \n",
//					shmTmp[0], n);
			if (n > 0) {
				nOutBins++;
				nElements += n;
			}
			nBins++;
		}

		__syncthreads();
		//	_maxOutBin = 3;

		if (threadIdx.x < _p) {
			counter = 0;
		}

		while ((nElements < _k) && (nBins < _maxTrialBin)
				&& (nOutBins < _maxOutBin)) {
#if 1
			// prepare the next vector for the selected p
			if (selected < _p) {

				if (threadIdx.x == 0) {
					bestDist[selected] = 1111111111111.;
					bestThread = 11111111;
					shmTmp[0] = bestThread;
					shmTmp[1] = 0;
				}
				__syncthreads();

				if (threadIdx.x == selected) {
					uint n = 0;
					for (int i = 0; i < nBins; i++) {
						if (state[selected * _maxTrialBin + i])
						n++;
					}
//					printf("\n nnnn: %d out of %d \n", n, nBins);
				}

				__syncthreads();

				for (int i = threadIdx.x; i < nBins; i += blockDim.x) {
					if (state[selected * _maxTrialBin + i]) { // this one has not be used previously
						// load starting vector
						for (int p = 0; p < _p; p++) {
							v[p] = outIdx[i * _p + p];
						}
						v[selected] += 1;
						if (v[selected] < _k1 * _Arows) {
							dist = 0.;
							for (int p = 0; p < _p; p++) {
								dist += val[v[p] + p * _k1 * _Arows];
							}
						} else {
							dist = 1010101.;
						}

						// figure out which is the best one   -- poor mans atomicMin
//						for (int v = 0; v < blockDim.x; v++)
//							if (bestDist[selected] > dist) {
//								bestDist[selected] = dist;
//							}

						while (bestDist[selected] > dist) {
							bestDist[selected] = dist;
							shmTmp[0] = i;
							// printf("up %d %f,  ", i, dist);
						}
						atomicAdd(shmTmp + 1, 1);

					}
				}
				__syncthreads();
#if 0
				// make sure just one thread writes its result
				for (int i = threadIdx.x; i < nBins; i += blockDim.x) {
					if (state[selected * _maxTrialBin + i]
							&& (dist == bestDist[selected]))
					bestThread = i;
				}
				__syncthreads();
#endif
				// bring the selected thread to the newest state
				if (threadIdx.x == selected) {

					dist = bestDist[selected];
					bestIdx = shmTmp[0]; // bestThread;

					if (bestIdx < nBins) {
						for (int p = 0; p < _p; p++) {
							v[p] = outIdx[bestIdx * _p + p];
						}
						v[selected] += 1;

						for (int p = 0; p < _p; p++) {
							bestV[p] = v[p];
						}
					}

//					printf(" newVec %d %d %d %d %f: ", selected, shmTmp[1],
//							counter, bestIdx, dist);
//					for (int p = 0; p < _p; p++) {
//						printf("%d ", bestV[p]);
//					}
//					printf("   ");
				}

				__syncthreads();
			}

#endif

			// try out a novel vector based on the last that came in
			if (threadIdx.x < _p) {
				for (int p = 0; p < _p; p++) {
					v[p] = outIdx[(nBins - 1) * _p + p];
				}
				v[threadIdx.x] += 1;
				if (v[threadIdx.x] < _k1 * _Arows) {
					dist = 0.;
					for (int p = 0; p < _p; p++) {
						dist += val[v[p] + p * _k1 * _Arows];
					}
				} else
				dist = 111111111.;

				// if this is the currently best, keep it
				if (dist < bestDist[threadIdx.x]) {
					bestDist[threadIdx.x] = dist;
					bestIdx = (nBins - 1);
					for (int p = 0; p < _p; p++) {
						bestV[p] = v[p];
					}
				}

			}

			__syncthreads();
			// find out the next best from the p proposed ones
			if (threadIdx.x == 0) {
				float selectDist = bestDist[0];
				selected = 0;

				for (int p = 1; p < _p; p++) {
					if (bestDist[p] < selectDist) {
						selectDist = bestDist[p];
						selected = p;
					}
				}
				//	printf("selDist: %f \n", selectDist);

			}

#if 1
			__syncthreads();
			if (threadIdx.x == selected) {
				state[threadIdx.x * _maxTrialBin + bestIdx] = false;
				counter++;
				// check if the vector has just been added
				same = true;
				for (int p = 0; p < _p; p++) {
					if (bestV[p] != outIdx[(nBins - 1) * _p + p])
					same = false;
				}
				if (!same) {
//					printf("\n");
					for (int p = 0; p < _p; p++) {
						outIdx[nBins * _p + p] = bestV[p];
//						printf("%d ", bestV[p]);
					}
//					printf("\n");
				}

				//nBins++;

			}
#endif

			__syncthreads();

#if 1
			// add to final output
			if (!same) {
				calcIdx(shmTmp, factor, outIdx + nBins * _p, idx, _k1 * _Arows,
						_p, _nClusters1, _nClusters2, _c1scale);

				if (threadIdx.x == 0) {

					outBin[nOutBins] = shmTmp[0];

					uint n = 0;
					if (outBin[nOutBins] < 4096 * 4096)
					n = _nElemPerBin[outBin[nOutBins]];

					//		printf("shmTmp: %d, %d %d    ", shmTmp[0], n, selected);
					// make sure to only keep filled bins
					if (n > 0) {
						nOutBins++;
						nElements += n;
					}
					nBins++;

					//	printf("nElem %d ", nElements);
				}
			}
#endif

//			if (threadIdx.x == 0)
//				nElements = _k;

			__syncthreads();
		}

// write out final list
#if 1
		for (int b = threadIdx.x; b < nOutBins; b += blockDim.x)
		_assign[iter * _maxOutBin + b] = outBin[b];

		if (threadIdx.x == 0) {
			_nBins[iter] = nOutBins;
		}
#endif

	} // iter
}
#endif

void ProTree::getKBestAssignment2(float *_assignVal, uint *_assignIdx,
		const float* _A, const float* _B, uint _Arows, uint _Brows,
		const uint *_assign1, uint _nClusters1, uint _k1) const {

	uint NP2 = log2(_k1 * _Arows);

	cout << "NP2 " << NP2 << endl;

//	assert(d_dim >= (2 * NP2));

	int nThreads = (d_dim > (2 * NP2)) ? d_dim : (2 * NP2);

	dim3 block(nThreads, 1, 1);
	dim3 grid((_Brows < 1024) ? _Brows : 1024, 1, 1);

	uint shm = (2 * nThreads + d_p + 2 * _k1 * d_p * d_nClusters2)
			* sizeof(float);

	cout << "shm" << shm << endl;

//	uint c1scale = pow(_nClusters1, d_p);
	uint c1scale = d_nClusters2;

	assignKBestClusterKernel2<<<grid, block, shm>>>(_assignVal, _assignIdx, _A,
			_B, _Arows, _Brows, d_dim, d_p, d_vl, _assign1, _nClusters1, _k1,
			NP2, c1scale);

	checkCudaErrors(cudaDeviceSynchronize());
}

void ProTree::getBins(uint *_bins, uint *_nBins, const float *_assignVal,
		const uint *_assignIdx, uint _N, uint _k1, uint _k2, uint _maxBins) {

	uint nThreads = 128; // 32;
	dim3 block(nThreads, 1, 1);
	dim3 grid((_N < 1024) ? _N : 1024, 1, 1);

//	grid.x = 1;
//
//	_N = 1;

	cudaMemset(_bins, 0, _N * _maxBins * sizeof(uint));

	uint maxTrials = 1;

//	uint shm = (2 * d_p * _k1 * d_nClusters + 2 * nThreads + d_p + 8
//			+ maxTrialBins * d_p * 2 + _maxBins * 3) * sizeof(float);

	uint shm = (2 * d_p * _k1 * d_nClusters2 + 2 * d_p + 3 * nThreads + _maxBins
			+ 3) * sizeof(float);

	cout << "shm: " << shm << endl;

	cout << "distCluster: " << d_distCluster << endl;

//	uint c1scale = pow(d_nClusters, d_p);
	uint c1scale = d_nClusters2;

	selectBinKernel<<<grid, block, shm>>>(_bins, _nBins, _assignVal, _assignIdx,
			d_binCounts, d_nClusters2, _N, d_p, d_vl, d_nClusters, d_nClusters2,
			_k1, _k2, maxTrials, _maxBins, c1scale, d_distSeq, d_numDistSeq,
			d_distCluster);

//	selectBinKernel<<<grid, block, shm>>>(_bins, _nBins, _assignVal, _assignIdx,
//			d_binCounts, d_nClusters2, _N, d_p, d_vl, d_nClusters, d_nClusters2,
//			_k1, _k2, maxTrialBins, _maxBins, c1scale);

	checkCudaErrors(cudaDeviceSynchronize());

}
__global__ void getKBestVectorsKernel(float *_bestDist, uint* _bestIdx,
		const float* _dbVec, const uint* _dbIdx, const uint* _binPrefix,
		const uint* _binCounts, const float* _Q, const uint* _assignedBins,
		const uint* _assignedNBins, uint _QN, uint _dim, uint _maxBins, uint _k,
		uint _maxVec) {

	extern __shared__ float shm[];

	float* shmIter = shm + _dim;

	float* val = shmIter;
	shmIter += _maxVec;
	uint* idx = (uint*) shmIter;
	shmIter += _maxVec;

// in shm;
	uint &count(*(uint*) shmIter);
	shmIter++;
	uint &nBins(*(uint*) shmIter);
	shmIter++;
	uint &currentBin(*(uint*) shmIter);
	shmIter++;
	uint &nVec(*(uint*) shmIter);
	shmIter++;

//	float minVal;
//	uint minIdx;

	idx[threadIdx.x] = 0;

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _QN; iter += gridDim.x) {

		__syncthreads();

		// load vector
		float b = _Q[iter * _dim + threadIdx.x];

		if (threadIdx.x == 0) {
//			printf("%d \n ", iter);
			nBins = _assignedNBins[iter];
//			printf("iter %d  nBins: %d \n ", iter, nBins);
			count = 0;
		}
		__syncthreads();

		// loop over the best assigned bins
		for (int bin = 0; bin < nBins; bin++) {
#if 1
			if (threadIdx.x == 0) {
				currentBin = _assignedBins[iter * _maxBins + bin];

				nVec = _binCounts[currentBin];

//				printf("currentBin %d, nVec: %d, prefix: %d \n", currentBin,
//						nVec, _binPrefix[currentBin]);
			}
			__syncthreads();

			// loop over all vectors of the bin
			for (int a = 0; a < nVec; a++) {
				if (count >= _maxVec)
					break;

//				if (threadIdx.x == 0)
//					printf("vec %d \n", _dbIdx[_binPrefix[currentBin] + a]);

				if (threadIdx.x < _dim) {
					shm[threadIdx.x] = sqr(
							b
									- _dbVec[_dbIdx[_binPrefix[currentBin] + a]
											* _dim + threadIdx.x]);
				}

				for (uint stride = _dim >> 1; stride > 0; stride >>= 1) {
					__syncthreads();

					if (threadIdx.x < stride)
						shm[threadIdx.x] += shm[threadIdx.x + stride];
				}
				__syncthreads();

				// store the result
				if (threadIdx.x == 0) {

					val[count] = shm[0];
					idx[count] = _dbIdx[_binPrefix[currentBin] + a];
//					printf("idx: %d dist: %f \n", idx[count], val[count]);

					count++;
				}

				__syncthreads();
			}
#endif

		}
#if 1

		// sort the results
		if ((threadIdx.x >= count) && (threadIdx.x < _maxVec))
			val[threadIdx.x] = 10000000.;

		__syncthreads();
//		if (threadIdx.x == 0) {
//			for (int i = 0; i < count; i++)
//
//				printf("before idx: %d dist: %f \n", idx[i], val[i]);
//		}
//
//		__syncthreads();

		bitonic2(val, idx, _maxVec);

//		if (threadIdx.x == 0) {
//			for (int i = 0; i < count; i++)
//
//				printf("after idx: %d dist: %f \n", idx[i], val[i]);
//		}

		if ((threadIdx.x >= count) && (threadIdx.x < _maxVec))
			val[threadIdx.x] = 0.;

		if (threadIdx.x < _k) {
			_bestDist[iter * _k + threadIdx.x] = val[threadIdx.x];
			_bestIdx[iter * _k + threadIdx.x] = idx[threadIdx.x];
		}
#endif
	}
}

void ProTree::getKBestVectors(float *_bestDist, uint *_bestIdx,
		const uint *_bins, const uint *_nBins, uint _maxBins, const float* _Q,
		uint _QN, uint _k) {

	uint nThreads = d_dim;
	dim3 block(nThreads, 1, 1);
	dim3 grid((_QN < 1024) ? _QN : 1024, 1, 1);

//	_QN = 1;

	uint maxVec = 2 * log2(_k);

	if (maxVec > d_dim)
		block = dim3(maxVec, 1, 1);

	uint shmSize = (d_dim + 2 * maxVec + 10) * sizeof(float);

	getKBestVectorsKernel<<<grid, block, shmSize>>>(_bestDist, _bestIdx,
			d_dbVec, d_dbIdx, d_binPrefix, d_binCounts, _Q, _bins, _nBins, _QN,
			d_dim, _maxBins, _k, maxVec);

	checkCudaErrors(cudaDeviceSynchronize());

}
#if 0
void ProTree::queryKNN(const float* _Q, uint _QN, uint _k) {

	uint* assignd;
	uint* assignd2;

	cudaMalloc(&assignd, _QN * d_p * sizeof(uint));
	cudaMalloc(&assignd2, _QN * d_p * sizeof(uint));

	getAssignment(assignd, d_codeBook, _Q, d_nClusters, _QN);

	getAssignment2(assignd2, d_codeBook2, _Q, d_nClusters2, _QN, assignd,
			d_nClusters);

// create list of neighboring first/level

}
#endif

void ProTree::testKNN(const float* _Q, uint _QN) {

//	outputVecUint("prefix", d_binPrefix + 816791, 20);
//	outputVecUint("prefix", d_binCounts + 816791, 20);
//	outputVecUint("dbidx", d_dbIdx + 789975, 9);

	uint k1 = 4;

	prepareDistSequence(d_nClusters2 * k1, d_p);

	uint* assignd;
	uint* assignd2;
	cudaMalloc(&assignd, k1 * d_p * _QN * sizeof(uint));
	cudaMalloc(&assignd2, k1 * d_p * _QN * sizeof(uint));

	getKBestAssignment(assignd, d_codeBook, _Q, d_nClusters, _QN, k1);

	float *assignVal;
	uint *assignIdx;

	cudaMalloc(&assignVal, _QN * d_p * k1 * d_nClusters2 * sizeof(float));
	cudaMalloc(&assignIdx, _QN * d_p * k1 * d_nClusters2 * sizeof(uint));

	getKBestAssignment2(assignVal, assignIdx, d_codeBook2, _Q, d_nClusters2,
			_QN, assignd, d_nClusters, k1);

	uint *idx = new uint[d_p * k1 * d_nClusters2];
	float *val = new float[d_p * k1 * d_nClusters2];

	cudaMemcpy(val, assignVal, d_p * k1 * d_nClusters2 * sizeof(float),
			cudaMemcpyDeviceToHost);

	for (int p = 0; p < d_p; p++) {
		for (int i = 0; i < k1 * d_nClusters2; i++)
			cout << val[p * k1 * d_nClusters2 + i] << " ";
		cout << endl << endl;
	}

	uint k2 = 40;
	k2 = 4096;
//	uint maxBins = 40;
	uint maxBins = 200;

	uint* nBins;
	uint* bins;

	cudaMalloc(&nBins, _QN * maxBins * sizeof(uint));
	cudaMalloc(&bins, _QN * maxBins * sizeof(uint));

//	getBins(bins, nBins, assignVal, assignIdx, _QN, k1, k2, maxBins);
	getBins(bins, nBins, assignVal, assignIdx, 1, k1, k2, maxBins);

	cout << "done with bins!!!!!!" << endl;
	k2 = 256;

	uint maxVec = k2;
	float* bestDist;
	uint* bestIdx;
	cudaMalloc(&bestDist, _QN * maxVec * sizeof(float));
	cudaMalloc(&bestIdx, _QN * maxVec * sizeof(uint));

//	getKBestVectors(bestDist, bestIdx, bins, nBins, maxBins, _Q, _QN, k2);
	getKBestVectors(bestDist, bestIdx, bins, nBins, maxBins, _Q, 1, k2);

	uint* bestIdxh = new uint[maxVec];
	float* bestDisth = new float[maxVec];

	cudaMemcpy(bestIdxh, bestIdx, maxVec * sizeof(uint),
			cudaMemcpyDeviceToHost);
	cudaMemcpy(bestDisth, bestDist, maxVec * sizeof(float),
			cudaMemcpyDeviceToHost);

	for (int i = 0; i < maxVec; i++) {
		cout << i << " " << bestIdxh[i] << "  " << bestDisth[i] << endl;
	}

	cout << endl;

	float* resd;
	cudaMalloc(&resd, d_p * d_NdbVec * sizeof(float));
	calcDist(resd, d_dbVec, _Q, d_NdbVec, 1);

//outputVec("Res:", resd, 20);

	float* resh = new float[d_p * d_NdbVec];

	cudaMemcpy(resh, resd, d_p * d_NdbVec * sizeof(float),
			cudaMemcpyDeviceToHost);

	vector<pair<float, uint> > ddd;
	ddd.resize(d_NdbVec);

	for (int i = 0; i < d_NdbVec; i++) {
		float val = 0.;
		for (int p = 0; p < d_p; p++)
			val += resh[i * d_p + p];
		ddd[i] = pair<float, uint>(val, i);
	}

	sort(ddd.begin(), ddd.end());

	getAssignment(assignd, d_codeBook, _Q, d_nClusters, 1);
	outputVecUint("assign: ", assignd, 4);

	getKBestAssignment(assignd, d_codeBook, _Q, d_nClusters, 1, k1);
	outputVecUint("assign: ", assignd, k1 * d_p);
	outputVecUint("assignIdx2-1: ", assignIdx, k1);
	outputVecUint("assignIdx2-2: ", assignIdx + k1 * d_nClusters2, k1);

	cout << "distance by brute-force search: " << endl;
	for (int i = 0; i < 20; i++) {
		cout << i << "  " << ddd[i].first << "  " << ddd[i].second << endl;
		getKBestAssignment(assignd, d_codeBook, d_dbVec + ddd[i].second * d_dim,
				d_nClusters, 1, k1);
//		outputVecUint("assign: ", assignd, k1 * d_p);
		getAssignment2(assignd2, d_codeBook2, d_dbVec + ddd[i].second * d_dim,
				d_nClusters2, 1, assignd, d_nClusters);
//		outputVecUint("assign2: ", assignd2, d_p);
		getKBestAssignment2(assignVal, assignIdx, d_codeBook2,
				d_dbVec + ddd[i].second * d_dim, d_nClusters2, 1, assignd,
				d_nClusters, k1);
		outputVecUint("assignIdx2-1: ", assignIdx, k1);
		outputVecUint("assignIdx2-2: ", assignIdx + k1 * d_nClusters2, k1);
//		outputVecUint("", assignd + 4, 4);
//		outputVecUint("", assignd + 8, 4);
	}

	cudaFree(bestIdx);
	cudaFree(bestDist);

	cudaFree(bins);
	cudaFree(nBins);
	cudaFree(assignIdx);
	cudaFree(assignVal);
	cudaFree(assignd2);
	cudaFree(assignd);

	delete[] val;
	delete[] idx;
}

__global__ void selectMultiBinKernel(uint* _assign, uint* _nBins,
		const float *_assignVal, const uint* _assignIdx,
		const uint* _nElemPerBin, uint _Arows, uint _Brows, uint _p, uint _vl,
		uint _nClusters1, uint _nClusters2, uint _k1, uint _k, uint _maxTrials,
		uint _maxOutBin, uint _c1scale, const uint *_distSeq, uint _groupParts,
		uint _nDBs) {

// follows the _distSequence
	extern __shared__ float shm[];

	float* shmIter = shm;

	float* val = shmIter;
	shmIter += _p * _k1 * _Arows;
	uint* idx = (uint*) shmIter;
	shmIter += _p * _k1 * _Arows;
	uint* numbers = (uint*) shmIter;
	shmIter += _groupParts;
	uint* denom = (uint*) shmIter;
	shmIter += _groupParts;

	uint* outIdx = (uint*) shmIter;
	shmIter += blockDim.x;

	float* dist = (float*) shmIter;
	shmIter += blockDim.x;

	uint* nElem = (uint*) shmIter;
	shmIter += blockDim.x;

	uint* outBin = (uint*) shmIter;
	shmIter += _maxOutBin;

	uint* nOutBins = (uint*) shmIter;
	shmIter += _nDBs;

	uint& nElements = *(uint*) shmIter;
	shmIter += 1;

	uint& nIter = *(uint*) shmIter;
	shmIter += 1;

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _Brows; iter += gridDim.x) {

		__syncthreads();

		// read the sorted assignment
		for (int p = 0; p < _p; p++) {
			if (threadIdx.x < _k1 * _Arows) {
				val[threadIdx.x + p * _k1 * _Arows] = _assignVal[iter * _k1 * _p
						* _Arows + p * _k1 * _Arows + threadIdx.x];
				idx[threadIdx.x + p * _k1 * _Arows] = _assignIdx[iter * _k1 * _p
						* _Arows + p * _k1 * _Arows + threadIdx.x];

			}
		}

		denom[0] = 1;
		numbers[0] = _nClusters2 * NUM_NEIGHBORS;
		for (int i = 1; i < _groupParts; i++) {
			numbers[i] = _nClusters2 * NUM_NEIGHBORS;
			denom[i] = denom[i - 1] * numbers[i - 1];
		}

		for (int db = 0; db < _nDBs; db++) {

			if (threadIdx.x == 0) {
				nOutBins[db] = 0;
				nElements = 0;
				nIter = 0;

			}

			__syncthreads();

			while ((nElements < _k) && (nIter < _maxTrials)) {

				// generate all possible bins within the bounds given by numbers[]
				// calc the corresponding binIdx in the DB and the distance to the cluster center
				dist[threadIdx.x] = 0.;

				// TODO fix 4
				uint bin[4]; // maximum number for p
				for (int p = 0; p < _groupParts; p++) {
					bin[p] = (_distSeq[nIter * blockDim.x + threadIdx.x]
							/ denom[p]) % numbers[p];
					dist[threadIdx.x] += val[bin[p]
							+ (p + db * _groupParts) * _k1 * _Arows];
					bin[p] =
							idx[bin[p] + (p + db * _groupParts) * _k1 * _Arows];

				}

//			printf("%d: %d %d %d %d === %f \n", threadIdx.x, bin[0], bin[1],
//					bin[2], bin[3], dist[threadIdx.x]);

				__syncthreads();

				// TODO _p1 + _p2
				outIdx[threadIdx.x] = calcIdxSequential(bin, _groupParts,
						_nClusters1, _nClusters2, _c1scale);

				//	printf("%d --- %d \n", threadIdx.x, outIdx[threadIdx.x]);

				__syncthreads();

				// sort all cluster centers based on the distance
				bitonic2(dist, outIdx, blockDim.x);

				//	if (outIdx[threadIdx.x] < )
				nElem[threadIdx.x] = _nElemPerBin[outIdx[threadIdx.x]];

				__syncthreads();

				// collect the number of vectors in all the bins
				// prepare output of bins with one or more vectors until the maximum number of vectors is reached
				// (performs a sequential reduction)
				if (threadIdx.x == 0) {

					for (int i = 0; i < blockDim.x; i++) {
						if ((nElements > _k) || (nOutBins[db] > _maxOutBin))
							break;

						printf("outIdx: %d, %f, %d \n", outIdx[i], dist[i],
								nElem[i]);

						int n = nElem[i];
						if (n > 0) {
							outBin[nOutBins[db]] = outIdx[i];
							nOutBins[db]++;
							nElements += n;
						}

					}

					printf("done with db %d \n", db);

					nIter++;
				}

				__syncthreads();
			}

			// write out result
			for (int b = threadIdx.x; b < nOutBins[db]; b += blockDim.x)
				_assign[(iter * _nDBs + db) * _maxOutBin + b] = outBin[b];
		}

		// write out nOutBins
		if (threadIdx.x < _nDBs) {
			_nBins[iter * _nDBs + threadIdx.x] = nOutBins[threadIdx.x];
		}

	}
}

#if 0

void ProTree::getMultiBins(uint *_bins, uint *_nBins, const float *_assignVal,
		const uint *_assignIdx, uint _N, uint _k1, uint _k2, uint _maxBins,
		uint _groupParts, uint _nDBs) {

	uint nThreads = 128; // 32;
	dim3 block(nThreads, 1, 1);
	dim3 grid((_N < 1024) ? _N : 1024, 1, 1);

	cudaMemset(_bins, 0, _nDBs * _N * _maxBins * sizeof(uint));

	uint maxTrials = 50;

	uint shm = (2 * d_p * _k1 * d_nClusters + 2 * d_p + 3 * nThreads + _maxBins
			+ 3) * sizeof(float);

	cout << "shm: " << shm << endl;

	selectMultiBinKernel<<<grid, block, shm>>>(_bins, _nBins, _assignVal,
			_assignIdx, d_binCounts, d_nClusters2, _N, d_p, d_vl, d_nClusters,
			d_nClusters2, _k1, _k2, maxTrials, _maxBins, d_distSeq, _groupParts,
			_nDBs);

	checkCudaErrors(cudaDeviceSynchronize());

}
#endif

void ProTree::getMultiBins(uint *_bins, uint *_nBins, const float *_assignVal,
		const uint *_assignIdx, uint _N, uint _k1, uint _k2, uint _maxBins,
		uint _groupParts, uint _nDBs) {

	uint nThreads = 128; // 32;
	dim3 block(nThreads, 1, 1);
	dim3 grid((_N < 1024) ? _N : 1024, 1, 1);

	cudaMemset(_bins, 0, _nDBs * _N * _maxBins * sizeof(uint));

	uint maxTrials = 50;

	uint shm = (2 * d_p * _k1 * d_nClusters2 + 2 * d_p + 3 * nThreads + _maxBins
			+ _nDBs + 3) * sizeof(float);

	cout << "shm: " << shm << endl;

//	uint c1scale = pow(d_nClusters, d_p);
	uint c1scale = d_nClusters2;

	selectMultiBinKernel<<<grid, block, shm>>>(_bins, _nBins, _assignVal,
			_assignIdx, d_binCounts, d_nClusters2, _N, d_p, d_vl, d_nClusters,
			d_nClusters2, _k1, _k2, maxTrials, _maxBins, c1scale, d_distSeq,
			_groupParts, _nDBs);

	checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void getMultiKVectorIDsKernel(uint* _bestIdx, uint* _nVec,
		const uint* _dbIdx, const uint* _binPrefix, const uint* _binCounts,
		uint _nDBBins, const float* _Q, const uint* _assignedBins,
		const uint* _assignedNBins, uint _QN, uint _dim, uint _maxBins, uint _k,
		uint _maxVecConsider, uint _maxVecOut, uint _groupParts, uint _nDBs) {

	extern __shared__ float shm[];

	float* shmIter = shm;

	float* val = shmIter;
	shmIter += _maxVecConsider;
	uint* idx = (uint*) shmIter;
	shmIter += _maxVecConsider;

// in shm;
	uint* nBins = (uint*) shmIter;
	shmIter += _nDBs;

	uint &count(*(uint*) shmIter);
	shmIter++;

	uint &currentBin(*(uint*) shmIter);
	shmIter++;
	uint &nVec(*(uint*) shmIter);
	shmIter++;

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _QN; iter += gridDim.x) {

		__syncthreads();

		val[threadIdx.x] = 0.;

		if (threadIdx.x < _nDBs) {
			nBins[threadIdx.x] = _assignedNBins[iter * _nDBs + threadIdx.x];
		}

		if (threadIdx.x == 0) {
			count = 0;
			nVec = 0;
		}

		__syncthreads();

		for (int db = 0; db < _nDBs; db++) {

			// loop over the best assigned bins
			for (int bin = 0; bin < nBins[db]; bin++) {

				if (threadIdx.x == 0) {

					count += nVec;

					currentBin = _assignedBins[(iter * _nDBs + db) * _maxBins
							+ bin];

					//	printf("currentBin %d \n", currentBin);
					nVec = _binCounts[db * _nDBBins + currentBin];

					if ((count + nVec) >= _maxVecConsider)
						nVec = _maxVecConsider - count;
				}
				__syncthreads();

				// fetch all the vector indices for the selected bin
				for (uint v = threadIdx.x; v < nVec; v += blockDim.x) {

					idx[count + v] = _dbIdx[_binPrefix[db * _nDBBins
							+ currentBin] + v];
					val[count + v] = idx[count + v];
				}

				__syncthreads();

			}
		}

		// sort the results
		if ((threadIdx.x >= count) && (threadIdx.x < _maxVecConsider))
			val[threadIdx.x] = 999999999.;

		__syncthreads();

		// sort all vectors by vector ID
		bitonic2(val, idx, _maxVecConsider);

		// each vector should only appear at maximum _nDBs times
		// count occurences
		if (threadIdx.x < count) {
			val[threadIdx.x] = 1.;
		} else if (threadIdx.x < _maxVecConsider)
			val[threadIdx.x] = 0.;

		for (int db = 1; db < _nDBs; db++) {
			if ((threadIdx.x + db) < count) {
				if (idx[threadIdx.x] == idx[threadIdx.x + db]) {
					val[threadIdx.x]++;
				}
			}
		}

		__syncthreads();

		// make sure to consider only the first occurence of each vectorID
		// (remove duplicates)
		if ((threadIdx.x + 1) < count) {
			if (idx[threadIdx.x] == idx[threadIdx.x + 1]) {
				val[threadIdx.x + 1] = 0.;
			}
		}

		// sort all vectorIDs descending by occurence
		bitonic2Descending(val, idx, _maxVecConsider);

		if (threadIdx.x == 0) {
			count = (count < _maxVecOut) ? count : _maxVecOut;
		}

		if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
			for (int i = 0; i < count; i++)
				printf("i: %d %d %f \n", i, idx[i], val[i]);
		}

		__syncthreads();

		if (threadIdx.x < count) {
			if (val[threadIdx.x] > 0.) {
				_bestIdx[iter * _maxVecOut + threadIdx.x] = idx[threadIdx.x];
				val[threadIdx.x] = 1.;
			}
		} else if (threadIdx.x < _maxVecConsider)
			val[threadIdx.x] = 0.;

		// count number of valid vectors (reduction)
		for (uint stride = _maxVecConsider >> 1; stride > 0; stride >>= 1) {
			__syncthreads();

			if (threadIdx.x < stride)
				val[threadIdx.x] += val[threadIdx.x + stride];
		}
		__syncthreads();

		if (threadIdx.x == 0) {
			_nVec[iter] = (uint) val[0];

		}

	}
}

//__global__ void getMultiKBestVectorsKernel(float *_bestDist, uint* _bestIdx,
//		const uint* _inIdx, const uint* _nVec, uint _maxVecIn,
//		const float* _dbVec, const uint* _dbIdx, const uint* _binPrefix,
//		const uint* _binCounts, const float* _Q, const uint* _assignedBins,
//		const uint* _assignedNBins, uint _QN, uint _dim, uint _maxBins, uint _k,
//		uint _maxVec, uint _groupParts, uint _nDBs)

__global__ void getMultiKBestVectorsKernel(float *_bestDist, uint* _bestIdx,
		const uint* _inIdx, const uint* _nVec, uint _maxVecIn,
		const float* _dbVec, const float* _Q, uint _QN, uint _dim,
		uint _maxBins, uint _k, uint _maxVec) {

	extern __shared__ float shm[];

	float* shmIter = shm + _dim;

	float* val = shmIter;
	shmIter += _maxVec;
	uint* idx = (uint*) shmIter;
	shmIter += _maxVec;

// in shm;
	uint &nVec(*(uint*) shmIter);
//	uint *nVec = (uint*) shmIter;
	shmIter++;

// loop over all corresponding vectors in B
	for (int iter = blockIdx.x; iter < _QN; iter += gridDim.x) {

		__syncthreads();

		if (threadIdx.x == 0) {
			nVec = _nVec[iter];
			nVec = (nVec < _maxVec) ? nVec : _maxVec;
		}
		__syncthreads();

		// load query vector
		float b = _Q[iter * _dim + threadIdx.x];

		// load all indices
		for (int a = threadIdx.x; a < nVec; a += blockDim.x) {
			idx[a] = _inIdx[iter * _maxVecIn + a];
		}

		__syncthreads();

//		if (threadIdx.x == 0) {
//			for (int a = 0; a < nVec; a++) {
//				printf("idx: %d %d \n", a, idx[a]);
//			}
//		}

		// loop over all selected vectors
		for (int a = 0; a < nVec; a++) {

			// compute the distance to the vector
			if (threadIdx.x < _dim)
				shm[threadIdx.x] = sqr(b - _dbVec[idx[a] * _dim + threadIdx.x]);

			for (uint stride = _dim >> 1; stride > 0; stride >>= 1) {
				__syncthreads();

				if (threadIdx.x < stride)
					shm[threadIdx.x] += shm[threadIdx.x + stride];
			}
			__syncthreads();

			// store the result
			if (threadIdx.x == 0) {

				val[a] = shm[0];

//				printf("idx: %d dist: %f \n", idx[a], val[a]);

			}
			__syncthreads();
		}

		// sort the results
		if ((threadIdx.x >= nVec) && (threadIdx.x < _maxVec))
			val[threadIdx.x] = 10000000.;

		__syncthreads();

		bitonic2(val, idx, _maxVec);

		if ((threadIdx.x >= nVec) && (threadIdx.x < _maxVec))
			val[threadIdx.x] = 0.;

		if (threadIdx.x < _k) {
			_bestDist[iter * _k + threadIdx.x] = val[threadIdx.x];
			_bestIdx[iter * _k + threadIdx.x] = idx[threadIdx.x];
		}

		__syncthreads();

	}
}

void ProTree::getMultiKBestVectors(float *_bestDist, uint *_bestIdx,
		const uint *_bins, const uint *_nBins, uint _maxBins, const float* _Q,
		uint _QN, uint _k) {

	uint maxVecConsider = 1024;
	uint maxVecOut = 1024;
	uint nThreads = maxVecConsider;
	dim3 block(nThreads, 1, 1);
	dim3 grid((_QN < 1024) ? _QN : 1024, 1, 1);

	uint shmSize = (2 * maxVecConsider + d_nDBs + 10) * sizeof(float);

	uint *selectIdx;   // array for storing nn vector IDs (size maxVecOut * _QN)
	uint *nVec;

	cudaMalloc(&selectIdx, _QN * maxVecOut * sizeof(uint));
	cudaMalloc(&nVec, _QN * sizeof(uint));

	getMultiKVectorIDsKernel<<<grid, block, shmSize>>>(selectIdx, nVec, d_dbIdx,
			d_binPrefix, d_binCounts, d_nBins, _Q, _bins, _nBins, _QN, d_dim,
			_maxBins, _k, maxVecConsider, maxVecOut, d_groupParts, d_nDBs);

	checkCudaErrors(cudaDeviceSynchronize());

	cout << "multi Vector IDs done" << endl;
//	_QN = 1;

	uint maxVec = 2 * log2(_k);

	nThreads = (maxVec > d_dim) ? maxVec : d_dim;
	block = dim3(nThreads, 1, 1);

	shmSize = (d_dim + 2 * maxVec + 10) * sizeof(float);

	cout << "maxVec: " << maxVec << " shm: " << shmSize << endl;

	getMultiKBestVectorsKernel<<<grid, block, shmSize>>>(_bestDist, _bestIdx,
			selectIdx, nVec, maxVecOut, d_dbVec, _Q, _QN, d_dim, _maxBins, _k,
			maxVec);

	checkCudaErrors(cudaDeviceSynchronize());

	cout << "multiKBestVectors done " << endl;

	cudaFree(nVec);
	cudaFree(selectIdx);

}

void ProTree::testMultiKNN(const float* _Q, uint _QN) {

	uint k1 = 32; // number of nearest neighbors selected on first level

	uint* assignd;
	cudaMalloc(&assignd, k1 * d_p * _QN * sizeof(uint));

	getKBestAssignment(assignd, d_codeBook, _Q, d_nClusters, _QN, k1);

	outputVecUint("assign1 ;", assignd, k1 * d_p);

	float *assignVal;
	uint *assignIdx;

	cudaMalloc(&assignVal, _QN * d_p * k1 * d_nClusters2 * sizeof(float));
	cudaMalloc(&assignIdx, _QN * d_p * k1 * d_nClusters2 * sizeof(uint));

	getKBestAssignment2(assignVal, assignIdx, d_codeBook2, _Q, d_nClusters2,
			_QN, assignd, d_nClusters, k1);

	outputVecUint("assign2", assignIdx, k1 * d_nClusters2);
	outputVec("assign2", assignVal, k1 * d_nClusters2);

	cout << "done with assignments" << endl;

	uint *idx = new uint[d_p * k1 * d_nClusters2];
	float *val = new float[d_p * k1 * d_nClusters2];

	cudaMemcpy(val, assignVal, d_p * k1 * d_nClusters2 * sizeof(float),
			cudaMemcpyDeviceToHost);

	for (int p = 0; p < d_p; p++) {
		for (int i = 0; i < k1 * d_nClusters2; i++)
			cout << val[p * k1 * d_nClusters2 + i] << " ";
		cout << endl << endl;
	}

	_QN = 1;

	uint k2 = 128;   // number of potential vectors for each DB
//	uint maxBins = 40;
	uint maxBins = 200;

	uint* nBins;
	uint* bins;

	cudaMalloc(&nBins, d_nDBs * _QN * maxBins * sizeof(uint));
	cudaMalloc(&bins, d_nDBs * _QN * maxBins * sizeof(uint));

	getMultiBins(bins, nBins, assignVal, assignIdx, _QN, k1, k2, maxBins,
			d_groupParts, d_nDBs);

	cout << "done with multi bins" << endl;

	uint maxVec = k2;
	float* bestDist;
	uint* bestIdx;
	cudaMalloc(&bestDist, _QN * maxVec * sizeof(float));
	cudaMalloc(&bestIdx, _QN * maxVec * sizeof(uint));

	getMultiKBestVectors(bestDist, bestIdx, bins, nBins, maxBins, _Q, _QN, k2);

	uint* bestIdxh = new uint[maxVec];
	float* bestDisth = new float[maxVec];

	cudaMemcpy(bestIdxh, bestIdx, maxVec * sizeof(uint),
			cudaMemcpyDeviceToHost);
	cudaMemcpy(bestDisth, bestDist, maxVec * sizeof(float),
			cudaMemcpyDeviceToHost);

	for (int i = 0; i < maxVec; i++) {
		cout << "best: " << i << " " << bestIdxh[i] << "  " << bestDisth[i]
				<< endl;
	}

	cout << endl;

/////////////////////////////////////////////////////////
// ?
/////////////////////////////////////////////////////////

	float* resd;
	cudaMalloc(&resd, d_p * d_NdbVec * sizeof(float));
	calcDist(resd, d_dbVec, _Q, d_NdbVec, 1);

//	outputVec("Res:", resd, 20);

	float* resh = new float[d_p * d_NdbVec];

	cudaMemcpy(resh, resd, d_p * d_NdbVec * sizeof(float),
			cudaMemcpyDeviceToHost);

	vector<pair<float, uint> > ddd;
	ddd.resize(d_NdbVec);

	for (int i = 0; i < d_NdbVec; i++) {
		float val = 0.;
		for (int p = 0; p < d_p; p++)
			val += resh[i * d_p + p];
		ddd[i] = pair<float, uint>(val, i);
	}

	sort(ddd.begin(), ddd.end());

	for (int i = 0; i < 20; i++)
		cout << " brute: " << ddd[i].second << "  " << ddd[i].first << endl;

#if 1

	getAssignment(assignd, d_codeBook, _Q, d_nClusters, 1);
	outputVecUint("assign: ", assignd, 4);

	getKBestAssignment(assignd, d_codeBook, _Q, d_nClusters, 1, k1);
	outputVecUint("assign: ", assignd, 4);
	outputVecUint("", assignd + 4, 4);
	outputVecUint("", assignd + 8, 4);

	cout << "distance by brute-force search: " << endl;
	for (int i = 0; i < 3; i++) {
		cout << i << "  " << ddd[i].first << endl;
		getKBestAssignment(assignd, d_codeBook, d_dbVec + ddd[i].second * d_dim,
				d_nClusters, 1, k1);

		getKBestAssignment2(assignVal, assignIdx, d_codeBook2,
				d_dbVec + ddd[i].second * d_dim, d_nClusters2, 1, assignd,
				d_nClusters, k1);

//		outputVecUint("assign: ", assignd, d_p);
//		outputVecUint("", assignd + d_p, d_p);
//		outputVecUint("", assignd + 2*d_p, d_p);
//		outputVecUint("", assignd + 3*d_p, d_p);
		outputVec("", assignVal, k1 * d_nClusters);
		for (int k = 0; k < 8; k++)
			outputVecUint("", assignIdx + (k * k1 * d_nClusters2),
					k1 * d_nClusters2);
	}

#endif

	cudaFree(bestIdx);
	cudaFree(bestDist);

	cudaFree(bins);
	cudaFree(nBins);
	cudaFree(assignIdx);
	cudaFree(assignVal);
	cudaFree(assignd);

	delete[] val;
	delete[] idx;
}

void ProTree::testLevel1(const float* _Q, uint _QN)
{

	uint k1 = 32; // number of nearest neighbors selected on first level

	uint* assignd;
	cudaMalloc(&assignd, k1 * d_p * _QN * sizeof(uint));

	getKBestAssignment(assignd, d_codeBook, _Q, d_nClusters, _QN, k1);

	outputVecUint("assign1 ;", assignd, k1 * d_p);

	float *assignVal;
	uint *assignIdx;

	cudaMalloc(&assignVal, _QN * d_p * k1 * d_nClusters2 * sizeof(float));
	cudaMalloc(&assignIdx, _QN * d_p * k1 * d_nClusters2 * sizeof(uint));

	_QN = 1;

//	uint k2 = 128;   // number of potential vectors for each DB
//	uint maxBins = 40;
//	uint maxBins = 200;

//	uint* nBins;
//	uint* bins;

//	uint maxVec = k2;
//	float* bestDist;
//	uint* bestIdx;

/////////////////////////////////////////////////////////
// ?
/////////////////////////////////////////////////////////

	float* resd;
	cudaMalloc(&resd, d_p * d_NdbVec * sizeof(float));
	calcDist(resd, d_dbVec, _Q, d_NdbVec, 1);

//	calcDist(resd, d_codeBook, _Q, d_nClusters, 1);
//	d_NdbVec = d_nClusters;

//	outputVec("Res:", resd, 20);

	float* resh = new float[d_p * d_NdbVec];

	cudaMemcpy(resh, resd, d_p * d_NdbVec * sizeof(float),
			cudaMemcpyDeviceToHost);

	vector<pair<float, uint> > ddd;
	ddd.resize(d_NdbVec);

	for (int i = 0; i < d_NdbVec; i++) {
		float val = 0.;
		for (int p = 0; p < d_p; p++)
			val += resh[i * d_p + p];
		ddd[i] = pair<float, uint>(val, i);
	}

	sort(ddd.begin(), ddd.end());

	for (int i = 0; i < 20; i++)
		cout << " brute: " << ddd[i].second << "  " << ddd[i].first << endl;

#if 1

	getAssignment(assignd, d_codeBook, _Q, d_nClusters, 1);
	outputVecUint("assign: ", assignd, 4);

	getKBestAssignment(assignd, d_codeBook, _Q, d_nClusters, 1, k1);
	outputVecUint("assign: ", assignd, 4);
	outputVecUint("", assignd + 4, 4);
	outputVecUint("", assignd + 8, 4);

	cout << "distance by brute-force search: " << endl;
	for (int i = 0; i < 3; i++) {
		cout << i << "  " << ddd[i].second << "  " << ddd[i].first << endl;

		for (int p = 0; p < d_p; p++)
			cout << resh[ddd[i].second * d_p + p] << "  ";
		cout << endl;

		getKBestAssignment(assignd, d_codeBook, d_dbVec + ddd[i].second * d_dim,
				d_nClusters, 1, k1);

		calcDist(resd, d_codeBook, d_dbVec + ddd[i].second * d_dim, d_nClusters,
				1);
		cudaMemcpy(resh, resd, d_p * d_nClusters * sizeof(float),
				cudaMemcpyDeviceToHost);

		for (int p = 0; p < d_p; p++) {
			vector<pair<float, uint> > distCode;
			distCode.resize(d_nClusters);

			for (int i = 0; i < d_nClusters; i++) {
				float val = resh[i * d_p + p];
				distCode[i] = pair<float, uint>(val, i);
			}

			sort(distCode.begin(), distCode.end());

			for (int i = 0; i < 8; i++)
				cout << distCode[i].second << "/" << distCode[i].first << "   ";
			cout << endl;
		}

//		getKBestAssignment2(assignVal, assignIdx, d_codeBook2,
//				d_dbVec + ddd[i].second * d_dim, d_nClusters2, 1, assignd,
//				d_nClusters, k1);

		outputVecUint("assign: ", assignd, d_p);
		outputVecUint("", assignd + d_p, d_p);
		outputVecUint("", assignd + 2 * d_p, d_p);
		outputVecUint("", assignd + 3 * d_p, d_p);
//			outputVec("", assignVal, k1* d_nClusters);
//		for (int k = 0; k < 8; k++)
//			outputVecUint("", assignIdx + (k * k1 * d_nClusters2),
//					k1 * d_nClusters2);
	}

#endif

	// cudaFree(bestIdx);
	// cudaFree(bestDist);

	// cudaFree(bins);
	// cudaFree(nBins);
	cudaFree(assignIdx);
	cudaFree(assignVal);
	cudaFree(assignd);

//	delete[] val;
//	delete[] idx;
}

}
/* namespace */

#endif /* NEARESTNEIGHBOR_PROTREE_C */
