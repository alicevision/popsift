#include "helper.hh"

#define OUTPUT
namespace pqt {

void checkCudaErrors( cudaError_t err )
{
    if( err != cudaSuccess )
    {
        std::cerr << "CUDA call failed, " << cudaGetErrorString(err) << std::endl;
        exit( -1 );
    }
}

__global__ void outputMatKernel(const float*_A, uint _rows,
		uint _cols) {
	if (threadIdx.x == 0) {
		for (int j = 0; j < _rows; j++) {
			for (int i = 0; i < _cols; i++)
				printf("%.4f ", _A[j * _cols + i]);
			printf("\n");
		}

	}
}

void outputMat(const std::string& _S, const float* _A,
		uint _rows, uint _cols) {
#ifndef OUTPUT
	return;
#endif
	cout << _S << endl;


	dim3 grid(1, 1, 1);
	dim3 block(16, 1, 1);
	outputMatKernel<<<grid, block>>>(_A, _rows, _cols);

	checkCudaErrors(cudaDeviceSynchronize());

}

__global__ void outputVecKernel(const float* _v, uint _n) {
	if (threadIdx.x == 0) {
		for (int i = 0; i < _n; i++)
			printf("%.4f ", _v[i]);
		printf("\n");

	}
}

void outputVec(const std::string& _S, const float* _v,
		uint _n) {
#ifndef OUTPUT
	return;
#endif

	cout << _S << endl;


	dim3 grid(1, 1, 1);
	dim3 block(16, 1, 1);
	outputVecKernel<<<grid, block>>>(_v, _n);

	checkCudaErrors(cudaDeviceSynchronize());

}


__global__ void outputVecUIntKernel(const uint* _v, uint _n) {
	if (threadIdx.x == 0) {
		for (int i = 0; i < _n; i++)
			printf("%5d ", _v[i]);
		printf("\n");

	}
}

void outputVecUint(const std::string& _S, const uint* _v,
		uint _n) {
#ifndef OUTPUT
	return;
#endif

	cout << _S << endl;


	dim3 grid(1, 1, 1);
	dim3 block(16, 1, 1);
	outputVecUIntKernel<<<grid, block>>>(_v, _n);

	checkCudaErrors(cudaDeviceSynchronize());

}

__global__ void outputVecIntKernel(const int* _v, uint _n) {
	if (threadIdx.x == 0) {
		for (int i = 0; i < _n; i++)
			printf("%5d ", _v[i]);
		printf("\n");

	}
}

void outputVecInt(const std::string& _S, const int* _v,
		uint _n) {
#ifndef OUTPUT
	return;
#endif

	cout << _S << endl;


	dim3 grid(1, 1, 1);
	dim3 block(16, 1, 1);
	outputVecIntKernel<<<grid, block>>>(_v, _n);

	checkCudaErrors(cudaDeviceSynchronize());

}

void countZeros(const std::string& _S, const uint* _v,
		uint _n) {
#ifndef OUTPUT
	return;
#endif

	uint* help = new uint[_n];

	cudaMemcpy( help, _v, _n * sizeof(uint), cudaMemcpyDeviceToHost );

	uint countZero = 0;

	for (int i = 0; i < _n; i++)
		if ( help[i] == 0) countZero++;

	cout << _S << " N: " << _n << " zeros " << countZero << " non-zero: " << (_n - countZero) << endl;

	delete[] help;
}



} /* namespace */
