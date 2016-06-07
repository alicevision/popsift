#include <math.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>

#define POP_CUDA_FATAL(err,s) { \
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl; \
        std::cerr << "    " << s << cudaGetErrorString(err) << std::endl; \
        exit( -__LINE__ ); \
    }
#define POP_CUDA_FATAL_TEST(err,s) if( err != cudaSuccess ) { POP_CUDA_FATAL(err,s); }


using namespace std;

#define ASIZE 3

__global__ void printme( cudaTextureObject_t tex )
{
    printf("call printme\n");
    for( float i=0; i<1.0; i+=0.01 ) {
        float offset = i+1.0;
        float res = tex1D<float>( tex, offset );
        printf( "a[%f] = %f (%f)\n", i, res, (i-res) );
        i += 0.0001;
        offset = i+1.0;
        res = tex1D<float>( tex, offset );
        printf( "a[%f] = %f (%f)\n", i, res, (i-res) );
    }
}

int main( )
{
    float in_array[ASIZE];
    in_array[0] = -0.5;
    in_array[1] = 0.5;
    in_array[2] = 1.5;

    cudaError_t err;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc( 32, 0, 0, 0, cudaChannelFormatKindFloat );
    // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc( 8, 0, 0, 0, cudaChannelFormatKindUnsigned );
    cudaArray_t array;

    err = cudaMallocArray( &array, &channelDesc, ASIZE, 1 );
    POP_CUDA_FATAL_TEST( err, "cudaMallocArray failed" );

    err = cudaMemcpyToArray( array, 0, 0, in_array, (ASIZE)*sizeof(float), cudaMemcpyHostToDevice );
    POP_CUDA_FATAL_TEST( err, "cudaMemcpyToArray failed" );

    cudaResourceDesc resDesc;
    memset( &resDesc, 0, sizeof(cudaResourceDesc ) );
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    // resDesc.resType = cudaResourceTypeLinear;
    // resDesc.res.linear.devPtr      = array;
    // resDesc.res.linear.desc        = channelDesc;
    // resDesc.res.linear.sizeInBytes = 10000*sizeof(float);

    cudaTextureDesc texDesc;
    memset( &texDesc, 0, sizeof(cudaTextureDesc) );
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    // texDesc.filterMode       = cudaFilterModePoint;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;
    // texDesc.readMode         = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;

    err = cudaCreateTextureObject ( &texObj, &resDesc, &texDesc, 0 );
    POP_CUDA_FATAL_TEST( err, "cudaCreateTextureObject failed" );

    printme<<<1,1>>>( texObj );

    cudaDeviceSynchronize();
}

