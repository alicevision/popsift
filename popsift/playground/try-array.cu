#include <cuda_runtime.h>
#include <iostream>

using namespace std;

__global__
void testPrintArray( cudaTextureObject_t obj, int width, int height, int levels )
{
    for( int z=0; z<levels; z++ ) {
        for( int y=0; y<height; y++ ) {
            for( int x=0; x<width; x++ ) {
                float v = tex2DLayered<float>( obj, x, y, z );
                int d = v;
                printf( "%d ", d );
            }
            printf("\n");
        }
        printf("\n");
    }
}

int main( )
{
    int width  = 10;
    int height = 4;
    int levels = 2;

    cudaArray_t           _dog_3d;
    cudaChannelFormatDesc _dog_3d_desc;
    cudaExtent            _dog_3d_ext;
    cudaTextureObject_t   _dog_3d_tex;

    _dog_3d_desc.f = cudaChannelFormatKindFloat;
    _dog_3d_desc.x = 32;
    _dog_3d_desc.y = 0;
    _dog_3d_desc.z = 0;
    _dog_3d_desc.w = 0;

    _dog_3d_ext.width  = width; // for cudaMalloc3DArray, width in elements
    _dog_3d_ext.height = height;
    _dog_3d_ext.depth  = levels;

    cudaError_t err;
    err = cudaMalloc3DArray( &_dog_3d,
                             &_dog_3d_desc,
                             _dog_3d_ext,
                             cudaArrayLayered | cudaArraySurfaceLoadStore );
    if( err != cudaSuccess ) {
        cerr << "CUDA malloc 3D array failed: " << cudaGetErrorString(err) << endl;
        exit( -1 );
    } else {
        cerr << "CUDA malloc 3D array worked" << endl;
    }

    cudaResourceDesc dog_res_desc;
    dog_res_desc.resType         = cudaResourceTypeArray;
    dog_res_desc.res.array.array = _dog_3d;

    cudaTextureDesc      dog_tex_desc;
    memset( &dog_tex_desc, 0, sizeof(cudaTextureDesc) );
    dog_tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
    dog_tex_desc.addressMode[0]   = cudaAddressModeClamp;
    dog_tex_desc.addressMode[1]   = cudaAddressModeClamp;
    dog_tex_desc.addressMode[2]   = cudaAddressModeClamp;
    dog_tex_desc.readMode         = cudaReadModeElementType; // read as float
    dog_tex_desc.filterMode       = cudaFilterModePoint; // no interpolation

    err = cudaCreateTextureObject( &_dog_3d_tex,
                                   &dog_res_desc,
                                   &dog_tex_desc, 0 );
    if( err != cudaSuccess ) {
        cerr << "CUDA create texture failed: " << cudaGetErrorString(err) << endl;
    } else {
        cerr << "CUDA create texture succeeded" << endl;
    }

    float* array;
    err = cudaMallocHost( &array, 2*4*10*sizeof(float) );
    if( err != cudaSuccess ) {
        cerr << "CUDA malloc host failed: " << cudaGetErrorString(err) << endl;
    } else {
        cerr << "CUDA malloc host succeeded" << endl;
    }

    for( int z=0; z<2; z++ ) {
        for( int y=0; y<4; y++ ) {
            for( int x=0; x<10; x++ ) {
                array[z*4*10 + y*10 + x] = x + y + z;
                cerr << x+y+z << " ";
            }
            cerr << endl;
        }
        cerr << endl;
    }

#if 0
    err = cudaMemcpyToArray( _dog_3d,
                             0, 0,
                             array,
                             10*4*2*sizeof(float),
                             // 10*4*sizeof(float),
                             cudaMemcpyHostToDevice );
#else
    cudaMemcpy3DParms s = { 0 };
    s.srcPtr = make_cudaPitchedPtr( array, 10*sizeof(float), 10, 4 );
    s.dstArray = _dog_3d;
    s.extent = make_cudaExtent( width, height, levels );
    s.kind = cudaMemcpyHostToDevice;
    err = cudaMemcpy3D( &s );
#endif
    if( err != cudaSuccess ) {
        cerr << "CUDA memcpy 3D failed: " << cudaGetErrorString(err) << endl;
    } else {
        cerr << "CUDA memcpy 3D succeeded" << endl;
    }

    testPrintArray
        <<<1,1>>>
        ( _dog_3d_tex, width, height, levels );

    err = cudaFreeHost( array );
    if( err != cudaSuccess ) {
        cerr << "CUDA free host failed" << endl;
    } else {
        cerr << "CUDA free host success" << endl;
    }

    err = cudaDestroyTextureObject( _dog_3d_tex );
    if( err != cudaSuccess ) {
        cerr << "CUDA destroy texture failed" << endl;
    } else {
        cerr << "CUDA destroy texture success" << endl;
    }

    err = cudaFreeArray( _dog_3d );
    if( err != cudaSuccess ) {
        cerr << "CUDA array free failed" << endl;
    } else {
        cerr << "CUDA array free success" << endl;
    }
}

