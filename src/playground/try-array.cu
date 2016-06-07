#include <cuda_runtime.h>
#include <iostream>

#define WITH_SURFACE

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

#ifdef WITH_SURFACE
__global__
void writeToSurface( cudaSurfaceObject_t obj, int width, int height, int levels )
{
    for( int z=threadIdx.z; z<levels; z += blockDim.z ) {
        for( int y=threadIdx.y; y<height; y += blockDim.y ) {
            for( int x=threadIdx.x; x<width; x += blockDim.x ) {
                float v = x - 10*y + 100*z;
                surf2DLayeredwrite( v,
                                    obj,
                                    x*sizeof(float), y, z,
                                    cudaBoundaryModeZero );
                                    // x, y, z,
                                    // cudaBoundaryModeTrap );
            }
        }
    }
}
#endif // WITH_SURFACE

int main( )
{
    int width  = 10;
    int height = 4;
    int levels = 2;

    cudaArray_t           _dog_3d;
    cudaChannelFormatDesc _dog_3d_desc;
    cudaExtent            _dog_3d_ext;
    cudaTextureObject_t   _dog_3d_tex;
#ifdef WITH_SURFACE
    cudaSurfaceObject_t   _dog_3d_surf;
#endif // WITH_SURFACE

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
        exit( -1 );
    } else {
        cerr << "CUDA create texture succeeded" << endl;
    }

    // cudaResourceDesc dog_res_desc;
    // dog_res_desc.resType         = cudaResourceTypeArray;
    // dog_res_desc.res.array.array = _dog_3d;

#ifdef WITH_SURFACE
    err = cudaCreateSurfaceObject( &_dog_3d_surf, &dog_res_desc );
    if( err != cudaSuccess ) {
        cerr << "CUDA create surface failed: " << cudaGetErrorString(err) << endl;
        exit( -1 );
    } else {
        cerr << "CUDA create surface succeeded" << endl;
    }
#endif // WITH_SURFACE

    float* array;
    err = cudaMallocHost( &array, width*height*levels*sizeof(float) );
    if( err != cudaSuccess ) {
        cerr << "CUDA malloc host failed: " << cudaGetErrorString(err) << endl;
        exit( -1 );
    } else {
        cerr << "CUDA malloc host succeeded" << endl;
    }

    cerr << "== host-sided printing of array ==" << endl;
    for( int z=0; z<levels; z++ ) {
        for( int y=0; y<height; y++ ) {
            for( int x=0; x<width; x++ ) {
                int val = x + 10*y + 100*z;
                array[(z*height + y)*width + x] = val;
                cerr << val << " ";
            }
            cerr << endl;
        }
        cerr << endl;
    }

#if 0
    // This does not work
    err = cudaMemcpyToArray( _dog_3d,
                             0, 0,
                             array,
                             10*4*2*sizeof(float),
                             // 10*4*sizeof(float),
                             cudaMemcpyHostToDevice );
#endif
    cudaMemcpy3DParms s = { 0 };
    s.srcPtr = make_cudaPitchedPtr( array, 10*sizeof(float), 10, 4 );
    s.dstArray = _dog_3d;
    s.extent = make_cudaExtent( width, height, levels );
    s.kind = cudaMemcpyHostToDevice;
    err = cudaMemcpy3D( &s );

    if( err != cudaSuccess ) {
        cerr << "CUDA memcpy 3D failed: " << cudaGetErrorString(err) << endl;
        exit( -1 );
    } else {
        cerr << "CUDA memcpy 3D succeeded" << endl;
    }

    cudaDeviceSynchronize();

    cerr << "== CUDA-sided printing of array ==" << endl;

    testPrintArray
        <<<1,1>>>
        ( _dog_3d_tex, 10, 4, 2 );
        // ( _dog_3d_tex, width, height, levels );

    err = cudaGetLastError( );
    if( err != cudaSuccess ) {
        cerr << "testPrintArray: " << cudaGetErrorString(err) << endl;
        exit( -1 );
    } else {
        cerr << "testPrintArray succeeded" << endl;
    }

#ifdef WITH_SURFACE
    cerr << "== CUDA-sided writing to surface" << endl;

    dim3 block( 8, 8, 1 );
    writeToSurface
        <<<1,block>>>
        ( _dog_3d_surf, width, height, levels );

    err = cudaGetLastError( );
    if( err != cudaSuccess ) {
        cerr << "writeToSurface: " << cudaGetErrorString(err) << endl;
        exit( -1 );
    } else {
        cerr << "writeToSurface succeeded" << endl;
    }

    cerr << "== CUDA-sided printing of array ==" << endl;

    testPrintArray
        <<<1,1>>>
        ( _dog_3d_tex, width, height, levels );

    err = cudaGetLastError( );
    if( err != cudaSuccess ) {
        cerr << "testPrintArray: " << cudaGetErrorString(err) << endl;
        exit( -1 );
    } else {
        cerr << "testPrintArray succeeded" << endl;
    }
#endif // WITH_SURFACE

    cudaDeviceSynchronize();

    memset( array, 0, width*height*levels*sizeof(float) );

    memset( &s, 0, sizeof(cudaMemcpy3DParms) );
    s.srcArray = _dog_3d;
    s.dstPtr = make_cudaPitchedPtr( array, width*sizeof(float), width, height );
    s.extent = make_cudaExtent( width, height, levels );
    s.kind = cudaMemcpyDeviceToHost;
    err = cudaMemcpy3D( &s );

    if( err != cudaSuccess ) {
        cerr << "CUDA memcpy 3D failed: " << cudaGetErrorString(err) << endl;
        exit( -1 );
    } else {
        cerr << "CUDA memcpy 3D succeeded" << endl;
    }

    cerr << "== host-sided printing of array ==" << endl;
    for( int z=0; z<2; z++ ) {
        for( int y=0; y<4; y++ ) {
            for( int x=0; x<10; x++ ) {
                cerr << array[(z*height + y)*width + x] << " ";
            }
            cerr << endl;
        }
        cerr << endl;
    }

    err = cudaFreeHost( array );
    if( err != cudaSuccess ) {
        cerr << "CUDA free host failed" << endl;
        exit( -1 );
    } else {
        cerr << "CUDA free host success" << endl;
    }

#ifdef WITH_SURFACE
    err = cudaDestroySurfaceObject( _dog_3d_surf );
    if( err != cudaSuccess ) {
        cerr << "CUDA destroy surface failed" << endl;
        exit( -1 );
    } else {
        cerr << "CUDA destroy surface success" << endl;
    }
#endif // WITH_SURFACE

    err = cudaDestroyTextureObject( _dog_3d_tex );
    if( err != cudaSuccess ) {
        cerr << "CUDA destroy texture failed" << endl;
        exit( -1 );
    } else {
        cerr << "CUDA destroy texture success" << endl;
    }

    err = cudaFreeArray( _dog_3d );
    if( err != cudaSuccess ) {
        cerr << "CUDA array free failed" << endl;
        exit( -1 );
    } else {
        cerr << "CUDA array free success" << endl;
    }
}

