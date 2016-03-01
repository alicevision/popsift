#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <algorithm>
#include <functional>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits>

#include <npp.h>

#include "s_pyramid.h"
#include "keep_time.h"
#include "debug_macros.h"
#include "align_macro.h"
#include "clamp.h"
#include "gauss_filter.h"
#include "write_plane_2d.h"

#undef PYRAMID_SPEED_TEST
#undef EXTREMA_SPEED_TEST
#define ALLOC_BULK

#define PYRAMID_PRINT_DEBUG 0

#define PYRAMID_V7_ON  false
#define PYRAMID_V8_ON  false
#define PYRAMID_V11_ON true
#define PYRAMID_V12_ON false


#define EXTREMA_V4 false //no cub
#define EXTREMA_V5 false // with cub
#define EXTREMA_V6 true // array?

#define EXTREMA_V4_ON true
#define ORIENTA_V1_ON true
#define ORIENTA_V2_ON false

using namespace std;

namespace popart {

#include "s_ori.v1.h"
#include "s_ori.v2.h"

/*************************************************************
 * CUDA device functions for printing debug information
 *************************************************************/

__global__
void py_print_corner_float( float* img, uint32_t pitch, uint32_t height, uint32_t level )
{
    const int xbase = 0;
    const int ybase = level * height + 0;
    for( int i=0; i<10; i++ ) {
        for( int j=0; j<10; j++ ) {
            printf("%3.3f ", img[(ybase+i)*pitch+xbase+j] );
        }
        printf("\n");
    }
    printf("\n");
}

__global__
void py_print_corner_float_transposed( float* img, uint32_t pitch, uint32_t height, uint32_t level )
{
    const int xbase = 0;
    const int ybase = level * height + 0;
    for( int i=0; i<10; i++ ) {
        for( int j=0; j<10; j++ ) {
            printf("%3.3f ", img[(ybase+j)*pitch+xbase+i] );
        }
        printf("\n");
    }
    printf("\n");
}

/*************************************************************
 * Callers for CUDA device functions that print debug information
 *************************************************************/

void Pyramid::debug_out_floats( float* data, uint32_t pitch, uint32_t height )
{
    py_print_corner_float
        <<<1,1>>>
        ( data,
          pitch,
          height,
          0 );

    test_last_error( __LINE__ );
}

void Pyramid::debug_out_floats_t( float* data, uint32_t pitch, uint32_t height )
{
    py_print_corner_float_transposed
        <<<1,1>>>
        ( data,
          pitch,
          height,
          0 );

    test_last_error( __LINE__ );
}

/*************************************************************
 * Host-sided debug function
 *************************************************************/

void Pyramid::test_last_error( int line )
{
    cudaError_t err;
    cudaDeviceSynchronize( );
    err = cudaGetLastError();
    if( err != cudaSuccess ) {
        printf("A problem in line %d, %s\n", line, cudaGetErrorString(err) );
        exit( -__LINE__ );
    }
}

/*************************************************************
 * Pyramid::Octave
 *************************************************************/

Pyramid::Octave::Octave( )
    : _data(0)
#ifndef USE_DOG_ARRAY
    , _dog_data(0)
#endif // not USE_DOG_ARRAY
    , _h_extrema_mgmt(0)
    , _d_extrema_mgmt(0)
    , _d_extrema(0)
    , _d_desc(0)
{ }

void Pyramid::Octave::allocExtrema( uint32_t layer_max_extrema )
{
    ExtremaMgmt*       mgmt;

    _d_extrema = new ExtremumCandidate*[ _levels ];

    POP_CUDA_MALLOC_HOST( &mgmt, _levels * sizeof(ExtremaMgmt) );
    memset( mgmt, 0, _levels * sizeof(ExtremaMgmt) );
    _h_extrema_mgmt = mgmt;

    POP_CUDA_MALLOC( &mgmt, _levels * sizeof(ExtremaMgmt) );
    POP_CUDA_MEMSET( mgmt, 0, _levels * sizeof(ExtremaMgmt) );
    _d_extrema_mgmt = mgmt;

    _h_extrema_mgmt[0].init( 0 );
    _h_extrema_mgmt[_levels-1].init( 0 );
    for( uint32_t i=1; i<_levels-1; i++ ) {
        _h_extrema_mgmt[i].init( layer_max_extrema );
    }

    POP_CUDA_MEMCPY_ASYNC( _d_extrema_mgmt,
                           _h_extrema_mgmt,
                           _levels * sizeof(ExtremaMgmt),
                           cudaMemcpyHostToDevice,
                           0,
                           true );

    _d_extrema[0] = 0;
    _d_extrema[_levels-1] = 0;
    for( uint32_t i=1; i<_levels-1; i++ ) {
        ExtremumCandidate* cand;
        POP_CUDA_MALLOC( &cand, sizeof(ExtremumCandidate)*_h_extrema_mgmt[i].max2 );
        _d_extrema[i] = cand;
    }
}

void Pyramid::Octave::freeExtrema( )
{
    for( uint32_t i=0; i<_levels; i++ ) {
        if( _h_desc    && _h_desc[i] )    cudaFreeHost( _h_desc[i] );
        if( _d_desc    && _d_desc[i] )    cudaFree(     _d_desc[i] );
        if( _d_extrema && _d_extrema[i] ) cudaFree(     _d_extrema[i] );
    }
    cudaFree( _d_extrema_mgmt );
    cudaFreeHost( _h_extrema_mgmt );
    delete [] _d_extrema;
    delete [] _d_desc;
    delete [] _h_desc;
}

void Pyramid::Octave::alloc( uint32_t width, uint32_t height, uint32_t levels, uint32_t layer_max_extrema )
{
    cudaError_t err;

    _levels            = levels;

    _d_desc = new Descriptor*[_levels];
    _h_desc = new Descriptor*[_levels];
    memset( _d_desc, 0, _levels*sizeof(void*) ); // dynamic size, alloc later
    memset( _h_desc, 0, _levels*sizeof(void*) ); // dynamic size, alloc later

#if (PYRAMID_PRINT_DEBUG==1)
    printf("    correcting to width %u, height %u\n", _width, _height );
#endif // (PYRAMID_PRINT_DEBUG==1)

    _data     = new Plane2D_float[_levels];

#ifdef ALLOC_BULK
    void*  ptr;
    size_t pitch;

    err = cudaMallocPitch( &ptr, &pitch, width * sizeof(float), height * _levels );
    POP_CUDA_FATAL_TEST( err, "Cannot allocate data CUDA memory: " );
    for( int i=0; i<_levels; i++ ) {
        _data[i] = Plane2D_float( width,
                                  height,
                                  (float*)( (intptr_t)ptr + i*(pitch*height) ),
                                  pitch );
    }
#else // not ALLOC_BULK
    for( int i=0; i<_levels; i++ ) {
        _data[i]  .allocDev( width, height );
    }
#endif // not ALLOC_BULK
    _intermediate_data.allocDev( width, height );
#ifdef USE_DOG_ARRAY
    _dog_3d_desc.f = cudaChannelFormatKindFloat;
    _dog_3d_desc.x = 32;
    _dog_3d_desc.y = 0;
    _dog_3d_desc.z = 0;
    _dog_3d_desc.w = 0;

    _dog_3d_ext.width  = width; // for cudaMalloc3DArray, width in elements
    _dog_3d_ext.height = height;
    _dog_3d_ext.depth  = _levels - 1;

    POP_PRINT_MEM( "(before DOG 3D array)" );
    err = cudaMalloc3DArray( &_dog_3d,
                             &_dog_3d_desc,
                             _dog_3d_ext,
                             cudaArrayLayered | cudaArraySurfaceLoadStore );
    POP_CUDA_FATAL_TEST( err, "Could not allocate 3D DoG array: " );
    POP_PRINT_MEM( "(after DOG 3D array)" );

    cudaResourceDesc dog_res_desc;
    dog_res_desc.resType         = cudaResourceTypeArray;
    dog_res_desc.res.array.array = _dog_3d;

    err = cudaCreateSurfaceObject( &_dog_3d_surf, &dog_res_desc );
    POP_CUDA_FATAL_TEST( err, "Could not create DoG surface: " );

    cudaTextureDesc      dog_tex_desc;
    memset( &dog_tex_desc, 0, sizeof(cudaTextureDesc) );
    dog_tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
    dog_tex_desc.addressMode[0]   = cudaAddressModeClamp;
    dog_tex_desc.addressMode[1]   = cudaAddressModeClamp;
    dog_tex_desc.addressMode[2]   = cudaAddressModeClamp;
    dog_tex_desc.readMode         = cudaReadModeElementType; // read as float
    dog_tex_desc.filterMode       = cudaFilterModePoint; // no interpolation

    // cudaResourceView dog_tex_view;
    // memset( &dog_tex_view, 0, sizeof(cudaResourceView) );
    // dog_tex_view.format     = cudaResViewFormatFloat1;
    // dog_tex_view.width      = width;
    // dog_tex_view.height     = height;
    // dog_tex_view.depth      = 1;
    // dog_tex_view.firstLayer = 0;
    // dog_tex_view.lastLayer  = _levels - 1;

    err = cudaCreateTextureObject( &_dog_3d_tex, &dog_res_desc, &dog_tex_desc, 0 );
    POP_CUDA_FATAL_TEST( err, "Could not create DoG texture: " );

#else // not USE_DOG_ARRAY
    _dog_data = new Plane2D_float[_levels-1];

    for( int i=0; i<_levels-1; i++ ) {
        _dog_data[i].allocDev( width, height );
    }
#endif // not USE_DOG_ARRAY

    _data_tex = new cudaTextureObject_t[_levels];

    cudaTextureDesc      data_tex_desc;
    cudaResourceDesc     data_res_desc;

    memset( &data_tex_desc, 0, sizeof(cudaTextureDesc) );
    data_tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
    data_tex_desc.addressMode[0]   = cudaAddressModeClamp;
    data_tex_desc.addressMode[1]   = cudaAddressModeClamp;
    data_tex_desc.addressMode[2]   = cudaAddressModeClamp;
    data_tex_desc.readMode         = cudaReadModeElementType; // read as float
    data_tex_desc.filterMode       = cudaFilterModePoint; // no interpolation

    memset( &data_res_desc, 0, sizeof(cudaResourceDesc) );
    data_res_desc.resType                  = cudaResourceTypePitch2D;
    data_res_desc.res.pitch2D.desc.f       = cudaChannelFormatKindFloat;
    data_res_desc.res.pitch2D.desc.x       = 32;
    data_res_desc.res.pitch2D.desc.y       = 0;
    data_res_desc.res.pitch2D.desc.z       = 0;
    data_res_desc.res.pitch2D.desc.w       = 0;
    for( int i=0; i<_levels; i++ ) {
        data_res_desc.res.pitch2D.devPtr       = _data[i].data;
        data_res_desc.res.pitch2D.pitchInBytes = _data[i].step;
        data_res_desc.res.pitch2D.width        = _data[i].getCols();
        data_res_desc.res.pitch2D.height       = _data[i].getRows();

        err = cudaCreateTextureObject( &_data_tex[i],
                                       &data_res_desc,
                                       &data_tex_desc, 0 );
        POP_CUDA_FATAL_TEST( err, "Could not create texture object: " );
    }
    data_res_desc.res.pitch2D.devPtr       = _intermediate_data.data;
    data_res_desc.res.pitch2D.pitchInBytes = _intermediate_data.step;
    data_res_desc.res.pitch2D.width        = _intermediate_data.getCols();
    data_res_desc.res.pitch2D.height       = _intermediate_data.getRows();

    err = cudaCreateTextureObject( &_interm_data_tex,
                                   &data_res_desc,
                                   &data_tex_desc, 0 );
    POP_CUDA_FATAL_TEST( err, "Could not create texture object: " );

    allocExtrema( layer_max_extrema );
}

void Pyramid::Octave::free( )
{
    cudaError_t err;

    freeExtrema( );

    err = cudaDestroyTextureObject( _interm_data_tex );
    POP_CUDA_FATAL_TEST( err, "Could not destroy texture object: " );
    for( int i=0; i<_levels; i++ ) {
        err = cudaDestroyTextureObject( _data_tex[i] );
        POP_CUDA_FATAL_TEST( err, "Could not destroy texture object: " );
    }

    delete [] _data_tex;

    _intermediate_data.freeDev( );
#ifdef ALLOC_BULK
    POP_CUDA_FREE( _data[0].data );
#else // not ALLOC_BULK
    for( int i=0; i<_levels; i++ ) {
        _data[i]  .freeDev( );
    }
#endif // not ALLOC_BULK
#ifdef USE_DOG_ARRAY
    err = cudaDestroyTextureObject( _dog_3d_tex );
    POP_CUDA_FATAL_TEST( err, "Could not destroy DoG texture: " );

    err = cudaDestroySurfaceObject( _dog_3d_surf );
    POP_CUDA_FATAL_TEST( err, "Could not destroy DoG surface: " );

    err = cudaFreeArray( _dog_3d );
    POP_CUDA_FATAL_TEST( err, "Could not free 3D DoG array: " );
#else // not USE_DOG_ARRAY
    for( int i=0; i<_levels-1; i++ ) {
        _dog_data[i].freeDev( );
    }

    delete [] _dog_data;
#endif // not USE_DOG_ARRAY

    delete [] _data;
}

void Pyramid::Octave::resetExtremaCount( )
{
    for( uint32_t i=1; i<_levels-1; i++ ) {
        _h_extrema_mgmt[i].counter = 0;
    }
    POP_CUDA_MEMCPY_ASYNC( _d_extrema_mgmt,
                           _h_extrema_mgmt,
                           _levels * sizeof(ExtremaMgmt),
                           cudaMemcpyHostToDevice,
                           0,
                           true );
}

void Pyramid::Octave::readExtremaCount( )
{
    assert( _h_extrema_mgmt );
    assert( _d_extrema_mgmt );
    POP_CUDA_MEMCPY_ASYNC( _h_extrema_mgmt,
                           _d_extrema_mgmt,
                           _levels * sizeof(ExtremaMgmt),
                           cudaMemcpyDeviceToHost,
                           0,
                           true );
}

uint32_t Pyramid::Octave::getExtremaCount( ) const
{
    uint32_t ct = 0;
    for( uint32_t i=1; i<_levels-1; i++ ) {
        ct += _h_extrema_mgmt[i].counter;
    }
    return ct;
}

uint32_t Pyramid::Octave::getExtremaCount( uint32_t level ) const
{
    if( level < 1 )         return 0;
    if( level > _levels-2 ) return 0;
    return _h_extrema_mgmt[level].counter;
}

void Pyramid::Octave::allocDescriptors( )
{
    for( uint32_t l=0; l<_levels; l++ ) {
        uint32_t sz = _h_extrema_mgmt[l].counter;
        if( sz == 0 ) {
            _d_desc[l] = 0;
            _h_desc[l] = 0;
        } else {
            POP_CUDA_MALLOC(      &_d_desc[l], sz * sizeof(Descriptor) );
            POP_CUDA_MALLOC_HOST( &_h_desc[l], sz * sizeof(Descriptor) );
        }
    }
}

void Pyramid::Octave::downloadDescriptor( )
{
    for( uint32_t l=0; l<_levels; l++ ) {
        uint32_t sz = _h_extrema_mgmt[l].counter;
        if( sz != 0 ) {
            POP_CUDA_MEMCPY_ASYNC( _h_desc[l],
                                   _d_desc[l],
                                   sz * sizeof(Descriptor),
                                   cudaMemcpyDeviceToHost,
                                   0,
                                   true );
        }
    }

    cudaDeviceSynchronize( );
}

void Pyramid::Octave::writeDescriptor( ostream& ostr )
{
    for( uint32_t l=0; l<_levels; l++ ) {
        Descriptor* desc = _h_desc[l];
        uint32_t sz = _h_extrema_mgmt[l].counter;
        for( int s=0; s<sz; s++ ) {
            ostr << "(";
            for( int i=0; i<128; i++ ) {
                ostr << setprecision(3) << desc[s].features[i] << " ";
                if( i % 16 == 15 ) ostr << endl;
            }
            ostr << ")" << endl;
        }
    }
}

Descriptor* Pyramid::Octave::getDescriptors( uint32_t level )
{
    return _d_desc[level];
}

/*************************************************************
 * Debug output: write an octave/level to disk as PGM
 *************************************************************/

void Pyramid::download_and_save_array( const char* basename, uint32_t octave, uint32_t level )
{
    if( octave < _num_octaves ) {
        _octaves[octave].download_and_save_array( basename, octave, level );
    } else {
        cerr << "Octave " << octave << " does not exist" << endl;
        return;
    }
}

void Pyramid::download_and_save_descriptors( const char* basename, uint32_t octave )
{
    _octaves[octave].downloadDescriptor( );

    struct stat st = {0};
    if (stat("dir-desc", &st) == -1) {
        mkdir("dir-desc", 0700);
    }
    ostringstream ostr;
    ostr << "dir-desc/desc-" << basename << "-o-" << octave << ".txt";
    ofstream of( ostr.str().c_str() );
    _octaves[octave].writeDescriptor( of );
}

void Pyramid::Octave::download_and_save_array( const char* basename, uint32_t octave, uint32_t level )
{
    if( level >= _levels ) {
        cerr << "Level " << level << " does not exist in Octave " << octave << endl;
        return;
    }

    struct stat st = {0};

#if 1
    {
        if (stat("dir-octave", &st) == -1) {
            mkdir("dir-octave", 0700);
        }

        ostringstream ostr;
        ostr << "dir-octave/" << basename << "-o-" << octave << "-l-" << level << ".pgm";
        cerr << "Writing " << ostr.str() << endl;
        popart::write_plane2D( ostr.str().c_str(), true, getData(level) );

        if( level == 0 ) {
            int width  = getData(level).getWidth();
            int height = getData(level).getHeight();

            Plane2D_float hostPlane_f;
            hostPlane_f.allocHost( width, height, CudaAllocated );
            hostPlane_f.memcpyFromDevice( getData(level) );

            uint32_t total_ct = 0;

            readExtremaCount( );
            cudaDeviceSynchronize( );
            for( uint32_t l=0; l<_levels; l++ ) {
                uint32_t ct = getExtremaCount( l );
                if( ct > 0 ) {
                    total_ct += ct;

                    ExtremumCandidate* cand = new ExtremumCandidate[ct];

                    POP_CUDA_MEMCPY( cand,
                                    _d_extrema[l],
                                    ct * sizeof(ExtremumCandidate),
                                    cudaMemcpyDeviceToHost );
                    for( uint32_t i=0; i<ct; i++ ) {
                        int32_t x = roundf( cand[i].xpos );
                        int32_t y = roundf( cand[i].ypos );
                        // cerr << "(" << x << "," << y << ") scale " << cand[i].sigma << " orient " << cand[i].angle_from_bemap << endl;
                        for( int32_t j=-4; j<=4; j++ ) {
                            hostPlane_f.ptr( clamp(y+j,height) )[ clamp(x,  width) ] = 255;
                            hostPlane_f.ptr( clamp(y,  height) )[ clamp(x+j,width) ] = 255;
                        }
                    }

                    delete [] cand;
                }
            }

            if( total_ct > 0 ) {
                if (stat("dir-feat", &st) == -1) {
                    mkdir("dir-feat", 0700);
                }


                ostringstream ostr;
                ostr << "dir-feat/" << basename << "-o-" << octave << "-l-" << level << ".pgm";
        #if 0
                ofstream of( ostr.str().c_str() );
                cerr << "Writing " << ostr.str() << endl;
                of << "P5" << endl
                   << width << " " << height << endl
                   << "255" << endl;
                of.write( (char*)hostPlane_c.data, hostPlane_c.getByteSize() );
                of.close();
        #endif

                popart::write_plane2D( ostr.str().c_str(), false, hostPlane_f );
            }

            hostPlane_f.freeHost( CudaAllocated );
        }
    }
#endif
#if 1
#ifdef USE_DOG_ARRAY
    if( level == _levels-1 ) {
        cudaError_t err;
        int width  = getData(0).getWidth();
        int height = getData(0).getHeight();

        if (stat("dir-dog", &st) == -1) {
            mkdir("dir-dog", 0700);
        }

        float* array;
        POP_CUDA_MALLOC_HOST( &array, width * height * (_levels-1) * sizeof(float) );

        cudaMemcpy3DParms s = { 0 };
        s.srcArray = _dog_3d;
        s.dstPtr = make_cudaPitchedPtr( array, width*sizeof(float), width, height );
        s.extent = make_cudaExtent( width, height, _levels-1 );
        s.kind = cudaMemcpyDeviceToHost;
        err = cudaMemcpy3D( &s );
        POP_CUDA_FATAL_TEST( err, "cudaMemcpy3D failed: " ); \

        for( int l=0; l<_levels-1; l++ ) {
            Plane2D_float p( width, height, &array[l*width*height], width*sizeof(float) );

            ostringstream ostr;
            ostr << "dir-dog/d-" << basename << "-o-" << octave << "-l-" << l << ".pgm";
            cerr << "Writing " << ostr.str() << endl;
            popart::write_plane2D( ostr.str().c_str(), true, p );
        }

        POP_CUDA_FREE_HOST( array );
    }
#else // not USE_DOG_ARRAY
    if (stat("dir-dog", &st) == -1) {
        mkdir("dir-dog", 0700);
    }

    if( level < _levels-1 ) {
        ostringstream ostr;
        ostr << "dir-dog/d-" << basename << "-o-" << octave << "-l-" << level << ".pgm";
        cerr << "Writing " << ostr.str() << endl;

        popart::write_plane2D( ostr.str().c_str(), true, getDogData(level) );
    }
#endif // not USE_DOG_ARRAY
#endif
}

/*************************************************************
 * Pyramid constructor
 *************************************************************/

Pyramid::Pyramid( Image* base, uint32_t octaves, uint32_t levels )
    : _num_octaves( octaves )
    , _levels( levels + 3 )
    , _keep_time_extrema_v4( 0 )
    , _keep_time_extrema_v5( 0 )
    , _keep_time_extrema_v6( 0 )
    , _keep_time_orient_v1(  0 )
    , _keep_time_orient_v2(  0 )
    , _keep_time_descr_v1(   0 )
{
    // cerr << "Entering " << __FUNCTION__ << endl;

    _octaves = new Octave[_num_octaves];

    uint32_t w = uint32_t(base->array.getCols());
    uint32_t h = uint32_t(base->array.getRows());
    for( uint32_t o=0; o<_num_octaves; o++ ) {
#if (PYRAMID_PRINT_DEBUG==1)
        printf("Allocating octave %u with width %u and height %u (%u levels)\n", o, w, h, _levels );
#endif // (PYRAMID_PRINT_DEBUG==1)
        _octaves[o].debugSetOctave( o );
        _octaves[o].alloc( w, h, _levels, 10000 );
        w = ceilf( w / 2.0f );
        h = ceilf( h / 2.0f );
    }
}

/*************************************************************
 * Pyramid destructor
 *************************************************************/

Pyramid::~Pyramid( )
{
    delete [] _octaves;
}

/*************************************************************
 * Build the pyramid in all levels, one octave
 *************************************************************/

void Pyramid::build( Image* base )
{
#ifdef PYRAMID_SPEED_TEST
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaError_t err;
    err = cudaEventCreate( &start );
    POP_CUDA_FATAL_TEST( err, "event create failed: " );
    err = cudaEventCreate( &stop );
    POP_CUDA_FATAL_TEST( err, "event create failed: " );

    for( int mode=0; mode<4; mode++ ) {
        float duration = 0.0f;
        for( int loop=0; loop<10; loop++ ) {
            err = cudaEventRecord( start, 0 );
            POP_CUDA_FATAL_TEST( err, "event record failed: " );
            switch( mode ) {
            case 0 :
                build_v7( base );
                POP_CHK;
                break;
            case 1 :
                build_v8( base );
                POP_CHK;
                break;
            case 2 :
                build_v11( base );
                POP_CHK;
                break;
            case 3 :
                build_v12( base );
                POP_CHK;
                break;
            }
            err = cudaEventRecord( stop, 0 );
            POP_CUDA_FATAL_TEST( err, "event record failed: " );
            err = cudaStreamSynchronize( 0 );
            POP_CUDA_FATAL_TEST( err, "stream sync failed: " );
            float diff;
            err = cudaEventElapsedTime( &diff, start, stop );
            POP_CUDA_FATAL_TEST( err, "elapsed time failed: " );
            duration += diff;
        }
        duration /= 10.0f;
        cerr << "Pyramid "
             << ( (mode==0) ? "V7" :
                  (mode==1) ? "V8" :
                  (mode==2) ? "V11" : "V12" )
             << " avg duration: " << duration << " ms" << endl;
    }

    err = cudaEventDestroy( start );
    POP_CUDA_FATAL_TEST( err, "event destroy failed: " );
    err = cudaEventDestroy( stop );
    POP_CUDA_FATAL_TEST( err, "event destroy failed: " );
#else // not PYRAMID_SPEED_TEST
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaError_t err;
    err = cudaEventCreate( &start );
    POP_CUDA_FATAL_TEST( err, "event create failed: " );
    err = cudaEventCreate( &stop );
    POP_CUDA_FATAL_TEST( err, "event create failed: " );

    err = cudaEventRecord( start, 0 );
    POP_CUDA_FATAL_TEST( err, "event record failed: " );

    if( PYRAMID_V7_ON  ) build_v7( base );
    if( PYRAMID_V8_ON  ) build_v8( base );
    if( PYRAMID_V11_ON ) build_v11( base );
    if( PYRAMID_V12_ON ) build_v12( base );

    err = cudaEventRecord( stop, 0 );
    POP_CUDA_FATAL_TEST( err, "event record failed: " );

    err = cudaStreamSynchronize( 0 );
    POP_CUDA_FATAL_TEST( err, "stream sync failed: " );
    float diff;
    err = cudaEventElapsedTime( &diff, start, stop );
    POP_CUDA_FATAL_TEST( err, "elapsed time failed: " );

    cerr << "Pyramid duration: " << diff << " ms" << endl;
    POP_CHK;
#endif // not PYRAMID_SPEED_TEST
}

void Pyramid::report_times( )
{
    cudaDeviceSynchronize();

    _keep_time_extrema_v4.report("    V4, time for finding extrema: " );
    _keep_time_extrema_v5.report("    V5, time for finding extrema: " );
    if( ORIENTA_V1_ON ) _keep_time_orient_v1. report("    V1, time for finding orientation: " );
    if( ORIENTA_V2_ON ) _keep_time_orient_v2. report("    V2, time for finding orientation: " );
    _keep_time_descr_v1.report("    V1, time for computing descriptors: " );

    for( int o=0; o<_num_octaves; o++ ) {
        cout << "Extrema for Octave " << o << ": ";
        for( int l=1; l<_levels-1; l++ ) {
            cout << setw(3) << _octaves[o].getExtremaCount( l ) << " ";
        }
        cout << "-> " << setw(4) << _octaves[o].getExtremaCount( ) << endl;
    }
}

void Pyramid::reset_extremum_counter( )
{
    for( int o=0; o<_num_octaves; o++ ) {
        _octaves[o].resetExtremaCount( );
    }
}

void Pyramid::find_extrema( float edgeLimit, float threshold )
{
#ifdef EXTREMA_SPEED_TEST
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaError_t err;
    err = cudaEventCreate( &start );
    POP_CUDA_FATAL_TEST( err, "event create failed: " );
    err = cudaEventCreate( &stop );
    POP_CUDA_FATAL_TEST( err, "event create failed: " );

    float duration = 0.0f;
    float min_duration = std::numeric_limits<float>::max();
    float max_duration = std::numeric_limits<float>::min();
    int   loop_len = 100;
    for( int loop=0; loop<loop_len; loop++ ) {
        err = cudaEventRecord( start, 0 );
        POP_CUDA_FATAL_TEST( err, "event record failed: " );
        reset_extremum_counter();
        find_extrema_v4( edgeLimit, threshold );
        err = cudaEventRecord( stop, 0 );
        POP_CUDA_FATAL_TEST( err, "event record failed: " );
        err = cudaStreamSynchronize( 0 );
        POP_CUDA_FATAL_TEST( err, "stream sync failed: " );
        float diff;
        err = cudaEventElapsedTime( &diff, start, stop );
        POP_CUDA_FATAL_TEST( err, "elapsed time failed: " );
        duration += diff;
        min_duration = min( min_duration, diff );
        max_duration = max( max_duration, diff );
    }
    duration /= loop_len;
    cerr << "find_extrema_v4 avg time " << duration << " ms "
         << "min " << min_duration << " ms "
         << "max " << max_duration << " ms" << endl;

    err = cudaEventDestroy( start );
    POP_CUDA_FATAL_TEST( err, "event destroy failed: " );
    err = cudaEventDestroy( stop );
    POP_CUDA_FATAL_TEST( err, "event destroy failed: " );

#else // not EXTREMA_SPEED_TEST
    if( EXTREMA_V4 ) {
        reset_extremum_counter();
        find_extrema_v4( edgeLimit, threshold );
    }

    if( EXTREMA_V5 ) {
        reset_extremum_counter();
        find_extrema_v5( edgeLimit, threshold );
    }

    if(EXTREMA_V6){
        reset_extremum_counter();
        find_extrema_v6( edgeLimit, threshold );
    }


#endif // not EXTREMA_SPEED_TEST

    for( int o=0; o<_num_octaves; o++ ) {
        _octaves[o].readExtremaCount( );
    }

    if( ORIENTA_V1_ON ) { orientation_v1( ); }
    if( ORIENTA_V2_ON ) { orientation_v2( ); }

    descriptors_v1( );
}

} // namespace popart

