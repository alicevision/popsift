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

#include <npp.h>

#include "s_pyramid.hpp"
#include "keep_time.hpp"
#include "debug_macros.hpp"
#include "align_macro.hpp"
#include "clamp.hpp"

#define PYRAMID_PRINT_DEBUG 0

#define PYRAMID_V6_ON false
#define PYRAMID_V7_ON true
#define EXTREMA_V4_ON true
#define ORIENTA_V1_ON false
#define ORIENTA_V2_ON true

using namespace std;

namespace popart {

__device__ __constant__ float d_gauss_filter[32];

#include "s_pyramid.v6.h"
#include "s_pyramid.v7.h"

#include "s_ori.v1.h"
#include "s_ori.v2.h"
#include "s_extrema.v4.h"

// #include "s_desc.v1.h"

/*************************************************************
 * CUDA device functions for printing debug information
 *************************************************************/
__global__
void print_gauss_filter_symbol( uint32_t columns )
{
    printf("Entering print_gauss_filter_symbol\n");
    for( uint32_t x=0; x<columns; x++ ) {
        printf("%0.3f ", d_gauss_filter[x] );
    }
    printf("\n");
}

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

    test_last_error( __LINE__, _stream );
}

void Pyramid::debug_out_floats_t( float* data, uint32_t pitch, uint32_t height )
{
    py_print_corner_float_transposed
        <<<1,1>>>
        ( data,
          pitch,
          height,
          0 );

    test_last_error( __LINE__, _stream );
}

/*************************************************************
 * Host-sided debug function
 *************************************************************/

void Pyramid::test_last_error( int line, cudaStream_t stream )
{
    cudaError_t err;
    cudaStreamSynchronize( stream );
    err = cudaGetLastError();
    if( err != cudaSuccess ) {
        printf("A problem in line %d, %s\n", line, cudaGetErrorString(err) );
        exit( -__LINE__ );
    }
}

/*************************************************************
 * Pyramid::Layer
 *************************************************************/

Pyramid::Layer::Layer( )
    : _data(0)
    , _data_2(0)
    , _t_data(0)
    , _dog_data(0)
    , _h_extrema_mgmt(0)
    , _d_extrema_mgmt(0)
    , _d_extrema(0)
    , _d_desc(0)
{ }

void Pyramid::Layer::allocExtrema( uint32_t layer_max_extrema )
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
                           _stream,
                           true );

    _d_extrema[0] = 0;
    _d_extrema[_levels-1] = 0;
    for( uint32_t i=1; i<_levels-1; i++ ) {
        ExtremumCandidate* cand;
        POP_CUDA_MALLOC( &cand, sizeof(ExtremumCandidate)*_h_extrema_mgmt[i].max2 );
        _d_extrema[i] = cand;
    }
}

void Pyramid::Layer::freeExtrema( )
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

void Pyramid::Layer::alloc( uint32_t width, uint32_t height, uint32_t levels, uint32_t layer_max_extrema, cudaStream_t stream )
{
    _pitch   = _width  = width;
    _t_pitch = _height = height;
    _levels            = levels;
    if( stream != 0 ) {
        _stream = stream;
    } else {
        POP_CUDA_STREAM_CREATE( &_stream );
    }

    _d_desc = new Descriptor*[_levels];
    _h_desc = new Descriptor*[_levels];
    memset( _d_desc, 0, _levels*sizeof(void*) ); // dynamic size, alloc later
    memset( _h_desc, 0, _levels*sizeof(void*) ); // dynamic size, alloc later

    align( _pitch,   128 ); // aligning both width
    align( _t_pitch, 128 ); // and height because we allocate transposed data
                           // Note: 128 may be excessive
#if (PYRAMID_PRINT_DEBUG==1)
    printf("    correcting to width %u, height %u\n", _width, _height );
#endif // (PYRAMID_PRINT_DEBUG==1)

    cudaError_t err;
    size_t p;

    err = cudaMallocPitch( &_data,   &p,   _pitch*sizeof(float), _levels * _t_pitch );
    POP_CUDA_FATAL_TEST( err, "cudaMallocPitch failed for array: " );
    _pitch = uint32_t(p/sizeof(float));
    assert( _pitch * sizeof(float) == p );
#if (PYRAMID_PRINT_DEBUG==1)
    printf("    data size in bytes is   %u (%u floats) * %u = %u\n",
           uint32_t(p), _pitch,
           _levels * _t_pitch,
           uint32_t(_levels * getByteSizeData() ) );
#endif // (PYRAMID_PRINT_DEBUG==1)

    err = cudaMallocPitch( &_data_2,   &p,   _pitch*sizeof(float), _levels * _t_pitch );
    POP_CUDA_FATAL_TEST( err, "cudaMallocPitch failed for array: " );
    assert( _pitch * sizeof(float) == p );

    err = cudaMallocPitch( &_dog_data, &p, _pitch*sizeof(float), (_levels-1) * _t_pitch );
    POP_CUDA_FATAL_TEST( err, "cudaMallocPitch failed for array: " );
    assert( _pitch * sizeof(float) == p );

    err = cudaMallocPitch( &_t_data, &p, _t_pitch*sizeof(float), _levels * _pitch );
    POP_CUDA_FATAL_TEST( err, "cudaMallocPitch failed for array: " );
    _t_pitch = uint32_t(p/sizeof(float));
    assert( _t_pitch * sizeof(float) == p );
#if (PYRAMID_PRINT_DEBUG==1)
    printf("    t-data size in bytes is %u (%u floats) * %u = %u\n",
           uint32_t(p), _t_pitch,
           _levels * _pitch,
           uint32_t(_levels * getByteSizeTransposedData() ) );
#endif // (PYRAMID_PRINT_DEBUG==1)

    allocExtrema( layer_max_extrema );
}

void Pyramid::Layer::free( )
{
    freeExtrema( );
    cudaFree( _t_data );
    cudaFree( _dog_data );
    cudaFree( _data_2 );
    cudaFree( _data );
    cudaStreamDestroy( _stream );
}

void Pyramid::Layer::resetExtremaCount( cudaStream_t stream )
{
    for( uint32_t i=1; i<_levels-1; i++ ) {
        _h_extrema_mgmt[i].counter = 0;
    }
    POP_CUDA_MEMCPY_ASYNC( _d_extrema_mgmt,
                           _h_extrema_mgmt,
                           _levels * sizeof(ExtremaMgmt),
                           cudaMemcpyHostToDevice, stream, true );
}

void Pyramid::Layer::readExtremaCount( cudaStream_t stream )
{
    POP_CUDA_MEMCPY_ASYNC( _h_extrema_mgmt,
                           _d_extrema_mgmt,
                           _levels * sizeof(ExtremaMgmt),
                           cudaMemcpyDeviceToHost, stream, true );
}

uint32_t Pyramid::Layer::getExtremaCount( ) const
{
    uint32_t ct = 0;
    for( uint32_t i=1; i<_levels-1; i++ ) {
        ct += _h_extrema_mgmt[i].counter;
    }
    return ct;
}

uint32_t Pyramid::Layer::getExtremaCount( uint32_t level ) const
{
    if( level < 1 )         return 0;
    if( level > _levels-2 ) return 0;
    return _h_extrema_mgmt[level].counter;
}

void Pyramid::Layer::allocDescriptors( )
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

void Pyramid::Layer::downloadDescriptor( cudaStream_t stream )
{
    for( uint32_t l=0; l<_levels; l++ ) {
        uint32_t sz = _h_extrema_mgmt[l].counter;
        if( sz != 0 ) {
            POP_CUDA_MEMCPY_ASYNC( _h_desc[l],
                                   _d_desc[l],
                                   sz * sizeof(Descriptor),
                                   cudaMemcpyDeviceToHost,
                                   stream,
                                   true );
        }
    }

    cudaStreamSynchronize( stream );
}

void Pyramid::Layer::writeDescriptor( ostream& ostr )
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

Descriptor* Pyramid::Layer::getDescriptors( uint32_t level )
{
    return _d_desc[level];
}

/*************************************************************
 * Debug output: write an octave/level to disk as PGM
 *************************************************************/

void Pyramid::download_and_save_array( const char* basename, uint32_t octave, uint32_t level )
{
    if( octave < _octaves ) {
        _layers[octave].download_and_save_array( basename, octave, level );
    } else {
        cerr << "Octave " << octave << " does not exist" << endl;
        return;
    }
}

void Pyramid::download_and_save_descriptors( const char* basename, uint32_t octave )
{
    _layers[octave].downloadDescriptor( 0 );

    struct stat st = {0};
    if (stat("dir-desc", &st) == -1) {
        mkdir("dir-desc", 0700);
    }
    ostringstream ostr;
    ostr << "dir-desc/desc-" << basename << "-o-" << octave << ".txt";
    ofstream of( ostr.str().c_str() );
    _layers[octave].writeDescriptor( of );
}

void Pyramid::Layer::download_and_save_array( const char* basename, uint32_t octave, uint32_t level )
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

        uint32_t       sz = getFloatSizeData();
        float*         f = new float        [ sz ];
        unsigned char* c = new unsigned char[ sz ];
        POP_CUDA_MEMCPY_ASYNC( f,
                               getData( level ),
                               getByteSizeData(),
                               cudaMemcpyDeviceToHost,
                               0,
                               true );
        for( uint32_t i=0; i<sz; i++ ) {
            c[i] = (unsigned char)(f[i]);
        }
        ostringstream ostr;
        ostr << "dir-octave/" << basename << "-o-" << octave << "-l-" << level << ".pgm";
        ofstream of( ostr.str().c_str() );
        of << "P5" << endl
           << _pitch << " " << _height << endl
           << "255" << endl;
        of.write( (char*)c, sz );
        of.close();

        if( level == 0 ) {
            uint32_t total_ct = 0;

            cerr << "calling from octave " << octave << endl;
            readExtremaCount( 0 );
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
                        for( int32_t j=-4; j<=4; j++ ) {
                            c[ clamp(y+j,_height)*_pitch + clamp(x,  _width) ] = 255;
                            c[ clamp(y,  _height)*_pitch + clamp(x+j,_width) ] = 255;
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
                ofstream of( ostr.str().c_str() );
                of << "P5" << endl
                   << _pitch << " " << _height << endl
                   << "255" << endl;
                of.write( (char*)c, sz );
                of.close();
            }
        }

        delete [] c;
        delete [] f;
    }
#endif
#if 0
    {
        if (stat("dir-transposed", &st) == -1) {
            mkdir("dir-transposed", 0700);
        }

        uint32_t       sz = getFloatSizeTransposedData();
        float*         f = new float        [ sz ];
        unsigned char* c = new unsigned char[ sz ];
        POP_CUDA_MEMCPY_ASYNC( f,
                               getTransposedData( level ),
                               getByteSizeTransposedData(),
                               cudaMemcpyDeviceToHost,
                               0,
                               true );
        for( uint32_t i=0; i<sz; i++ ) {
            c[i] = (unsigned char)(f[i]);
        }
        ostringstream ostr;
        ostr << "images/" << "t-" << basename << "-o-" << octave << "-l-" << level << ".pgm";
        ofstream of( ostr.str().c_str() );
        of << "P5" << endl
           << _t_pitch << " " << _width << endl
           << "255" << endl;
        of.write( (char*)c, sz );
        delete [] c;
        delete [] f;
    }
#endif
#if 1
    if (stat("dir-dog", &st) == -1) {
        mkdir("dir-dog", 0700);
    }

    if( level < _levels-1 ) {
        uint32_t       sz = getFloatSizeDogData();
        float*         f = new float        [ sz ];
        uint16_t*      c = new uint16_t[ sz ];
        POP_CUDA_MEMCPY_ASYNC( f,
                               getDogData( level ),
                               getByteSizeDogData(),
                               cudaMemcpyDeviceToHost,
                               0,
                               true );
        for( uint32_t i=0; i<sz; i++ ) {
            float fm = f[i] * 256.0;
            c[i] = htons( (uint16_t)fm );
        }
        ostringstream ostr;
        ostr << "dir-dog/" << "d-" << basename << "-o-" << octave << "-l-" << level << ".pgm";
        ofstream of( ostr.str().c_str() );
        of << "P5" << endl
           << _pitch << " " << _height << endl
           << "65535" << endl;
        of.write( (char*)c, 2*sz );
        delete [] c;
        delete [] f;
    }
#endif
}

/*************************************************************
 * Pyramid constructor
 *************************************************************/

Pyramid::Pyramid( Image* base, uint32_t octaves, uint32_t levels, cudaStream_t stream )
    : _octaves( octaves )
    , _levels( levels + 3 )
    , _stream( stream )
    , _keep_time_pyramid_v6( stream )
    , _keep_time_pyramid_v7( stream )
    , _keep_time_extrema_v4( stream )
    , _keep_time_orient_v1(  stream )
    , _keep_time_orient_v2(  stream )
    , _keep_time_descr_v1(   stream )
{
    cerr << "Entering " << __FUNCTION__ << endl;

    _layers = new Layer[octaves];

    uint32_t w = uint32_t(base->u_width / sizeof(float));
    uint32_t h = uint32_t(base->u_height);
    for( uint32_t o=0; o<_octaves; o++ ) {
#if (PYRAMID_PRINT_DEBUG==1)
        printf("Allocating octave %u with width %u and height %u (%u levels)\n", o, w, h, _levels );
#endif // (PYRAMID_PRINT_DEBUG==1)
        if( o==0 ) {
            _layers[o].alloc( w, h, _levels, 10000, _stream );
        } else {
            _layers[o].alloc( w, h, _levels, 10000 );
        }
        w = ceilf( w / 2.0f );
        h = ceilf( h / 2.0f );
    }
}

/*************************************************************
 * Pyramid destructor
 *************************************************************/

Pyramid::~Pyramid( )
{
    delete [] _layers;
}

/*************************************************************
 * Initialize the Gauss filter table in constant memory
 *************************************************************/

void Pyramid::init_filter( float sigma0, uint32_t levels, cudaStream_t stream )
{
    cerr << "Entering " << __FUNCTION__ << endl;
    if( sigma0 > 2.0 )
    {
        cerr << __FILE__ << ":" << __LINE__ << ", ERROR: "
             << " Sigma > 2.0 is not supported. Re-size __constant__ array and recompile."
             << endl;
        exit( -__LINE__ );
    }
    if( levels > 12 )
    {
        cerr << __FILE__ << ":" << __LINE__ << ", ERROR: "
             << " More than 12 levels not supported. Re-size __constant__ array and recompile."
             << endl;
        exit( -__LINE__ );
    }

    float local_filter[32];
    // const int W = GAUSS_SPAN; // no filter wider than 25; 32 is just for alignment
    // assert( W % 2 == 1 ); // filters should be symmetric, i.e. odd-sized
    // const double mean = GAUSS_ONE_SIDE_RANGE; // is always (GAUSS_SPAN-1)/2

    float sigma = sigma0;
    double sum = 0.0;
    for (int x = 0; x < GAUSS_SPAN; ++x) {
            /* Should be:
             * kernel[x] = exp( -0.5 * (pow((x-mean)/sigma, 2.0) ) )
             *           / sqrt(2 * M_PI * sigma * sigma);
             _w /= 2;
             _h /= 2;
             * but the denominator is constant and we divide by sum anyway
             */
        local_filter[x] = exp( -0.5 * (pow( double(x-GAUSS_ONE_SIDE_RANGE)/sigma, 2.0) ) );
        sum += local_filter[x];
    }

    for (int x = 0; x < GAUSS_SPAN; ++x) 
        local_filter[x] /= sum;

    cudaError_t err;
    err = cudaMemcpyToSymbolAsync( d_gauss_filter,
                                   local_filter,
                                   32*sizeof(float),
                                   0,
                                   cudaMemcpyHostToDevice,
                                   stream );
    POP_CUDA_FATAL_TEST( err, "cudaMemcpyToSymbol failed for Gauss kernel initialization: " );

    if( false ) {
        print_gauss_filter_symbol
            <<<1,1,0,stream>>>
            ( GAUSS_SPAN );
        err = cudaGetLastError();
        POP_CUDA_FATAL_TEST( err, "print_gauss_filter_symbol failed: " );
    }
}

/*************************************************************
 * Build the pyramid in all levels, one octave
 *************************************************************/

void Pyramid::build( Image* base, uint32_t idx )
{
    if( PYRAMID_V6_ON ) {
        #if (PYRAMID_PRINT_DEBUG==1)
        printf("V6: Grouping %dx1x1 read to shared, one octaves, one levels\n", V6_WIDTH);
        #endif // (PYRAMID_PRINT_DEBUG==1)
        build_v6( base );
    }

    if( PYRAMID_V7_ON ) {
        #if (PYRAMID_PRINT_DEBUG==1)
        printf("V7: Grouping %dx1x1 read directly, one octaves, one levels\n", V7_WIDTH);
        #endif // (PYRAMID_PRINT_DEBUG==1)
        build_v7( base );
    }

#if 0
    if( true  ) {
        #if (PYRAMID_PRINT_DEBUG==1)
        printf("V9: Grouping %dx1x1 read float4s, one octaves, one levels\n", V9_WIDTH);
        #endif // (PYRAMID_PRINT_DEBUG==1)
        build_v9( base );
    }
#endif
}

void Pyramid::report_times( )
{
    cudaDeviceSynchronize();
    if( PYRAMID_V6_ON ) _keep_time_pyramid_v6.report("    V6, time for building pyramid: " );
    if( PYRAMID_V7_ON ) _keep_time_pyramid_v7.report("    V7, time for building pyramid: " );
    // _keep_time_extrema_v3.report("    V3, time for finding extrema: " );
    _keep_time_extrema_v4.report("    V4, time for finding extrema: " );
    if( ORIENTA_V1_ON ) _keep_time_orient_v1. report("    V1, time for finding orientation: " );
    if( ORIENTA_V2_ON ) _keep_time_orient_v2. report("    V2, time for finding orientation: " );
    _keep_time_descr_v1.report("    V1, time for computing descriptors: " );

    for( int o=0; o<_octaves; o++ ) {
        cout << "Extrema for Octave " << o << ": ";
        for( int l=1; l<_levels-1; l++ ) {
            cout << setw(3) << _layers[o].getExtremaCount( l ) << " ";
        }
        cout << "-> " << setw(4) << _layers[o].getExtremaCount( ) << endl;
    }
}

void Pyramid::reset_extremum_counter( )
{
    for( int o=0; o<_octaves; o++ ) {
        _layers[o].resetExtremaCount( _stream );
    }
}

void Pyramid::find_extrema( float edgeLimit, float threshold )
{
    cerr << "enter " << __FUNCTION__ << endl;
    reset_extremum_counter();

    // find_extrema_v3( 2 );
    find_extrema_v4( 2, edgeLimit, threshold );

    for( int o=0; o<_octaves; o++ ) {
        _layers[o].readExtremaCount( _stream );
    }

    if( ORIENTA_V1_ON ) { orientation_v1( ); }
    if( ORIENTA_V2_ON ) { orientation_v2( ); }

    descriptors_v1( );
    cerr << "leave " << __FUNCTION__ << endl;
}

} // namespace popart

