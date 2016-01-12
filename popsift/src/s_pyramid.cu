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

#include "s_pyramid.h"
#include "keep_time.h"
#include "debug_macros.h"
#include "align_macro.h"
#include "clamp.h"
#include "gauss_filter.h"

#define PYRAMID_PRINT_DEBUG 0

#define PYRAMID_V6_ON true
#define PYRAMID_V7_ON true
#define PYRAMID_V8_ON true

#define EXTREMA_V4_ON true
#define ORIENTA_V1_ON false
#define ORIENTA_V2_ON true

using namespace std;

namespace popart {

#include "s_ori.v1.h"
#include "s_ori.v2.h"
#include "s_extrema.v4.h"

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
 * Pyramid::Octave
 *************************************************************/

Pyramid::Octave::Octave( )
    : _data(0)
    , _t_data(0)
    , _dog_data(0)
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

void Pyramid::Octave::alloc( uint32_t width, uint32_t height, uint32_t levels, uint32_t layer_max_extrema, cudaStream_t stream )
{
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

#if (PYRAMID_PRINT_DEBUG==1)
    printf("    correcting to width %u, height %u\n", _width, _height );
#endif // (PYRAMID_PRINT_DEBUG==1)

    _data     = new Plane2D_float[_levels];
    _t_data   = new Plane2D_float[_levels];
    _dog_data = new Plane2D_float[_levels-1];

    for( int i=0; i<_levels; i++ ) {
        _data[i]  .allocDev( width, height );
        _t_data[i].allocDev( height, width );
    }
    _intermediate_data.allocDev( width, height );
    for( int i=0; i<_levels-1; i++ ) {
        _dog_data[i].allocDev( width, height );
    }

    _data_tex = new cudaTextureObject_t[_levels];

    memset( &_data_tex_desc, 0, sizeof(cudaTextureDesc) );
    _data_tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
    _data_tex_desc.addressMode[0]   = cudaAddressModeClamp;
    _data_tex_desc.addressMode[1]   = cudaAddressModeClamp;
    _data_tex_desc.addressMode[2]   = cudaAddressModeClamp;
    _data_tex_desc.readMode         = cudaReadModeElementType; // read as float
    _data_tex_desc.filterMode       = cudaFilterModePoint; // no interpolation


    memset( &_data_res_desc, 0, sizeof(cudaResourceDesc) );
    _data_res_desc.resType                  = cudaResourceTypePitch2D;
    _data_res_desc.res.pitch2D.desc.f       = cudaChannelFormatKindFloat;
    _data_res_desc.res.pitch2D.desc.x       = 32;
    _data_res_desc.res.pitch2D.desc.y       = 0;
    _data_res_desc.res.pitch2D.desc.z       = 0;
    _data_res_desc.res.pitch2D.desc.w       = 0;
    for( int i=0; i<_levels; i++ ) {
        assert( _data[i].elemSize() == 4 );
        _data_res_desc.res.pitch2D.devPtr       = _data[i].data;
        _data_res_desc.res.pitch2D.pitchInBytes = _data[i].step;
        _data_res_desc.res.pitch2D.width        = _data[i].getCols();
        _data_res_desc.res.pitch2D.height       = _data[i].getRows();

        cudaError_t err;
        err = cudaCreateTextureObject( &_data_tex[i],
                                       &_data_res_desc,
                                       &_data_tex_desc, 0 );
        POP_CUDA_FATAL_TEST( err, "Could not create texture object: " );
    }

    allocExtrema( layer_max_extrema );
}

void Pyramid::Octave::free( )
{
    freeExtrema( );

    for( int i=0; i<_levels; i++ ) {
        cudaError_t err;
        err = cudaDestroyTextureObject( _data_tex[i] );
        POP_CUDA_FATAL_TEST( err, "Could not destroy texture object: " );
    }

    delete [] _data_tex;

    _intermediate_data.freeDev( );
    for( int i=0; i<_levels; i++ ) {
        _data[i]  .freeDev( );
        _t_data[i].freeDev( );
    }
    for( int i=0; i<_levels-1; i++ ) {
        _dog_data[i].freeDev( );
    }

    delete [] _data;
    delete [] _t_data;
    delete [] _dog_data;

    cudaStreamDestroy( _stream );
}

void Pyramid::Octave::resetExtremaCount( cudaStream_t stream )
{
    for( uint32_t i=1; i<_levels-1; i++ ) {
        _h_extrema_mgmt[i].counter = 0;
    }
    POP_CUDA_MEMCPY_ASYNC( _d_extrema_mgmt,
                           _h_extrema_mgmt,
                           _levels * sizeof(ExtremaMgmt),
                           cudaMemcpyHostToDevice, stream, true );
}

void Pyramid::Octave::readExtremaCount( cudaStream_t stream )
{
    POP_CUDA_MEMCPY_ASYNC( _h_extrema_mgmt,
                           _d_extrema_mgmt,
                           _levels * sizeof(ExtremaMgmt),
                           cudaMemcpyDeviceToHost, stream, true );
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

void Pyramid::Octave::downloadDescriptor( cudaStream_t stream )
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
    _octaves[octave].downloadDescriptor( 0 );

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

        Plane2D_float& devPlane( getData(level) );

        int width  = devPlane.getWidth();
        int height = devPlane.getHeight();

        Plane2D_float  hostPlane;
        Plane2D_uint8  hostPlane_c;
        hostPlane.allocHost( width, height, popart::Unaligned );
        hostPlane.memcpyFromDevice( devPlane );
        hostPlane_c.allocHost( width, height, popart::Unaligned );

        for( int y=0; y<height; y++ )
            for( int x=0; x<width; x++ )
                hostPlane_c.ptr(y)[x] = (unsigned char)( hostPlane.ptr(y)[x] );

        ostringstream ostr;
        ostr << "dir-octave/" << basename << "-o-" << octave << "-l-" << level << ".pgm";
        ofstream of( ostr.str().c_str() );
        of << "P5" << endl
           << width << " " << height << endl
           << "255" << endl;
        of.write( (char*)hostPlane_c.data, hostPlane_c.getByteSize() );
        of.close();

        if( level == 0 ) {
            uint32_t total_ct = 0;

            cerr << "calling " << __FUNCTION__ << " from octave " << octave << endl;
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
                            hostPlane_c.ptr( clamp(y+j,height) )[ clamp(x,  width) ] = 255;
                            hostPlane_c.ptr( clamp(y,  height) )[ clamp(x+j,width) ] = 255;
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
                   << width << " " << height << endl
                   << "255" << endl;
                of.write( (char*)hostPlane_c.data, hostPlane_c.getByteSize() );
                of.close();
            }
        }

        hostPlane_c.freeHost( popart::Unaligned );
        hostPlane  .freeHost( popart::Unaligned );
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
        Plane2D_float& devPlane( getDogData(level) );
        int width  = devPlane.getWidth();
        int height = devPlane.getHeight();

        Plane2D_float  f;
        Plane2D_uint16 c;
        f.allocHost( width, height, popart::Unaligned );
        c.allocHost( width, height, popart::Unaligned );
        devPlane.memcpyToHost( f );
        for( int y=0; y<height; y++ ) {
            for( int x=0; x<width; x++ ) {
                float fm = f.ptr(y)[x] * 256.0;
                c.ptr(y)[x] = htons( (uint16_t)fm );
            }
        }
        ostringstream ostr;
        ostr << "dir-dog/d-" << basename << "-o-" << octave << "-l-" << level << ".pgm";
        ofstream of( ostr.str().c_str() );
        of << "P5" << endl
           << width << " " << height << endl
           << 65536 << endl;
        of.write( (char*)c.data, 2*f.getByteSize() );
        c.freeHost( popart::Unaligned );
        f.freeHost( popart::Unaligned );
    }
#endif
}

/*************************************************************
 * Pyramid constructor
 *************************************************************/

Pyramid::Pyramid( Image* base, uint32_t octaves, uint32_t levels, cudaStream_t stream )
    : _num_octaves( octaves )
    , _levels( levels + 3 )
    , _stream( stream )
    , _keep_time_extrema_v4( stream )
    , _keep_time_orient_v1(  stream )
    , _keep_time_orient_v2(  stream )
    , _keep_time_descr_v1(   stream )
{
    cerr << "Entering " << __FUNCTION__ << endl;

    _octaves = new Octave[_num_octaves];

    uint32_t w = uint32_t(base->array.getCols());
    uint32_t h = uint32_t(base->array.getRows());
    for( uint32_t o=0; o<_num_octaves; o++ ) {
#if (PYRAMID_PRINT_DEBUG==1)
        printf("Allocating octave %u with width %u and height %u (%u levels)\n", o, w, h, _levels );
#endif // (PYRAMID_PRINT_DEBUG==1)
        if( o==0 ) {
            _octaves[o].alloc( w, h, _levels, 10000, _stream );
        } else {
            _octaves[o].alloc( w, h, _levels, 10000 );
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
    delete [] _octaves;
}

/*************************************************************
 * Build the pyramid in all levels, one octave
 *************************************************************/

void Pyramid::build( Image* base )
{
    if( PYRAMID_V8_ON ) build_v8( base );
    if( PYRAMID_V7_ON ) build_v7( base );
    if( PYRAMID_V6_ON ) build_v6( base );
}

void Pyramid::report_times( )
{
    cudaDeviceSynchronize();

    _keep_time_extrema_v4.report("    V4, time for finding extrema: " );
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
        _octaves[o].resetExtremaCount( _stream );
    }
}

void Pyramid::find_extrema( float edgeLimit, float threshold )
{
    cerr << "enter " << __FUNCTION__ << endl;
    reset_extremum_counter();

    // find_extrema_v3( 2 );
    find_extrema_v4( 2, edgeLimit, threshold );

    for( int o=0; o<_num_octaves; o++ ) {
        _octaves[o].readExtremaCount( _stream );
    }

    if( ORIENTA_V1_ON ) { orientation_v1( ); }
    if( ORIENTA_V2_ON ) { orientation_v2( ); }

    descriptors_v1( );
    cerr << "leave " << __FUNCTION__ << endl;
}

} // namespace popart

