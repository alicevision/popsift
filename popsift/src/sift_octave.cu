#include <sstream>
#include <sys/stat.h>
#include <new> // for placement new

#include "sift_pyramid.h"
#include "debug_macros.h"
#include "clamp.h"
#include "write_plane_2d.h"
#include "sift_octave.h"

// #define PYRAMID_PRINT_DEBUG 0

using namespace std;

namespace popart {

/*************************************************************
 * Octave
 *************************************************************/

Octave::Octave( )
    : _data(0)
    , _h_extrema_mgmt(0)
    , _d_extrema_mgmt(0)
    , _h_extrema(0)
    , _d_extrema(0)
#if defined(PREALLOC_DESC) && defined(USE_DYNAMIC_PARALLELISM)
    , _d_desc_pre(0)
    , _h_desc_pre(0)
#else
    , _d_desc(0)
#endif
{ }


void Octave::alloc( int width, int height, int levels, int layer_max_extrema )
{
    cudaError_t err;

    _w      = width;
    _h      = height;
    _levels = levels;

#if (PYRAMID_PRINT_DEBUG==1)
    printf("    correcting to width %u, height %u\n", _width, _height );
#endif // (PYRAMID_PRINT_DEBUG==1)

    _data = new Plane2D_float[_levels];

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

    _intermediate_data.allocDev( width, height );

    _dog_3d_desc.f = cudaChannelFormatKindFloat;
    _dog_3d_desc.x = 32;
    _dog_3d_desc.y = 0;
    _dog_3d_desc.z = 0;
    _dog_3d_desc.w = 0;

    _dog_3d_ext.width  = width; // for cudaMalloc3DArray, width in elements
    _dog_3d_ext.height = height;
    _dog_3d_ext.depth  = _levels - 1;

    err = cudaMalloc3DArray( &_dog_3d,
                             &_dog_3d_desc,
                             _dog_3d_ext,
                             cudaArrayLayered | cudaArraySurfaceLoadStore );
    POP_CUDA_FATAL_TEST( err, "Could not allocate 3D DoG array: " );

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

    _streams = new cudaStream_t[_levels];
    _gauss_done = new cudaEvent_t[_levels];
    for( int i=0; i<_levels; i++ ) {
        _streams[i]    = popart::cuda::stream_create( __FILE__, __LINE__ );
        _gauss_done[i] = popart::cuda::event_create( __FILE__, __LINE__ );
    }

    _data_tex = new cudaTextureObject_t[_levels];

    cudaTextureDesc      data_tex_desc;
    cudaResourceDesc     data_res_desc;

    memset( &data_tex_desc, 0, sizeof(cudaTextureDesc) );
    data_tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
    data_tex_desc.addressMode[0]   = cudaAddressModeClamp;
    data_tex_desc.addressMode[1]   = cudaAddressModeClamp;
    data_tex_desc.addressMode[2]   = cudaAddressModeClamp;
    data_tex_desc.readMode         = cudaReadModeElementType; // read as float
    // data_tex_desc.filterMode       = cudaFilterModePoint; // no interpolation
    data_tex_desc.filterMode       = cudaFilterModeLinear; // bilinear interpolation

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

    _d_extrema = new Extremum*[ _levels ];
    _h_extrema = new Extremum*[ _levels ];

    _h_extrema_mgmt = popart::cuda::malloc_hstT<ExtremaMgmt>( _levels, __FILE__, __LINE__ );
    _h_extrema_mgmt[0].init( 0 );
    _h_extrema_mgmt[_levels-1].init( 0 );
    for( uint32_t i=1; i<_levels-1; i++ ) {
        _h_extrema_mgmt[i].init( layer_max_extrema );
    }

    _d_extrema_mgmt = popart::cuda::malloc_devT<ExtremaMgmt>( _levels, __FILE__, __LINE__ );
    popcuda_memcpy_sync( _d_extrema_mgmt,
                         _h_extrema_mgmt,
                         _levels * sizeof(ExtremaMgmt),
                         cudaMemcpyHostToDevice );

    _h_extrema[0] = 0;
    _h_extrema[_levels-1] = 0;
    _d_extrema[0] = 0;
    _d_extrema[_levels-1] = 0;
    for( uint32_t i=1; i<_levels-1; i++ ) {
        _d_extrema[i] = popart::cuda::malloc_devT<Extremum>( _h_extrema_mgmt[i].getOrientationMax(), __FILE__, __LINE__ );
        _h_extrema[i] = popart::cuda::malloc_hstT<Extremum>( _h_extrema_mgmt[i].getOrientationMax(), __FILE__, __LINE__ );
    }

#if defined(PREALLOC_DESC) && defined(USE_DYNAMIC_PARALLELISM)
    _d_desc_pre = new Descriptor*[_levels];
    _h_desc_pre = new Descriptor*[_levels];

    _max_desc_pre = _h_extrema_mgmt[1].getOrientationMax(); // 1.25 * layer_max_extrema

    for( uint32_t l=0; l<_levels; l++ ) {
        uint32_t sz = _h_extrema_mgmt[l].getOrientationMax();
        if( sz == 0 ) {
            _d_desc_pre[l] = 0;
            _h_desc_pre[l] = 0;
        } else {
            _d_desc_pre[l] = popart::cuda::malloc_devT<Descriptor>( sz, __FILE__, __LINE__ );
            _h_desc_pre[l] = popart::cuda::malloc_hstT<Descriptor>( sz, __FILE__, __LINE__ );
        }
    }
#else
    _d_desc = new Descriptor*[_levels];
    _h_desc = new Descriptor*[_levels];
    memset( _d_desc, 0, _levels*sizeof(void*) ); // dynamic size, alloc later
    memset( _h_desc, 0, _levels*sizeof(void*) ); // dynamic size, alloc later
#endif
}

void Octave::free( )
{
    cudaError_t err;

    for( int i=0; i<_levels; i++ ) {
#if defined(PREALLOC_DESC) && defined(USE_DYNAMIC_PARALLELISM)
        if( _h_desc_pre && _h_desc_pre[i] ) cudaFreeHost( _h_desc_pre[i] );
        if( _d_desc_pre && _d_desc_pre[i] ) cudaFree(     _d_desc_pre[i] );
#else
        if( _h_desc    && _h_desc[i] )    cudaFreeHost( _h_desc[i] );
        if( _d_desc    && _d_desc[i] )    cudaFree( _d_desc[i] );
#endif
        if( _h_extrema && _h_extrema[i] ) cudaFreeHost( _h_extrema[i] );
        if( _d_extrema && _d_extrema[i] ) cudaFree( _d_extrema[i] );
    }
    cudaFree( _d_extrema_mgmt );
    cudaFreeHost( _h_extrema_mgmt );
    delete [] _d_extrema;
    delete [] _h_extrema;
#if defined(PREALLOC_DESC) && defined(USE_DYNAMIC_PARALLELISM)
    delete [] _d_desc_pre;
    delete [] _h_desc_pre;
#else
    delete [] _d_desc;
    delete [] _h_desc;
#endif

    err = cudaDestroyTextureObject( _interm_data_tex );
    POP_CUDA_FATAL_TEST( err, "Could not destroy texture object: " );
    for( int i=0; i<_levels; i++ ) {
        err = cudaDestroyTextureObject( _data_tex[i] );
        POP_CUDA_FATAL_TEST( err, "Could not destroy texture object: " );
    }

    delete [] _data_tex;

    for( int i=0; i<_levels; i++ ) {
        popart::cuda::stream_destroy( _streams[i], __FILE__, __LINE__ );
        popart::cuda::event_destroy( _gauss_done[i], __FILE__, __LINE__ );
    }
    delete [] _streams;
    delete [] _gauss_done;

    _intermediate_data.freeDev( );
    POP_CUDA_FREE( _data[0].data );

    err = cudaDestroyTextureObject( _dog_3d_tex );
    POP_CUDA_FATAL_TEST( err, "Could not destroy DoG texture: " );

    err = cudaDestroySurfaceObject( _dog_3d_surf );
    POP_CUDA_FATAL_TEST( err, "Could not destroy DoG surface: " );

    err = cudaFreeArray( _dog_3d );
    POP_CUDA_FATAL_TEST( err, "Could not free 3D DoG array: " );

    delete [] _data;
}

#if 0
void Octave::resetExtremaCount( )
{
    for( uint32_t i=1; i<_levels-1; i++ ) {
        _h_extrema_mgmt[i].resetCounter();
    }
    POP_CUDA_MEMCPY_ASYNC( _d_extrema_mgmt,
                           _h_extrema_mgmt,
                           _levels * sizeof(ExtremaMgmt),
                           cudaMemcpyHostToDevice,
                           0,
                           true );
}
#endif

void Octave::readExtremaCount( )
{
    assert( _h_extrema_mgmt );
    assert( _d_extrema_mgmt );
    popcuda_memcpy_async( _h_extrema_mgmt,
                          _d_extrema_mgmt,
                          _levels * sizeof(ExtremaMgmt),
                          cudaMemcpyDeviceToHost,
                          _streams[0] );
}

int Octave::getExtremaCount( ) const
{
    int ct = 0;
    for( uint32_t i=1; i<_levels-1; i++ ) {
        ct += _h_extrema_mgmt[i].getCounter();
    }
    return ct;
}

int Octave::getExtremaCount( uint32_t level ) const
{
    if( level < 1 )         return 0;
    if( level > _levels-2 ) return 0;
    return _h_extrema_mgmt[level].getCounter();
}

#if defined(PREALLOC_DESC) && defined(USE_DYNAMIC_PARALLELISM)
#else
void Octave::allocDescriptors( )
{
    for( uint32_t l=0; l<_levels; l++ ) {
        int sz = _h_extrema_mgmt[l].getCounter();
        if( sz == 0 ) {
            _d_desc[l] = 0;
            _h_desc[l] = 0;
        } else {
            POP_CUDA_MALLOC(      &_d_desc[l], sz * sizeof(Descriptor) );
            POP_CUDA_MALLOC_HOST( &_h_desc[l], sz * sizeof(Descriptor) );
        }
    }
}
#endif

void Octave::downloadDescriptor( )
{
    for( uint32_t l=0; l<_levels; l++ ) {
        int sz = _h_extrema_mgmt[l].getCounter();
        if( sz != 0 ) {
            if( _h_extrema[l] == 0 ) continue;

#if defined(PREALLOC_DESC) && defined(USE_DYNAMIC_PARALLELISM)
            popcuda_memcpy_async( _h_desc_pre[l],
                                  _d_desc_pre[l],
                                  sz * sizeof(Descriptor),
                                  cudaMemcpyDeviceToHost,
                                  0 );
#else
            popcuda_memcpy_async( _h_desc[l],
                                  _d_desc[l],
                                  sz * sizeof(Descriptor),
                                  cudaMemcpyDeviceToHost,
                                  0 );
#endif
            popcuda_memcpy_async( _h_extrema[l],
                                  _d_extrema[l],
                                  sz * sizeof(Extremum),
                                  cudaMemcpyDeviceToHost,
                                  0 );
        }
    }

    cudaDeviceSynchronize( );
}

void Octave::writeDescriptor( ostream& ostr, float downsampling_factor )
{
    for( uint32_t l=0; l<_levels; l++ ) {
        if( _h_extrema[l] == 0 ) continue;

        Extremum* cand = _h_extrema[l];

#if defined(PREALLOC_DESC) && defined(USE_DYNAMIC_PARALLELISM)
        Descriptor* desc = _h_desc_pre[l];
#else
        Descriptor* desc = _h_desc[l];
#endif
        int sz = _h_extrema_mgmt[l].getCounter();
        for( int s=0; s<sz; s++ ) {
            const float reduce = downsampling_factor;

            ostr << setprecision(5)
                 << ( cand[s].xpos - 0.0f ) * pow( 2.0, _debug_octave_id + reduce ) << " "
                 << ( cand[s].ypos - 0.0f ) * pow( 2.0, _debug_octave_id + reduce ) << " "
                 << cand[s].sigma * pow( 2.0, _debug_octave_id + reduce ) << " "
                 << cand[s].orientation << " ";
            for( int i=0; i<128; i++ ) {
                ostr << setprecision(3) << desc[s].features[i] << " ";
            }
            ostr << endl;
        }
    }
}

Descriptor* Octave::getDescriptors( uint32_t level )
{
#if defined(PREALLOC_DESC) && defined(USE_DYNAMIC_PARALLELISM)
    return _d_desc_pre[level];
#else
    return _d_desc[level];
#endif
}

/*************************************************************
 * Debug output: write an octave/level to disk as PGM
 *************************************************************/

void Octave::download_and_save_array( const char* basename, uint32_t octave, uint32_t level )
{
    // cerr << "Calling " << __FUNCTION__ << " for octave " << octave << endl;

    if( level >= _levels ) {
        // cerr << "Level " << level << " does not exist in Octave " << octave << endl;
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
        // cerr << "Writing " << ostr.str() << endl;
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

                    Extremum* cand = new Extremum[ct];

                    popcuda_memcpy_sync( cand,
                                         _d_extrema[l],
                                         ct * sizeof(Extremum),
                                         cudaMemcpyDeviceToHost );
                    for( uint32_t i=0; i<ct; i++ ) {
                        int32_t x = roundf( cand[i].xpos );
                        int32_t y = roundf( cand[i].ypos );
                        // cerr << "(" << x << "," << y << ") scale " << cand[i].sigma << " orient " << cand[i].orientation << endl;
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

                if (stat("dir-feat-txt", &st) == -1) {
                    mkdir("dir-feat-txt", 0700);
                }


                ostringstream ostr;
                ostr << "dir-feat/" << basename << "-o-" << octave << "-l-" << level << ".pgm";
                ostringstream ostr2;
                ostr2 << "dir-feat-txt/" << basename << "-o-" << octave << "-l-" << level << ".txt";
        #if 0
                ofstream of( ostr.str().c_str() );
                // cerr << "Writing " << ostr.str() << endl;
                of << "P5" << endl
                   << width << " " << height << endl
                   << "255" << endl;
                of.write( (char*)hostPlane_c.data, hostPlane_c.getByteSize() );
                of.close();
        #endif

                popart::write_plane2D( ostr.str().c_str(), false, hostPlane_f );
                popart::write_plane2Dunscaled( ostr2.str().c_str(), false, hostPlane_f );
            }

            hostPlane_f.freeHost( CudaAllocated );
        }
    }
#endif
#if 1
    if( level == _levels-1 ) {
        cudaError_t err;
        int width  = getData(0).getWidth();
        int height = getData(0).getHeight();

        if (stat("dir-dog", &st) == -1) {
            mkdir("dir-dog", 0700);
        }

        if (stat("dir-dog-txt", &st) == -1) {
            mkdir("dir-dog-txt", 0700);
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
            // cerr << "Writing " << ostr.str() << endl;
            popart::write_plane2D( ostr.str().c_str(), true, p );

            ostringstream ostr2;
            ostr2 << "dir-dog-txt/d-" << basename << "-o-" << octave << "-l-" << l << ".txt";
            popart::write_plane2Dunscaled( ostr2.str().c_str(), true, p );
        }

        POP_CUDA_FREE_HOST( array );
    }
#endif
}

} // namespace popart

