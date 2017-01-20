/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <sstream>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#define stat _stat
#define mkdir(path, perm) _mkdir(path)
#endif

#include <new> // for placement new

#include "sift_pyramid.h"
#include "sift_constants.h"
#include "common/debug_macros.h"
#include "common/clamp.h"
#include "common/write_plane_2d.h"
#include "sift_octave.h"

/* Define this only for debugging the descriptor by writing
 * the dominent orientation in readable form (otherwise
 * incompatible with other tools).
 */
#undef PRINT_WITH_ORIENTATION

using namespace std;

namespace popsift {

Octave::Octave( )
    : _data(0)
    , _h_extrema_counter(0)
    , _d_extrema_counter(0)
    , _h_featvec_counter(0)
    , _d_featvec_counter(0)
    , _h_extrema(0)
    , _d_extrema(0)
    , _d_desc(0)
    , _h_desc(0)
{ }


void Octave::alloc( int width, int height, int levels, int gauss_group )
{
    _w           = width;
    _h           = height;
    _levels      = levels;
    _gauss_group = gauss_group;

    alloc_data_planes( );
    alloc_data_tex( );

    alloc_interm_plane( );
    alloc_interm_tex( );

    alloc_dog_array( );
    alloc_dog_tex( );

    alloc_extrema_mgmt( );
    alloc_extrema( );

    alloc_streams( );
    alloc_events( );

    _d_desc = new Descriptor*[_levels];
    _h_desc = new Descriptor*[_levels];

    for( int l=0; l<_levels; l++ ) {
        int sz = h_consts.orientations;
        if( sz == 0 ) {
            _d_desc[l] = 0;
            _h_desc[l] = 0;
        } else {
            _d_desc[l] = popsift::cuda::malloc_devT<Descriptor>( sz, __FILE__, __LINE__ );
            _h_desc[l] = popsift::cuda::malloc_hstT<Descriptor>( sz, __FILE__, __LINE__ );
        }
    }
}

void Octave::free( )
{
    for( int i=0; i<_levels; i++ ) {
        if( _h_desc && _h_desc[i] ) cudaFreeHost( _h_desc[i] );
        if( _d_desc && _d_desc[i] ) cudaFree(     _d_desc[i] );
    }
    delete [] _d_desc;
    delete [] _h_desc;

    free_events( );
    free_streams( );

    free_extrema( );
    free_extrema_mgmt( );

    free_dog_tex( );
    free_dog_array( );

    free_interm_tex( );
    free_interm_plane( );

    free_data_tex( );
    free_data_planes( );
}

void Octave::reset_extrema_mgmt( )
{
    cudaStream_t stream = _streams[0];
    cudaEvent_t  ev     = _extrema_done[0];

    memset( _h_extrema_counter, 0, _levels * sizeof(int) );
    popcuda_memset_async( _d_extrema_counter, 0, _levels * sizeof(int), stream );
    popcuda_memset_async( _d_extrema_num_blocks, 0, _levels * sizeof(int), stream );
    popcuda_memset_async( _d_featvec_counter, 0, _levels * sizeof(int), stream );
#if 0
    popcuda_memset_async( _d_orientation_num_blocks, 0, _levels * sizeof(int), stream );
#endif

    cudaEventRecord( ev, stream );
}

void Octave::readExtremaCount( )
{
    assert( _h_extrema_counter );
    assert( _d_extrema_counter );
    assert( _h_featvec_counter );
    assert( _d_featvec_counter );
    popcuda_memcpy_async( _h_extrema_counter,
                          _d_extrema_counter,
                          _levels * sizeof(int),
                          cudaMemcpyDeviceToHost,
                          _streams[0] );
    popcuda_memcpy_async( _h_featvec_counter,
                          _d_featvec_counter,
                          _levels * sizeof(int),
                          cudaMemcpyDeviceToHost,
                          _streams[0] );
}

int Octave::getExtremaCount( ) const
{
    int ct = 0;
    for( uint32_t i=1; i<_levels-1; i++ ) {
        ct += _h_extrema_counter[i];
    }
    return ct;
}

int Octave::getDescriptorCount( ) const
{
    int ct = 0;
    for( uint32_t i=1; i<_levels-1; i++ ) {
        ct += _h_featvec_counter[i];
    }
    return ct;
}

void Octave::downloadDescriptor( const Config& conf )
{
    for( uint32_t l=0; l<_levels; l++ ) {
        int sz = _h_extrema_counter[l];
        if( sz != 0 ) {
            if( _h_extrema[l] == 0 ) continue;

            popcuda_memcpy_async( _h_extrema[l],
                                  _d_extrema[l],
                                  sz * sizeof(Extremum),
                                  cudaMemcpyDeviceToHost,
                                  0 );
        }
        sz = _h_featvec_counter[l];
        if( sz != 0 ) {
            popcuda_memcpy_async( _h_desc[l],
                                  _d_desc[l],
                                  sz * sizeof(Descriptor),
                                  cudaMemcpyDeviceToHost,
                                  0 );
        }
    }

    cudaDeviceSynchronize( );
}

void Octave::writeDescriptor( const Config& conf, ostream& ostr, bool really )
{
    for( uint32_t l=0; l<_levels; l++ ) {
        if( _h_extrema[l] == 0 ) continue;

        Extremum* cand = _h_extrema[l];

        Descriptor* desc = _h_desc[l];

        int sz = _h_extrema_counter[l];
        for( int s=0; s<sz; s++ ) {
            for( int ori=0; ori<cand[s].num_ori; ori++ ) {
                const float up_fac = conf.getUpscaleFactor();

                float xpos  = cand[s].xpos * pow( 2.0, _debug_octave_id - up_fac );
                float ypos  = cand[s].ypos * pow( 2.0, _debug_octave_id - up_fac );
                float sigma = cand[s].sigma * pow( 2.0, _debug_octave_id - up_fac );
                float dom_or = cand[s].orientation[ori];
                dom_or = dom_or / M_PI2 * 360;
                if( dom_or < 0 ) dom_or += 360;

#ifdef PRINT_WITH_ORIENTATION
                ostr << setprecision(5)
                    << xpos << " "
                    << ypos << " "
                    << sigma << " "
                    << dom_or << " ";
#else
                ostr << setprecision(5)
                    << xpos << " " << ypos << " "
                    << 1.0f / ( sigma * sigma )
                    << " 0 "
                    << 1.0f / ( sigma * sigma ) << " ";
#endif
                if( really ) {
                    int feat_vec_index = cand[s].idx_ori + ori;
                    for( int i=0; i<128; i++ ) {
                        ostr << desc[feat_vec_index].features[i] << " ";
                    }
                }
                ostr << endl;
            }
        }
    }
}

void Octave::copyExtrema( const Config& conf, Feature* feature, Descriptor* descBuffer )
{
    int num_extrema     = getExtremaCount();
    int num_descriptors = getDescriptorCount();

    for( uint32_t l=0; l<_levels; l++ ) {
        Extremum*   ext     = _h_extrema[l];
        Descriptor* desc    = _h_desc[l];
        int         ext_sz  = _h_extrema_counter[l];
        int         desc_sz = _h_featvec_counter[l];

        memcpy( descBuffer, desc, desc_sz * sizeof( Descriptor ) );
        for( int i=0; i<ext_sz; i++ ) {
            const float up_fac = conf.getUpscaleFactor();

            float xpos    = ext[i].xpos * pow( 2.0, _debug_octave_id - up_fac );
            float ypos    = ext[i].ypos * pow( 2.0, _debug_octave_id - up_fac );
            float sigma   = ext[i].sigma * pow( 2.0, _debug_octave_id - up_fac );
            int   num_ori = ext[i].num_ori;

            feature[i].xpos      = xpos;
            feature[i].ypos      = ypos;
            feature[i].sigma     = sigma;
            feature[i].num_descs = num_ori;


            int ori;
            for( ori=0; ori<num_ori; ori++ ) {
                int desc_idx = ext[i].idx_ori + ori;
                feature[i].orientation[ori] = ext[i].orientation[ori];
                feature[i].desc[ori]        = &descBuffer[desc_idx];
            }
            for( ; ori<ORIENTATION_MAX_COUNT; ori++ ) {
                feature[i].orientation[ori] = 0;
                feature[i].desc[ori] = 0;
            }
        }

        feature    += ext_sz;
        descBuffer += desc_sz;
    }

}

Descriptor* Octave::getDescriptors( uint32_t level )
{
    return _d_desc[level];
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

    {
        if (stat("dir-octave", &st) == -1) {
            mkdir("dir-octave", 0700);
        }

        ostringstream ostr;
        ostr << "dir-octave/" << basename << "-o-" << octave << "-l-" << level << ".pgm";
        popsift::write_plane2Dunscaled( ostr.str().c_str(), true, getData(level) );

        if (stat("dir-octave-dump", &st) == -1) {
            mkdir("dir-octave-dump", 0700);
        }

        ostringstream ostr2;
        ostr2 << "dir-octave-dump/" << basename << "-o-" << octave << "-l-" << level << ".dump";
        popsift::dump_plane2Dfloat( ostr2.str().c_str(), true, getData(level) );

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
                uint32_t ct = getExtremaCountH( l ); // getExtremaCount( l );
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

                popsift::write_plane2D( ostr.str().c_str(), false, hostPlane_f );
                popsift::write_plane2Dunscaled( ostr2.str().c_str(), false, hostPlane_f );
            }

            hostPlane_f.freeHost( CudaAllocated );
        }
    }

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

        if (stat("dir-dog-dump", &st) == -1) {
            mkdir("dir-dog-dump", 0700);
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
            popsift::write_plane2D( ostr.str().c_str(), true, p );

            ostringstream pstr;
            pstr << "dir-dog-txt/d-" << basename << "-o-" << octave << "-l-" << l << ".txt";
            popsift::write_plane2Dunscaled( pstr.str().c_str(), true, p, 127 );

            ostringstream qstr;
            qstr << "dir-dog-dump/d-" << basename << "-o-" << octave << "-l-" << l << ".dump";
            popsift::dump_plane2Dfloat( qstr.str().c_str(), true, p );
        }

        POP_CUDA_FREE_HOST( array );
    }
}

void Octave::alloc_data_planes( )
{
    cudaError_t err;
    void*       ptr;
    size_t      pitch;

    _data = new Plane2D_float[_levels];

    err = cudaMallocPitch( &ptr, &pitch, _w * sizeof(float), _h * _levels );
    POP_CUDA_FATAL_TEST( err, "Cannot allocate data CUDA memory: " );
    for( int i=0; i<_levels; i++ ) {
        _data[i] = Plane2D_float( _w,
                                  _h,
                                  (float*)( (intptr_t)ptr + i*(pitch*_h) ),
                                  pitch );
    }
}

void Octave::free_data_planes( )
{
    POP_CUDA_FREE( _data[0].data );
    delete [] _data;
}

void Octave::alloc_data_tex( )
{
    cudaError_t err;

    _data_tex = new cudaTextureObject_t[_levels];

    cudaTextureDesc      data_tex_desc;
    cudaResourceDesc     data_res_desc;

    memset( &data_tex_desc, 0, sizeof(cudaTextureDesc) );
    data_tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
    data_tex_desc.addressMode[0]   = cudaAddressModeClamp;
    data_tex_desc.addressMode[1]   = cudaAddressModeClamp;
    data_tex_desc.addressMode[2]   = cudaAddressModeClamp;
    data_tex_desc.readMode         = cudaReadModeElementType; // read as float
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
}

void Octave::free_data_tex( )
{
    cudaError_t err;

    for( int i=0; i<_levels; i++ ) {
        err = cudaDestroyTextureObject( _data_tex[i] );
        POP_CUDA_FATAL_TEST( err, "Could not destroy texture object: " );
    }

    delete [] _data_tex;
}

void Octave::alloc_interm_plane( )
{
    _intermediate_data.allocDev( _w, _h );
}

void Octave::free_interm_plane( )
{
    _intermediate_data.freeDev( );
}

void Octave::alloc_interm_tex( )
{
    cudaError_t err;

    cudaTextureDesc      interm_data_tex_desc;
    cudaResourceDesc     interm_data_res_desc;

    memset( &interm_data_tex_desc, 0, sizeof(cudaTextureDesc) );
    interm_data_tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
    interm_data_tex_desc.addressMode[0]   = cudaAddressModeClamp;
    interm_data_tex_desc.addressMode[1]   = cudaAddressModeClamp;
    interm_data_tex_desc.addressMode[2]   = cudaAddressModeClamp;
    interm_data_tex_desc.readMode         = cudaReadModeElementType; // read as float
    interm_data_tex_desc.filterMode       = cudaFilterModePoint;

    memset( &interm_data_res_desc, 0, sizeof(cudaResourceDesc) );
    interm_data_res_desc.resType                  = cudaResourceTypePitch2D;
    interm_data_res_desc.res.pitch2D.desc.f       = cudaChannelFormatKindFloat;
    interm_data_res_desc.res.pitch2D.desc.x       = 32;
    interm_data_res_desc.res.pitch2D.desc.y       = 0;
    interm_data_res_desc.res.pitch2D.desc.z       = 0;
    interm_data_res_desc.res.pitch2D.desc.w       = 0;

    interm_data_res_desc.res.pitch2D.devPtr       = _intermediate_data.data;
    interm_data_res_desc.res.pitch2D.pitchInBytes = _intermediate_data.step;
    interm_data_res_desc.res.pitch2D.width        = _intermediate_data.getCols();
    interm_data_res_desc.res.pitch2D.height       = _intermediate_data.getRows();

    err = cudaCreateTextureObject( &_interm_data_tex,
                                   &interm_data_res_desc,
                                   &interm_data_tex_desc, 0 );
    POP_CUDA_FATAL_TEST( err, "Could not create texture object: " );
}

void Octave::free_interm_tex( )
{
    cudaError_t err;

    err = cudaDestroyTextureObject( _interm_data_tex );
    POP_CUDA_FATAL_TEST( err, "Could not destroy texture object: " );
}

void Octave::alloc_dog_array( )
{
    cudaError_t err;

    _dog_3d_desc.f = cudaChannelFormatKindFloat;
    _dog_3d_desc.x = 32;
    _dog_3d_desc.y = 0;
    _dog_3d_desc.z = 0;
    _dog_3d_desc.w = 0;

    _dog_3d_ext.width  = _w; // for cudaMalloc3DArray, width in elements
    _dog_3d_ext.height = _h;
    _dog_3d_ext.depth  = _levels - 1;

    err = cudaMalloc3DArray( &_dog_3d,
                             &_dog_3d_desc,
                             _dog_3d_ext,
                             cudaArrayLayered | cudaArraySurfaceLoadStore );
    POP_CUDA_FATAL_TEST( err, "Could not allocate 3D DoG array: " );
}

void Octave::free_dog_array( )
{
    cudaError_t err;

    err = cudaFreeArray( _dog_3d );
    POP_CUDA_FATAL_TEST( err, "Could not free 3D DoG array: " );
}

void Octave::alloc_dog_tex( )
{
    cudaError_t err;

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

    err = cudaCreateTextureObject( &_dog_3d_tex, &dog_res_desc, &dog_tex_desc, 0 );
    POP_CUDA_FATAL_TEST( err, "Could not create DoG texture: " );
}

void Octave::free_dog_tex( )
{
    cudaError_t err;

    err = cudaDestroyTextureObject( _dog_3d_tex );
    POP_CUDA_FATAL_TEST( err, "Could not destroy DoG texture: " );

    err = cudaDestroySurfaceObject( _dog_3d_surf );
    POP_CUDA_FATAL_TEST( err, "Could not destroy DoG surface: " );
}

void Octave::alloc_extrema_mgmt( )
{
    _h_extrema_counter        = popsift::cuda::malloc_hstT<int>( _levels, __FILE__, __LINE__ );
    _d_extrema_counter        = popsift::cuda::malloc_devT<int>( _levels, __FILE__, __LINE__ );
    _d_extrema_num_blocks     = popsift::cuda::malloc_devT<int>( _levels, __FILE__, __LINE__ );
    _h_featvec_counter        = popsift::cuda::malloc_hstT<int>( _levels, __FILE__, __LINE__ );
    _d_featvec_counter        = popsift::cuda::malloc_devT<int>( _levels, __FILE__, __LINE__ );
}

void Octave::free_extrema_mgmt( )
{
    cudaFree( _d_extrema_num_blocks );
    cudaFree( _d_extrema_counter );
    cudaFreeHost( _h_extrema_counter );
    cudaFree( _d_featvec_counter );
    cudaFreeHost( _h_featvec_counter );
}

void Octave::alloc_extrema( )
{
    _d_extrema = new Extremum*[ _levels ];
    _h_extrema = new Extremum*[ _levels ];

    _h_extrema[0] = 0;
    _h_extrema[_levels-1] = 0;
    _d_extrema[0] = 0;
    _d_extrema[_levels-1] = 0;

    _d_feat_to_ext_map = new int*[ _levels ];
    _h_feat_to_ext_map = new int*[ _levels ];

    _h_feat_to_ext_map[0] = 0;
    _h_feat_to_ext_map[_levels-1] = 0;
    _d_feat_to_ext_map[0] = 0;
    _d_feat_to_ext_map[_levels-1] = 0;

    int levels            = _levels - 2;

    Extremum* d = popsift::cuda::malloc_devT<Extremum>( levels * h_consts.extrema, __FILE__, __LINE__ );
    Extremum* h = popsift::cuda::malloc_hstT<Extremum>( levels * h_consts.extrema, __FILE__, __LINE__ );

    for( uint32_t i=1; i<_levels-1; i++ ) {
        const int offset = i-1;
        _d_extrema[i] = &d[ offset * h_consts.extrema ];
        _h_extrema[i] = &h[ offset * h_consts.extrema ];
    }

    int* mapd = popsift::cuda::malloc_devT<int>( levels * h_consts.orientations, __FILE__, __LINE__ );
    int* maph = popsift::cuda::malloc_hstT<int>( levels * h_consts.orientations, __FILE__, __LINE__ );

    for( uint32_t i=1; i<_levels-1; i++ ) {
        const int offset = i-1;
        _d_feat_to_ext_map[i] = &mapd[ offset * h_consts.orientations ];
        _h_feat_to_ext_map[i] = &maph[ offset * h_consts.orientations ];
    }
}

void Octave::free_extrema( )
{
    cudaFreeHost( _h_feat_to_ext_map[1] );
    cudaFree(     _d_feat_to_ext_map[1] );
    cudaFreeHost( _h_extrema[1] );
    cudaFree(     _d_extrema[1] );
    delete [] _d_extrema;
    delete [] _h_extrema;
}

void Octave::alloc_streams( )
{
    _streams = new cudaStream_t[_levels];

    for( int i=0; i<_levels; i++ ) {
        _streams[i]    = popsift::cuda::stream_create( __FILE__, __LINE__ );
    }
}

void Octave::free_streams( )
{
    for( int i=0; i<_levels; i++ ) {
        popsift::cuda::stream_destroy( _streams[i], __FILE__, __LINE__ );
    }
    delete [] _streams;
}

void Octave::alloc_events( )
{
    _gauss_done   = new cudaEvent_t[_levels];
    _dog_done     = new cudaEvent_t[_levels];
    _extrema_done = new cudaEvent_t[_levels];
    for( int i=0; i<_levels; i++ ) {
        _gauss_done[i]   = popsift::cuda::event_create( __FILE__, __LINE__ );
        _dog_done[i]     = popsift::cuda::event_create( __FILE__, __LINE__ );
        _extrema_done[i] = popsift::cuda::event_create( __FILE__, __LINE__ );
    }
}

void Octave::free_events( )
{
    for( int i=0; i<_levels; i++ ) {
        popsift::cuda::event_destroy( _gauss_done[i],   __FILE__, __LINE__ );
        popsift::cuda::event_destroy( _dog_done[i],     __FILE__, __LINE__ );
        popsift::cuda::event_destroy( _extrema_done[i], __FILE__, __LINE__ );
    }

    delete [] _gauss_done;
    delete [] _dog_done;
    delete [] _extrema_done;
}

} // namespace popsift

