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
#define mkdir(name, mode) _mkdir(name)
#endif

#include <new> // for placement new

#include "sift_pyramid.h"
#include "sift_constants.h"
#include "common/debug_macros.h"
#include "common/clamp.h"
#include "common/write_plane_2d.h"
#include "sift_octave.h"

using namespace std;

namespace popsift {

Octave::Octave()
{ }


void Octave::alloc(int width, int height, int levels, int gauss_group)
{
    _max_w = _w = width;
    _max_h = _h = height;
    _levels = levels;

    alloc_data_planes();
    alloc_data_tex();

    alloc_interm_plane();
    alloc_interm_tex();

    alloc_dog_array();
    alloc_dog_tex();

    alloc_streams();
    alloc_events();
}

void Octave::resetDimensions( int w, int h )
{
    if( w == _w && h == _h ) {
        return;
    }

    _w = w;
    _h = h;

    if( _w <= _max_w && _h <= _max_h ) {
        free_dog_tex();
        free_dog_array();

        free_interm_tex();

        free_data_tex();
        free_data_planes();

        alloc_data_planes();
        alloc_data_tex();

        _intermediate_data.resetDimensions( w, h );
        alloc_interm_tex();

        alloc_dog_array();
        alloc_dog_tex();
    } else {
        _max_w = max( _w, _max_w );
        _max_h = max( _h, _max_h );

        free_dog_tex();
        free_dog_array();

        free_interm_tex();
        free_interm_plane();

        free_data_tex();
        free_data_planes();

        alloc_data_planes();
        alloc_data_tex();

        alloc_interm_plane();
        alloc_interm_tex();

        alloc_dog_array();
        alloc_dog_tex();
    }
}

void Octave::free()
{
    free_events();
    free_streams();

    free_dog_tex();
    free_dog_array();

    free_interm_tex();
    free_interm_plane();

    free_data_tex();
    free_data_planes();
}

/*************************************************************
 * Debug output: write an octave/level to disk as PGM
 *************************************************************/

void Octave::download_and_save_array( const char* basename, int octave )
{
        struct stat st = { 0 };

#if 0
        {
            if (level == 0) {
                int width  = getWidth();
                int height = getHeight();

                Plane2D_float hostPlane_f;
                hostPlane_f.allocHost(width, height, CudaAllocated);
                hostPlane_f.memcpyFromDevice(getData(level));

                uint32_t total_ct = 0;

                readExtremaCount();
                cudaDeviceSynchronize();
                for (uint32_t l = 0; l<_levels; l++) {
                    uint32_t ct = getExtremaCountH(l); // getExtremaCount( l );
                    if (ct > 0) {
                        total_ct += ct;

                        Extremum* cand = new Extremum[ct];

                        popcuda_memcpy_sync(cand,
                            _d_extrema[l],
                            ct * sizeof(Extremum),
                            cudaMemcpyDeviceToHost);
                        for (uint32_t i = 0; i<ct; i++) {
                            int32_t x = roundf(cand[i].xpos);
                            int32_t y = roundf(cand[i].ypos);
                            // cerr << "(" << x << "," << y << ") scale " << cand[i].sigma << " orient " << cand[i].orientation << endl;
                            for (int32_t j = -4; j <= 4; j++) {
                                hostPlane_f.ptr(clamp(y + j, height))[clamp(x, width)] = 255;
                                hostPlane_f.ptr(clamp(y, height))[clamp(x + j, width)] = 255;
                            }
                        }

                        delete[] cand;
                    }
                }

                if (total_ct > 0) {
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

                    popsift::write_plane2D(ostr.str().c_str(), false, hostPlane_f);
                    popsift::write_plane2Dunscaled(ostr2.str().c_str(), false, hostPlane_f);
                }

                hostPlane_f.freeHost(CudaAllocated);
            }
        }
#endif

            cudaError_t err;
            int width  = getWidth();
            int height = getHeight();

            if (stat("dir-octave", &st) == -1) {
                mkdir("dir-octave", 0700);
            }

            if (stat("dir-octave-dump", &st) == -1) {
                mkdir("dir-octave-dump", 0700);
            }

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
            POP_CUDA_MALLOC_HOST(&array, width * height * _levels * sizeof(float));

            cudaMemcpy3DParms s = { 0 };
            memset( &s, 0, sizeof(cudaMemcpy3DParms) );
            s.srcArray = _data;
            s.dstPtr   = make_cudaPitchedPtr( array, width * sizeof(float), width, height );
            s.extent   = make_cudaExtent( width, height, _levels );
            s.kind     = cudaMemcpyDeviceToHost;
            err = cudaMemcpy3D(&s);
            POP_CUDA_FATAL_TEST(err, "cudaMemcpy3D failed: ");

            for( int l = 0; l<_levels; l++ ) {
                Plane2D_float p(width, height, &array[l*width*height], width * sizeof(float));

                ostringstream ostr;
                ostr << "dir-octave/" << basename << "-o-" << octave << "-l-" << l << ".pgm";
                popsift::write_plane2Dunscaled( ostr.str().c_str(), false, p );

                ostringstream ostr2;
                ostr2 << "dir-octave-dump/" << basename << "-o-" << octave << "-l-" << l << ".dump";
                popsift::dump_plane2Dfloat(ostr2.str().c_str(), false, p );
            }

            memset( &s, 0, sizeof(cudaMemcpy3DParms) );
            s.srcArray = _dog_3d;
            s.dstPtr = make_cudaPitchedPtr(array, width * sizeof(float), width, height);
            s.extent = make_cudaExtent(width, height, _levels - 1);
            s.kind = cudaMemcpyDeviceToHost;
            err = cudaMemcpy3D(&s);
            POP_CUDA_FATAL_TEST(err, "cudaMemcpy3D failed: ");

            for (int l = 0; l<_levels - 1; l++) {
                Plane2D_float p(width, height, &array[l*width*height], width * sizeof(float));

                ostringstream ostr;
                ostr << "dir-dog/d-" << basename << "-o-" << octave << "-l-" << l << ".pgm";
                popsift::write_plane2D(ostr.str().c_str(), false, p);

                ostringstream pstr;
                pstr << "dir-dog-txt/d-" << basename << "-o-" << octave << "-l-" << l << ".txt";
                popsift::write_plane2Dunscaled(pstr.str().c_str(), false, p, 127);

                ostringstream qstr;
                qstr << "dir-dog-dump/d-" << basename << "-o-" << octave << "-l-" << l << ".dump";
                popsift::dump_plane2Dfloat(qstr.str().c_str(), false, p);
            }

            POP_CUDA_FREE_HOST(array);
}

void Octave::alloc_data_planes()
{
        cudaError_t err;

        _data_desc.f = cudaChannelFormatKindFloat;
        _data_desc.x = 32;
        _data_desc.y = 0;
        _data_desc.z = 0;
        _data_desc.w = 0;

        _data_ext.width  = _w; // for cudaMalloc3DArray, width in elements
        _data_ext.height = _h;
        _data_ext.depth  = _levels;

        err = cudaMalloc3DArray( &_data,
                                 &_data_desc,
                                 _data_ext,
                                 cudaArrayLayered | cudaArraySurfaceLoadStore);
        POP_CUDA_FATAL_TEST(err, "Could not allocate Blur level array: ");
}

void Octave::free_data_planes()
{
        cudaError_t err;

        err = cudaFreeArray( _data );
        POP_CUDA_FATAL_TEST(err, "Could not free Blur level array: ");
}

    void Octave::alloc_data_tex()
    {
        cudaError_t err;

        cudaResourceDesc res_desc;
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = _data;

        err = cudaCreateSurfaceObject(&_data_surf, &res_desc);
        POP_CUDA_FATAL_TEST(err, "Could not create Blur data surface: ");

        cudaTextureDesc      tex_desc;

        memset(&tex_desc, 0, sizeof(cudaTextureDesc));
        tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
        tex_desc.addressMode[0]   = cudaAddressModeClamp;
        tex_desc.addressMode[1]   = cudaAddressModeClamp;
        tex_desc.addressMode[2]   = cudaAddressModeClamp;
        tex_desc.readMode         = cudaReadModeElementType; // read as float
        tex_desc.filterMode       = cudaFilterModePoint; // no interpolation

        err = cudaCreateTextureObject( &_data_tex_point, &res_desc, &tex_desc, 0 );
        POP_CUDA_FATAL_TEST(err, "Could not create Blur data point texture: ");

        memset(&tex_desc, 0, sizeof(cudaTextureDesc));
        tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
        tex_desc.addressMode[0]   = cudaAddressModeClamp;
        tex_desc.addressMode[1]   = cudaAddressModeClamp;
        tex_desc.addressMode[2]   = cudaAddressModeClamp;
        tex_desc.readMode         = cudaReadModeElementType; // read as float
        tex_desc.filterMode       = cudaFilterModeLinear; // no interpolation

        err = cudaCreateTextureObject( &_data_tex_linear, &res_desc, &tex_desc, 0 );
        POP_CUDA_FATAL_TEST(err, "Could not create Blur data point texture: ");
    }

    void Octave::free_data_tex()
    {
        cudaError_t err;

        err = cudaDestroyTextureObject(_data_tex_point);
        POP_CUDA_FATAL_TEST(err, "Could not destroy Blur data point texture: ");

        err = cudaDestroyTextureObject(_data_tex_linear);
        POP_CUDA_FATAL_TEST(err, "Could not destroy Blur data linear texture: ");

        err = cudaDestroySurfaceObject(_data_surf);
        POP_CUDA_FATAL_TEST(err, "Could not destroy Blur data surface: ");
    }

    void Octave::alloc_interm_plane()
    {
        _intermediate_data.allocDev( _max_w, _max_h );
        _intermediate_data.resetDimensions( _w, _h );
    }

    void Octave::free_interm_plane()
    {
        _intermediate_data.freeDev();
    }

    void Octave::alloc_interm_tex()
    {
        cudaError_t err;

        cudaTextureDesc      interm_data_tex_desc;
        cudaResourceDesc     interm_data_res_desc;

        memset(&interm_data_tex_desc, 0, sizeof(cudaTextureDesc));
        interm_data_tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
        interm_data_tex_desc.addressMode[0] = cudaAddressModeClamp;
        interm_data_tex_desc.addressMode[1] = cudaAddressModeClamp;
        interm_data_tex_desc.addressMode[2] = cudaAddressModeClamp;
        interm_data_tex_desc.readMode = cudaReadModeElementType; // read as float
        interm_data_tex_desc.filterMode = cudaFilterModePoint;

        memset(&interm_data_res_desc, 0, sizeof(cudaResourceDesc));
        interm_data_res_desc.resType = cudaResourceTypePitch2D;
        interm_data_res_desc.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
        interm_data_res_desc.res.pitch2D.desc.x = 32;
        interm_data_res_desc.res.pitch2D.desc.y = 0;
        interm_data_res_desc.res.pitch2D.desc.z = 0;
        interm_data_res_desc.res.pitch2D.desc.w = 0;

        interm_data_res_desc.res.pitch2D.devPtr = _intermediate_data.data;
        interm_data_res_desc.res.pitch2D.pitchInBytes = _intermediate_data.step;
        interm_data_res_desc.res.pitch2D.width = _intermediate_data.getCols();
        interm_data_res_desc.res.pitch2D.height = _intermediate_data.getRows();

        err = cudaCreateTextureObject(&_interm_data_tex,
            &interm_data_res_desc,
            &interm_data_tex_desc, 0);
        POP_CUDA_FATAL_TEST(err, "Could not create texture object: ");
    }

    void Octave::free_interm_tex()
    {
        cudaError_t err;

        err = cudaDestroyTextureObject(_interm_data_tex);
        POP_CUDA_FATAL_TEST(err, "Could not destroy texture object: ");
    }

    void Octave::alloc_dog_array()
    {
        cudaError_t err;

        _dog_3d_desc.f = cudaChannelFormatKindFloat;
        _dog_3d_desc.x = 32;
        _dog_3d_desc.y = 0;
        _dog_3d_desc.z = 0;
        _dog_3d_desc.w = 0;

        _dog_3d_ext.width = _w; // for cudaMalloc3DArray, width in elements
        _dog_3d_ext.height = _h;
        _dog_3d_ext.depth = _levels - 1;

        err = cudaMalloc3DArray(&_dog_3d,
            &_dog_3d_desc,
            _dog_3d_ext,
            cudaArrayLayered | cudaArraySurfaceLoadStore);
        POP_CUDA_FATAL_TEST(err, "Could not allocate 3D DoG array: ");
    }

    void Octave::free_dog_array()
    {
        cudaError_t err;

        err = cudaFreeArray(_dog_3d);
        POP_CUDA_FATAL_TEST(err, "Could not free 3D DoG array: ");
    }

    void Octave::alloc_dog_tex()
    {
        cudaError_t err;

        cudaResourceDesc dog_res_desc;
        dog_res_desc.resType = cudaResourceTypeArray;
        dog_res_desc.res.array.array = _dog_3d;

        err = cudaCreateSurfaceObject(&_dog_3d_surf, &dog_res_desc);
        POP_CUDA_FATAL_TEST(err, "Could not create DoG surface: ");

        cudaTextureDesc      dog_tex_desc;
        memset(&dog_tex_desc, 0, sizeof(cudaTextureDesc));
        dog_tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
        dog_tex_desc.addressMode[0] = cudaAddressModeClamp;
        dog_tex_desc.addressMode[1] = cudaAddressModeClamp;
        dog_tex_desc.addressMode[2] = cudaAddressModeClamp;
        dog_tex_desc.readMode = cudaReadModeElementType; // read as float
        dog_tex_desc.filterMode = cudaFilterModePoint; // no interpolation

        err = cudaCreateTextureObject(&_dog_3d_tex_point, &dog_res_desc, &dog_tex_desc, 0);
        POP_CUDA_FATAL_TEST(err, "Could not create DoG texture: ");

        dog_tex_desc.filterMode = cudaFilterModeLinear; // linear interpolation
        err = cudaCreateTextureObject(&_dog_3d_tex_linear, &dog_res_desc, &dog_tex_desc, 0);
        POP_CUDA_FATAL_TEST(err, "Could not create DoG texture: ");
    }

    void Octave::free_dog_tex()
    {
        cudaError_t err;

        err = cudaDestroyTextureObject(_dog_3d_tex_linear);
        POP_CUDA_FATAL_TEST(err, "Could not destroy DoG texture: ");

        err = cudaDestroyTextureObject(_dog_3d_tex_point);
        POP_CUDA_FATAL_TEST(err, "Could not destroy DoG texture: ");

        err = cudaDestroySurfaceObject(_dog_3d_surf);
        POP_CUDA_FATAL_TEST(err, "Could not destroy DoG surface: ");
    }

    void Octave::alloc_streams()
    {
        _stream = popsift::cuda::stream_create(__FILE__, __LINE__);
    }

    void Octave::free_streams()
    {
        popsift::cuda::stream_destroy( _stream, __FILE__, __LINE__ );
    }

    void Octave::alloc_events()
    {
        _scale_done   = popsift::cuda::event_create(__FILE__, __LINE__);
        _extrema_done = popsift::cuda::event_create(__FILE__, __LINE__);
        _ori_done     = popsift::cuda::event_create(__FILE__, __LINE__);
        _desc_done    = popsift::cuda::event_create(__FILE__, __LINE__);
    }

    void Octave::free_events()
    {
        popsift::cuda::event_destroy( _scale_done,   __FILE__, __LINE__);
        popsift::cuda::event_destroy( _extrema_done, __FILE__, __LINE__);
        popsift::cuda::event_destroy( _ori_done,     __FILE__, __LINE__);
        popsift::cuda::event_destroy( _desc_done,    __FILE__, __LINE__);
    }

} // namespace popsift
