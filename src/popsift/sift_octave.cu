/*
* Copyright 2016, Simula Research Laboratory
*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "common/clamp.h"
#include "common/debug_macros.h"
#include "common/write_plane_2d.h"
#include "sift_constants.h"
#include "sift_octave.h"
#include "sift_pyramid.h"

#include <sys/stat.h>

#include <new> // for placement new
#include <sstream>
#ifdef _WIN32
#include <direct.h>
#define stat _stat
#define mkdir(name, mode) _mkdir(name)
#endif

using namespace std;

namespace popsift {

Octave::Octave()
{ }


void Octave::alloc( const Config& conf, int width, int height, int levels, int gauss_group )
{
    _max_w = _w = width;
    _max_h = _h = height;
    _levels = levels;

    _w_grid_divider = float(_w) / conf.getFilterGridSize();
    _h_grid_divider = float(_h) / conf.getFilterGridSize();

    alloc_data_planes();
    alloc_data_tex();

    alloc_interm_array();
    alloc_interm_tex();

    alloc_dog_array();
    alloc_dog_tex();

    alloc_streams();
    alloc_events();
}

void Octave::resetDimensions( const Config& conf, int w, int h )
{
    if( w == _w && h == _h ) {
        return;
    }

    _w = w;
    _h = h;

    _w_grid_divider = float(_w) / conf.getFilterGridSize();
    _h_grid_divider = float(_h) / conf.getFilterGridSize();

    if( _w > _max_w || _h > _max_h ) {
        _max_w = max( _w, _max_w );
        _max_h = max( _h, _max_h );
    }

    free_dog_tex();
    free_dog_array();

    free_interm_tex();
    free_interm_array();

    free_data_tex();
    free_data_planes();

    alloc_data_planes();
    alloc_data_tex();

    alloc_interm_array();
    alloc_interm_tex();

    alloc_dog_array();
    alloc_dog_tex();
}

void Octave::free()
{
    free_events();
    free_streams();

    free_dog_tex();
    free_dog_array();

    free_interm_tex();
    free_interm_array();

    free_data_tex();
    free_data_planes();
}

/*************************************************************
 * Debug output: write an octave/level to disk as PGM
 *************************************************************/

void Octave::download_and_save_array( const char* basename, int octave )
{
    struct stat st = { 0 };

    cudaError_t err;
    int width  = getWidth();
    int height = getHeight();

    if (stat("dir-octave", &st) == -1) {
        mkdir("dir-octave", 0700);
    }

    if (stat("dir-interm", &st) == -1) {
        mkdir("dir-interm", 0700);
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
        const Plane2D_float p(width, height, &array[l*width*height], width * sizeof(float));

        ostringstream ostr;
        ostr << "dir-octave/" << basename << "-o-" << octave << "-l-" << l << ".pgm";
        popsift::write_plane2Dunscaled( ostr.str().c_str(), false, p );

        ostringstream ostr2;
        ostr2 << "dir-octave-dump/" << basename << "-o-" << octave << "-l-" << l << ".dump";
        popsift::dump_plane2Dfloat(ostr2.str().c_str(), false, p );
    }

    memset( &s, 0, sizeof(cudaMemcpy3DParms) );
    s.srcArray = _intm;
    s.dstPtr   = make_cudaPitchedPtr( array, width * sizeof(float), width, height );
    s.extent   = make_cudaExtent( width, height, _levels );
    s.kind     = cudaMemcpyDeviceToHost;
    err = cudaMemcpy3D(&s);
    POP_CUDA_FATAL_TEST(err, "cudaMemcpy3D failed: ");

    for( int l = 0; l<_levels; l++ ) {
        const Plane2D_float p(width, height, &array[l*width*height], width * sizeof(float));

        ostringstream ostr;
        ostr << "dir-interm/" << basename << "-o-" << octave << "-l-" << l << ".pgm";
        popsift::write_plane2Dunscaled( ostr.str().c_str(), false, p );
    }

    memset( &s, 0, sizeof(cudaMemcpy3DParms) );
    s.srcArray = _dog_3d;
    s.dstPtr = make_cudaPitchedPtr(array, width * sizeof(float), width, height);
    s.extent = make_cudaExtent(width, height, _levels - 1);
    s.kind = cudaMemcpyDeviceToHost;
    err = cudaMemcpy3D(&s);
    POP_CUDA_FATAL_TEST(err, "cudaMemcpy3D failed: ");

    for (int l = 0; l<_levels - 1; l++) {
        const Plane2D_float p(width, height, &array[l*width*height], width * sizeof(float));

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

    // Observed on gfx90a and on gfx1100 (ROCm 7.2.1): a layered array written
    // layer by layer through surf2DLayeredwrite reads back as a single layer for
    // every layer index, because the write passed the layer index in the mipmap
    // level slot. Filed as ROCm/clr#275; ROCm/rocm-systems#6683 corrects it, but
    // that is not in ROCm 7.2.x. A non-layered 3D array with surf3D/tex3D access
    // is coherent, so drop cudaArrayLayered on HIP. The blur levels are addressed
    // by the z coordinate instead of the layer index. See cuda_to_hip.h,
    // sift_textures.h and common/assist.h. CUDA keeps a real layered array.
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
    err = cudaMalloc3DArray( &_data, &_data_desc, _data_ext, cudaArraySurfaceLoadStore );
#else
    err = cudaMalloc3DArray( &_data,
                             &_data_desc,
                             _data_ext,
                             cudaArrayLayered | cudaArraySurfaceLoadStore);
#endif
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

    cudaResourceDesc res_desc{};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = _data;

    err = cudaCreateSurfaceObject(&_data_surf, &res_desc);
    POP_CUDA_FATAL_TEST(err, "Could not create Blur data surface: ");

    cudaTextureDesc      tex_desc{};

    memset(&tex_desc, 0, sizeof(cudaTextureDesc));
    tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
    tex_desc.addressMode[0]   = cudaAddressModeClamp;
    tex_desc.addressMode[1]   = cudaAddressModeClamp;
    tex_desc.addressMode[2]   = cudaAddressModeClamp;
    tex_desc.readMode         = cudaReadModeElementType; // read as float
    tex_desc.filterMode       = cudaFilterModePoint; // no interpolation

    err = cudaCreateTextureObject( &_data_tex_point, &res_desc, &tex_desc, nullptr );
    POP_CUDA_FATAL_TEST(err, "Could not create Blur data point texture: ");

    memset(&tex_desc, 0, sizeof(cudaTextureDesc));
    tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
    tex_desc.addressMode[0]   = cudaAddressModeClamp;
    tex_desc.addressMode[1]   = cudaAddressModeClamp;
    tex_desc.addressMode[2]   = cudaAddressModeClamp;
    tex_desc.readMode         = cudaReadModeElementType; // read as float
    tex_desc.filterMode       = cudaFilterModeLinear; // hardware bilinear (CUDA)
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
    // Observed on gfx90a (ROCm 7.2.1): hipCreateTextureObject rejects hardware
    // linear filtering on element-read float arrays ("operation not supported").
    // Create the texture with point filtering and do bilinear interpolation in
    // software in readTex(). Empirical on this device/ROCm; re-verify on RDNA.
    tex_desc.filterMode       = cudaFilterModePoint;
#endif

    err = cudaCreateTextureObject( &_data_tex_linear.tex, &res_desc, &tex_desc, nullptr );
    POP_CUDA_FATAL_TEST(err, "Could not create Blur data point texture: ");
}

void Octave::free_data_tex()
{
        cudaError_t err;

        err = cudaDestroyTextureObject(_data_tex_point);
        POP_CUDA_FATAL_TEST(err, "Could not destroy Blur data point texture: ");

        err = cudaDestroyTextureObject(_data_tex_linear.tex);
        POP_CUDA_FATAL_TEST(err, "Could not destroy Blur data linear texture: ");

        err = cudaDestroySurfaceObject(_data_surf);
        POP_CUDA_FATAL_TEST(err, "Could not destroy Blur data surface: ");
}

void Octave::alloc_interm_array()
{
    cudaError_t err;

    _intm_desc.f = cudaChannelFormatKindFloat;
    _intm_desc.x = 32;
    _intm_desc.y = 0;
    _intm_desc.z = 0;
    _intm_desc.w = 0;

    _intm_ext.width  = _w; // for cudaMalloc3DArray, width in elements
    _intm_ext.height = _h;
    _intm_ext.depth  = _levels;

    // Non-layered 3D array on HIP (see alloc_data_planes()).
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
    err = cudaMalloc3DArray( &_intm, &_intm_desc, _intm_ext, cudaArraySurfaceLoadStore );
#else
    err = cudaMalloc3DArray( &_intm,
                             &_intm_desc,
                             _intm_ext,
                             cudaArrayLayered | cudaArraySurfaceLoadStore);
#endif
    POP_CUDA_FATAL_TEST(err, "Could not allocate Intermediate layered array: ");
}

void Octave::free_interm_array()
{
    cudaError_t err;

    err = cudaFreeArray( _intm );
    POP_CUDA_FATAL_TEST(err, "Could not free Intermediate layered array: ");
}

void Octave::alloc_interm_tex()
{
    cudaError_t err;

    cudaResourceDesc res_desc{};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = _intm;

    err = cudaCreateSurfaceObject(&_intm_surf, &res_desc);
    POP_CUDA_FATAL_TEST(err, "Could not create Blur intermediate surface: ");

    cudaTextureDesc      tex_desc{};

    memset(&tex_desc, 0, sizeof(cudaTextureDesc));
    tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
    tex_desc.addressMode[0]   = cudaAddressModeClamp;
    tex_desc.addressMode[1]   = cudaAddressModeClamp;
    tex_desc.addressMode[2]   = cudaAddressModeClamp;
    tex_desc.readMode         = cudaReadModeElementType; // read as float
    tex_desc.filterMode       = cudaFilterModePoint; // no interpolation

    err = cudaCreateTextureObject( &_intm_tex_point, &res_desc, &tex_desc, nullptr );
    POP_CUDA_FATAL_TEST(err, "Could not create Blur intermediate point texture: ");

    tex_desc.filterMode       = cudaFilterModeLinear; // hardware bilinear (CUDA)
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
    // See alloc_data_tex(): HIP rejects linear filtering on element-read float
    // arrays, so use point filtering here and interpolate in software in readTex().
    tex_desc.filterMode       = cudaFilterModePoint;
#endif

    err = cudaCreateTextureObject( &_intm_tex_linear.tex, &res_desc, &tex_desc, nullptr );
    POP_CUDA_FATAL_TEST(err, "Could not create Blur intermediate point texture: ");
}

void Octave::free_interm_tex()
{
    cudaError_t err;

    err = cudaDestroyTextureObject(_intm_tex_point);
    POP_CUDA_FATAL_TEST(err, "Could not destroy Blur intermediate point texture: ");

    err = cudaDestroyTextureObject(_intm_tex_linear.tex);
    POP_CUDA_FATAL_TEST(err, "Could not destroy Blur intermediate linear texture: ");

    err = cudaDestroySurfaceObject(_intm_surf);
    POP_CUDA_FATAL_TEST(err, "Could not destroy Blur intermediate surface: ");
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

        // Non-layered 3D array on HIP (see alloc_data_planes()).
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
        err = cudaMalloc3DArray(&_dog_3d, &_dog_3d_desc, _dog_3d_ext, cudaArraySurfaceLoadStore);
#else
        err = cudaMalloc3DArray(&_dog_3d,
            &_dog_3d_desc,
            _dog_3d_ext,
            cudaArrayLayered | cudaArraySurfaceLoadStore);
#endif
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

        cudaResourceDesc dog_res_desc{};
        dog_res_desc.resType = cudaResourceTypeArray;
        dog_res_desc.res.array.array = _dog_3d;

        err = cudaCreateSurfaceObject(&_dog_3d_surf, &dog_res_desc);
        POP_CUDA_FATAL_TEST(err, "Could not create DoG surface: ");

        cudaTextureDesc      dog_tex_desc{};
        memset(&dog_tex_desc, 0, sizeof(cudaTextureDesc));
        dog_tex_desc.normalizedCoords = 0; // addressed (x,y) in [width,height]
        dog_tex_desc.addressMode[0] = cudaAddressModeClamp;
        dog_tex_desc.addressMode[1] = cudaAddressModeClamp;
        dog_tex_desc.addressMode[2] = cudaAddressModeClamp;
        dog_tex_desc.readMode = cudaReadModeElementType; // read as float
        dog_tex_desc.filterMode = cudaFilterModePoint; // no interpolation

        err = cudaCreateTextureObject(&_dog_3d_tex_point, &dog_res_desc, &dog_tex_desc, 0);
        POP_CUDA_FATAL_TEST(err, "Could not create DoG texture: ");
}

void Octave::free_dog_tex()
{
    cudaError_t err;

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
