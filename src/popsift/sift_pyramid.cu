/*
* Copyright 2016, Simula Research Laboratory
*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "common/assist.h"
#include "common/debug_macros.h"
#include "sift_config.h"
#include "sift_extremum.h"
#include "sift_pyramid.h"

#include <sys/stat.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#ifdef _WIN32
#include <direct.h>
#define stat _stat
#define mkdir(path, perm) _mkdir(path)
#endif

#if POPSIFT_IS_DEFINED(POPSIFT_USE_NVTX)
#include <nvToolsExtCuda.h>
#else
#define nvtxRangePushA(a)
#define nvtxRangePop()
#endif

#define PYRAMID_PRINT_DEBUG 0

using namespace std;

namespace popsift {

__global__
    void py_print_corner_float(float* img, uint32_t pitch, uint32_t height, uint32_t level)
{
    const int xbase = 0;
    const int ybase = level * height + 0;
    for (int i = 0; i<10; i++) {
        for (int j = 0; j<10; j++) {
            printf("%3.3f ", img[(ybase + i)*pitch + xbase + j]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__
    void py_print_corner_float_transposed(float* img, uint32_t pitch, uint32_t height, uint32_t level)
{
    const int xbase = 0;
    const int ybase = level * height + 0;
    for (int i = 0; i<10; i++) {
        for (int j = 0; j<10; j++) {
            printf("%3.3f ", img[(ybase + j)*pitch + xbase + i]);
        }
        printf("\n");
    }
    printf("\n");
}

void Pyramid::download_and_save_array( const char* basename )
{
    for( int o=0; o<_num_octaves; o++ )
    _octaves[o].download_and_save_array( basename, o );
}

/*
 * Note this is only for debug output. FeaturesHost has functions for final writing.
 */
void Pyramid::save_descriptors( const Config& conf, FeaturesHost* features, const char* basename )
{
    struct stat st = { 0 };
    if (stat("dir-desc", &st) == -1) {
        mkdir("dir-desc", 0700);
    }
    ostringstream ostr;
    ostr << "dir-desc/desc-" << basename << ".txt";
    ofstream of(ostr.str().c_str());
    writeDescriptor( conf, of, features, true, true );

    if (stat("dir-fpt", &st) == -1) {
        mkdir("dir-fpt", 0700);
    }
    ostringstream ostr2;
    ostr2 << "dir-fpt/desc-" << basename << ".txt";
    ofstream of2(ostr2.str().c_str());
    writeDescriptor( conf, of2, features, false, true );
}

Pyramid::Pyramid( const Config& config,
                  int width,
                  int height )
    : _num_octaves( config.octaves )
    , _levels( config.levels + 3 )
    , _assume_initial_blur( config.hasInitialBlur() )
    , _initial_blur( config.getInitialBlur() )
{
    _octaves = new Octave[_num_octaves];

    int w = width;
    int h = height;

    // memset( &hct,         0, sizeof(ExtremaCounters) );
    // cudaMemcpyToSymbol( dct, &hct, sizeof(ExtremaCounters), 0, cudaMemcpyHostToDevice );

    cudaMallocManaged( &_ct, sizeof(ExtremaCounters) );
    memset( _ct, 0, sizeof(ExtremaCounters) );

    cudaMallocManaged( &_buf, sizeof(ExtremaBuffers) );
    memset( _buf, 0, sizeof(ExtremaBuffers) );

    _d_extrema_num_blocks = popsift::cuda::malloc_devT<int>( _num_octaves, __FILE__, __LINE__ );

    for (int o = 0; o<_num_octaves; o++) {
        _octaves[o].debugSetOctave(o);
        _octaves[o].alloc( config, w, h, _levels, _gauss_group );
        w = ceilf(w / 2.0f);
        h = ceilf(h / 2.0f);
    }

    int sz = _num_octaves * h_consts.max_extrema;
    _buf->i_ext_dat[0] = popsift::cuda::malloc_mgdT<InitialExtremum>( sz, __FILE__, __LINE__);
    _buf->i_ext_off[0] = popsift::cuda::malloc_mgdT<int>( sz, __FILE__, __LINE__);
    for (int o = 1; o<_num_octaves; o++) {
        _buf->i_ext_dat[o] = _buf->i_ext_dat[0] + (o*h_consts.max_extrema);
        _buf->i_ext_off[o] = _buf->i_ext_off[0] + (o*h_consts.max_extrema);
    }
    for (int o = _num_octaves; o<MAX_OCTAVES; o++) {
        _buf->i_ext_dat[o] = nullptr;
        _buf->i_ext_off[o] = nullptr;
    }

    sz = h_consts.max_extrema;
    _buf->extrema  = popsift::cuda::malloc_mgdT<Extremum>( sz, __FILE__, __LINE__);
    _buf->features = popsift::cuda::malloc_mgdT<Feature>( sz, __FILE__, __LINE__);

    _buf->ext_allocated = sz;

    sz = max( 2 * h_consts.max_extrema, h_consts.max_orientations );
    _buf->desc            = popsift::cuda::malloc_mgdT<Descriptor>( sz, __FILE__, __LINE__);
    _buf->feat_to_ext_map = popsift::cuda::malloc_mgdT<int>( sz, __FILE__, __LINE__);
    _buf->ori_allocated = sz;

    cudaStreamCreate( &_download_stream );
}

void Pyramid::resetDimensions( const Config& conf, int width, int height )
{
    int w = width;
    int h = height;

    for (int o = 0; o<_num_octaves; o++) {
        _octaves[o].resetDimensions( conf, w, h );
        w = ceilf(w / 2.0f);
        h = ceilf(h / 2.0f);
    }
}

void Pyramid::reallocExtrema( int numExtrema )
{
    if( numExtrema > _buf->ext_allocated ) {
        numExtrema = ( ( numExtrema + 1024 ) & ( ~(1024-1) ) );
        cudaFree( _buf->extrema );
        cudaFree( _buf->features );

        int sz = numExtrema;
        _buf->extrema  = popsift::cuda::malloc_mgdT<Extremum>( sz, __FILE__, __LINE__);
        _buf->features = popsift::cuda::malloc_mgdT<Feature>( sz, __FILE__, __LINE__);
        _buf->ext_allocated = sz;

        numExtrema *= 2;
        if( numExtrema > _buf->ori_allocated ) {
            cudaFree( _buf->desc );
            cudaFree( _buf->feat_to_ext_map );

            sz = numExtrema;
            _buf->desc = popsift::cuda::malloc_mgdT<Descriptor>( sz, __FILE__, __LINE__);
            _buf->feat_to_ext_map = popsift::cuda::malloc_mgdT<int>( sz, __FILE__, __LINE__);
            _buf->ori_allocated = sz;
        }
    }
}

Pyramid::~Pyramid()
{
    cudaStreamDestroy( _download_stream );

    cudaFree(     _d_extrema_num_blocks );
    cudaFree(     _buf->i_ext_dat[0] );
    cudaFree(     _buf->i_ext_off[0] );
    cudaFree(     _buf->features );
    cudaFree(     _buf->extrema );
    cudaFree(     _buf->desc );
    cudaFree(     _buf->feat_to_ext_map );

    cudaFree( _ct );
    cudaFree( _buf );

    delete[] _octaves;
}

void Pyramid::step1( const Config& conf, popsift::ImageBase* img )
{
    reset_extrema_mgmt( );
    build_pyramid( conf, img );
}

void Pyramid::step2( const Config& conf )
{
    find_extrema( conf );

    orientation( conf );

    descriptors( conf );
}

/* Important detail: this function takes the pointer descriptor_base as input
 * and computes offsets from this pointer on the device side. Those pointers
 * are then written into Feature data structures.
 * descriptor_base can be a device pointer or a host pointer, it works in both
 * cases.
 * This is possible because pointer arithmetic between Intel hosts and NVidia
 * GPUs are compatible.
 */
__global__
void prep_features( Descriptor* descriptor_base, ExtremaCounters* ct, ExtremaBuffers* buf, int up_fac )
{
    int offset = blockIdx.x * 32 + threadIdx.x;
    if( offset >= ct->ext_total ) return;
    const Extremum& ext = buf->extrema [offset];
    Feature&        fet = buf->features[offset];

    const int   octave  = ext.octave;
    const float xpos    = ext.xpos  * powf(2.0f, float(octave - up_fac));
    const float ypos    = ext.ypos  * powf(2.0f, float(octave - up_fac));
    const float sigma   = ext.sigma * powf(2.0f, float(octave - up_fac));
    const int   num_ori = ext.num_ori;

    fet.xpos    = xpos;
    fet.ypos    = ypos;
    fet.sigma   = sigma;
    fet.num_ori = num_ori;

    fet.debug_octave = octave;

    int ori;
    for( ori = 0; ori<num_ori; ori++ ) {
        fet.desc[ori]        = descriptor_base + ( ext.idx_ori + ori );
        fet.orientation[ori] = ext.orientation[ori];
    }
    for( ; ori<ORIENTATION_MAX_COUNT; ori++ ) {
        fet.desc[ori]        = nullptr;
        fet.orientation[ori] = 0;
    }
}

FeaturesHost* Pyramid::get_descriptors( const Config& conf )
{
    const float up_fac = conf.getUpscaleFactor();

    readDescCountersFromDevice();

    nvtxRangePushA( "download descriptors" );
    FeaturesHost* features = new FeaturesHost( _ct->ext_total, _ct->ori_total );

    if( _ct->ext_total == 0 || _ct->ori_total == 0 )
    {
        nvtxRangePop();
        return features;
    }

    dim3 grid( grid_divide( _ct->ext_total, 32 ) );
    prep_features<<<grid,32,0,_download_stream>>>( features->getDescriptors(), _ct, _buf, up_fac );
    POP_SYNC_CHK;

    nvtxRangePushA( "register host memory" );
    features->pin( );
    nvtxRangePop();
    popcuda_memcpy_async( features->getFeatures(),
                          _buf->features,
                          _ct->ext_total * sizeof(Feature),
                          cudaMemcpyDeviceToHost,
                          _download_stream );

    popcuda_memcpy_async( features->getDescriptors(),
                          _buf->desc,
                          _ct->ori_total * sizeof(Descriptor),
                          cudaMemcpyDeviceToHost,
                          _download_stream );
    cudaStreamSynchronize( _download_stream );
    nvtxRangePushA( "unregister host memory" );
    features->unpin( );
    nvtxRangePop();
    nvtxRangePop();

    return features;
}

void Pyramid::clone_device_descriptors_sub( const Config& conf, FeaturesDev* features )
{
    const float up_fac = conf.getUpscaleFactor();

    dim3 grid( grid_divide( _ct->ext_total, 32 ) );
    prep_features<<<grid,32,0,_download_stream>>>( features->getDescriptors(), _ct, _buf, up_fac );
    POP_SYNC_CHK;

    popcuda_memcpy_async( features->getFeatures(),
                          _buf->features,
                          _ct->ext_total * sizeof(Feature),
                          cudaMemcpyDeviceToDevice,
                          _download_stream );

    popcuda_memcpy_async( features->getDescriptors(),
                          _buf->desc,
                          _ct->ori_total * sizeof(Descriptor),
                          cudaMemcpyDeviceToDevice,
                          _download_stream );

    popcuda_memcpy_async( features->getReverseMap(),
                          _buf->feat_to_ext_map,
                          _ct->ori_total * sizeof(int),
                          cudaMemcpyDeviceToDevice,
                          _download_stream );
}

FeaturesDev* Pyramid::clone_device_descriptors( const Config& conf )
{
    readDescCountersFromDevice();

    FeaturesDev* features = new FeaturesDev( _ct->ext_total, _ct->ori_total );

    clone_device_descriptors_sub( conf, features );

    cudaStreamSynchronize( _download_stream );

    return features;
}

void Pyramid::reset_extrema_mgmt()
{
    memset( _ct, 0, sizeof(ExtremaCounters) );

    popcuda_memset_sync( _d_extrema_num_blocks, 0, _num_octaves * sizeof(int) );

}

void Pyramid::readDescCountersFromDevice( )
{
    // cudaMemcpyFromSymbol( &hct, dct, sizeof(ExtremaCounters), 0, cudaMemcpyDeviceToHost );
    // we must not copy mgmt memory explicitly, just wait for the device driver
    cudaDeviceSynchronize();
}

void Pyramid::readDescCountersFromDevice( cudaStream_t s )
{
    cudaStreamSynchronize( s );
}

void Pyramid::writeDescCountersToDevice( )
{
    cudaDeviceSynchronize();
}

void Pyramid::writeDescCountersToDevice( cudaStream_t s )
{
    cudaStreamSynchronize( s );
}

int* Pyramid::getNumberOfBlocks( int octave )
{
    return &_d_extrema_num_blocks[octave];
}

/*
 * Note this is only for debug output. FeaturesHost has functions for final writing.
 */
void Pyramid::writeDescriptor( const Config& conf, ostream& ostr, FeaturesHost* features, bool really, bool with_orientation )
{
    if( features->getFeatureCount() == 0 ) return;

    const float up_fac = conf.getUpscaleFactor();

    for( int ext_idx = 0; ext_idx<_ct->ext_total; ext_idx++ ) {
        const Feature& ext = features->getFeatures()[ext_idx];
        const int   octave  = ext.debug_octave;
        const float xpos    = ext.xpos  * pow(2.0f, octave - up_fac);
        const float ypos    = ext.ypos  * pow(2.0f, octave - up_fac);
        const float sigma   = ext.sigma * pow(2.0f, octave - up_fac);
        for( int ori = 0; ori<ext.num_ori; ori++ ) {
            // const int   ori_idx = ext.idx_ori + ori;
            float       dom_ori = ext.orientation[ori];

            dom_ori = dom_ori / M_PI2 * 360;
            if (dom_ori < 0) dom_ori += 360;

            const Descriptor& desc  = *ext.desc[ori];

            if( with_orientation )
                ostr << setprecision(5)
                     << xpos << " "
                     << ypos << " "
                     << sigma << " "
                     << dom_ori << " ";
            else
                ostr << setprecision(5)
                     << xpos << " " << ypos << " "
                     << 1.0f / (sigma * sigma)
                     << " 0 "
                     << 1.0f / (sigma * sigma) << " ";

            if (really) {
                for (float feature : desc.features)
                {
                    ostr << feature << " ";
                }
            }
            ostr << endl;
        }
    }
}


} // namespace popsift
