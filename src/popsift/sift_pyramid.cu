/*
* Copyright 2016, Simula Research Laboratory
*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#define stat _stat
#define mkdir(path, perm) _mkdir(path)
#endif

#include "sift_pyramid.h"
#include "sift_extremum.h"
#include "common/debug_macros.h"
#include "common/assist.h"

#ifdef USE_NVTX
#include <nvToolsExtCuda.h>
#else
#define nvtxRangePushA(a)
#define nvtxRangePop()
#endif

#define PYRAMID_PRINT_DEBUG 0

using namespace std;

namespace popsift {

__device__
ExtremaCounters dct;
ExtremaCounters hct;

__device__
ExtremaBuffers  dbuf;
ExtremaBuffers  dbuf_shadow; // just for managing memories
ExtremaBuffers  hbuf;

__device__
DevBuffers      dobuf;
DevBuffers      dobuf_shadow; // just for managing memories

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

void Pyramid::save_descriptors( const Config& conf, Features* features, const char* basename )
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

    memset( &hct,         0, sizeof(ExtremaCounters) );
    cudaMemcpyToSymbol( dct, &hct, sizeof(ExtremaCounters), 0, cudaMemcpyHostToDevice );

    memset( &hbuf,        0, sizeof(ExtremaBuffers) );
    memset( &dbuf_shadow, 0, sizeof(ExtremaBuffers) );

    _d_extrema_num_blocks = popsift::cuda::malloc_devT<int>( _num_octaves, __FILE__, __LINE__ );

    for (int o = 0; o<_num_octaves; o++) {
        _octaves[o].debugSetOctave(o);
        _octaves[o].alloc( config, w, h, _levels, _gauss_group );
        w = ceilf(w / 2.0f);
        h = ceilf(h / 2.0f);
    }

    int sz = _num_octaves * h_consts.max_extrema;
    dobuf_shadow.i_ext_dat[0] = popsift::cuda::malloc_devT<InitialExtremum>( sz, __FILE__, __LINE__);
    dobuf_shadow.i_ext_off[0] = popsift::cuda::malloc_devT<int>( sz, __FILE__, __LINE__);
    for (int o = 1; o<_num_octaves; o++) {
        dobuf_shadow.i_ext_dat[o] = dobuf_shadow.i_ext_dat[0] + (o*h_consts.max_extrema);
        dobuf_shadow.i_ext_off[o] = dobuf_shadow.i_ext_off[0] + (o*h_consts.max_extrema);
    }
    for (int o = _num_octaves; o<MAX_OCTAVES; o++) {
        dobuf_shadow.i_ext_dat[o] = 0;
        dobuf_shadow.i_ext_off[o] = 0;
    }

    sz = h_consts.max_extrema;
    dobuf_shadow.extrema      = popsift::cuda::malloc_devT<Extremum>( sz, __FILE__, __LINE__);
    dobuf_shadow.features     = popsift::cuda::malloc_devT<Feature>( sz, __FILE__, __LINE__);
    hbuf       .ext_allocated = sz;
    dbuf_shadow.ext_allocated = sz;

    sz = max( 2 * h_consts.max_extrema, h_consts.max_orientations );
    hbuf       .desc               = popsift::cuda::malloc_hstT<Descriptor>( sz, __FILE__, __LINE__);
    dbuf_shadow.desc               = popsift::cuda::malloc_devT<Descriptor>( sz, __FILE__, __LINE__);
    dobuf_shadow.feat_to_ext_map   = popsift::cuda::malloc_devT<int>( sz, __FILE__, __LINE__);
    hbuf       .ori_allocated = sz;
    dbuf_shadow.ori_allocated = sz;

    cudaMemcpyToSymbol( dbuf,  &dbuf_shadow,  sizeof(ExtremaBuffers), 0, cudaMemcpyHostToDevice );
    cudaMemcpyToSymbol( dobuf, &dobuf_shadow, sizeof(DevBuffers),     0, cudaMemcpyHostToDevice );

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
    if( numExtrema > hbuf.ext_allocated ) {
        numExtrema = ( ( numExtrema + 1024 ) & ( ~(1024-1) ) );
        cudaFree( dobuf_shadow.extrema );
        cudaFree( dobuf_shadow.features );

        int sz = numExtrema;
        dobuf_shadow.extrema  = popsift::cuda::malloc_devT<Extremum>( sz, __FILE__, __LINE__);
        dobuf_shadow.features = popsift::cuda::malloc_devT<Feature>( sz, __FILE__, __LINE__);
        hbuf       .ext_allocated = sz;
        dbuf_shadow.ext_allocated = sz;

        numExtrema *= 2;
        if( numExtrema > hbuf.ori_allocated ) {
            cudaFreeHost( hbuf       .desc );
            cudaFree(     dbuf_shadow.desc );
            cudaFree(     dobuf_shadow.feat_to_ext_map );

            sz = numExtrema;
            hbuf       .desc             = popsift::cuda::malloc_hstT<Descriptor>( sz, __FILE__, __LINE__);
            dbuf_shadow.desc             = popsift::cuda::malloc_devT<Descriptor>( sz, __FILE__, __LINE__);
            dobuf_shadow.feat_to_ext_map = popsift::cuda::malloc_devT<int>( sz, __FILE__, __LINE__);
            hbuf       .ori_allocated = sz;
            dbuf_shadow.ori_allocated = sz;
        }

        cudaMemcpyToSymbol( dbuf,  &dbuf_shadow,  sizeof(ExtremaBuffers), 0, cudaMemcpyHostToDevice );
        cudaMemcpyToSymbol( dobuf, &dobuf_shadow, sizeof(DevBuffers),     0, cudaMemcpyHostToDevice );
    }
}

Pyramid::~Pyramid()
{
    cudaStreamDestroy( _download_stream );

    cudaFree(     dobuf_shadow.i_ext_dat[0] );
    cudaFree(     dobuf_shadow.i_ext_off[0] );
    cudaFree(     dobuf_shadow.features );
    cudaFree(     dobuf_shadow.extrema );
    cudaFreeHost( hbuf        .desc );
    cudaFree(     dbuf_shadow .desc );
    cudaFree(     dobuf_shadow.feat_to_ext_map );

    delete[] _octaves;
}

void Pyramid::step1( const Config& conf, popsift::Image* img )
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

__global__
void prep_features( Descriptor* descriptor_base, int up_fac )
{
    int offset = blockIdx.x * 32 + threadIdx.x;
    if( offset >= dct.ext_total ) return;
    const Extremum& ext = dobuf.extrema [offset];
    Feature&        fet = dobuf.features[offset];

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
        fet.desc[ori]        = 0;
        fet.orientation[ori] = 0;
    }
}

Features* Pyramid::get_descriptors( const Config& conf )
{
    const float up_fac = conf.getUpscaleFactor();

    readDescCountersFromDevice();

    nvtxRangePushA( "download descriptors" );
    Features* features = new Features( hct.ext_total, hct.ori_total );

    dim3 grid( grid_divide( hct.ext_total, 32 ) );
    prep_features<<<grid,32,0,_download_stream>>>( features->getDescriptors(), up_fac );

    nvtxRangePushA( "register host memory" );
    features->pin( );
    nvtxRangePop();
    popcuda_memcpy_async( features->getFeatures(),
                          dobuf_shadow.features,
                          hct.ext_total * sizeof(Feature),
                          cudaMemcpyDeviceToHost,
                          _download_stream );

    popcuda_memcpy_async( features->getDescriptors(),
                          dbuf_shadow.desc,
                          hct.ori_total * sizeof(Descriptor),
                          cudaMemcpyDeviceToHost,
                          _download_stream );
    cudaStreamSynchronize( _download_stream );
    nvtxRangePushA( "unregister host memory" );
    features->unpin( );
    nvtxRangePop();
    nvtxRangePop();

    return features;
}

void Pyramid::reset_extrema_mgmt()
{
    memset( &hct,         0, sizeof(ExtremaCounters) );
    cudaMemcpyToSymbol( dct, &hct, sizeof(ExtremaCounters), 0, cudaMemcpyHostToDevice );

    popcuda_memset_sync( _d_extrema_num_blocks, 0, _num_octaves * sizeof(int) );

}

void Pyramid::readDescCountersFromDevice( )
{
    cudaMemcpyFromSymbol( &hct, dct, sizeof(ExtremaCounters), 0, cudaMemcpyDeviceToHost );
}

void Pyramid::readDescCountersFromDevice( cudaStream_t s )
{
    cudaMemcpyFromSymbolAsync( &hct, dct, sizeof(ExtremaCounters), 0, cudaMemcpyDeviceToHost, s );
}

void Pyramid::writeDescCountersToDevice( )
{
    cudaMemcpyToSymbol( dct, &hct, sizeof(ExtremaCounters), 0, cudaMemcpyHostToDevice );
}

void Pyramid::writeDescCountersToDevice( cudaStream_t s )
{
    cudaMemcpyToSymbolAsync( dct, &hct, sizeof(ExtremaCounters), 0, cudaMemcpyHostToDevice, s );
}

int* Pyramid::getNumberOfBlocks( int octave )
{
    return &_d_extrema_num_blocks[octave];
}

void Pyramid::writeDescriptor( const Config& conf, ostream& ostr, Features* features, bool really, bool with_orientation )
{
    if( features->getFeatureCount() == 0 ) return;

    const float up_fac = conf.getUpscaleFactor();

    for( int ext_idx = 0; ext_idx<hct.ext_total; ext_idx++ ) {
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

            const Descriptor& desc  = *ext.desc[ori]; // hbuf.desc[ori_idx];

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
                     // << " 0 "
                     << " " << octave << " "
                     << 1.0f / (sigma * sigma) << " ";

            if (really) {
                for (int i = 0; i<128; i++) {
                    ostr << desc.features[i] << " ";
                }
            }
            ostr << endl;
        }
    }
}


} // namespace popsift
