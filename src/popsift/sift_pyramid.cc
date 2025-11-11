/*
* Copyright 2016, Simula Research Laboratory
*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "common/assist.h"
#include "common/grid.h"
#include "common/debug_macros.h"
#include "sift_config.h"
#include "sift_extremum.h"
#include "sift_pyramid.h"

#include <sys/stat.h>

#include <cstdio>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#ifdef _WIN32
#include <direct.h>
#define stat _stat
#define mkdir(path, perm) _mkdir(path)
#endif

#define PYRAMID_PRINT_DEBUG 0

using namespace std;

namespace popsift {

ExtremaCounters dct;
ExtremaBuffers dbuf;

void py_print_corner_float( Grid& g, float* img, uint32_t pitch, uint32_t height, uint32_t level)
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

void py_print_corner_float_transposed( Grid& g, float* img, uint32_t pitch, uint32_t height, uint32_t level)
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

void Pyramid::download_and_save_array( const Config& conf, const char* basename )
{
    POP_INFO2( conf.silent(), "enter " << __PRETTY_FUNCTION__ );
    for( int o=0; o<_num_octaves; o++ )
        _octaves[o].download_and_save_array( conf, basename, o );
}

/*
 * Note this is only for debug output. Features has functions for final writing.
 */
void Pyramid::save_descriptors( const Config& conf, std::unique_ptr<FeaturesHost>& features, const char* basename )
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

sycl::queue Pyramid::initializeQueue()
{
    try {
        // Try GPU first
        sycl::queue q(sycl::gpu_selector_v);
         
        auto device = q.get_device();
        std::cout << "[PopSift] Using GPU: " 
                   << device.get_info<sycl::info::device::name>() 
                   << std::endl;
         
        return q;
    }
    catch (sycl::exception const& e) {
        std::cerr << "[PopSift WARNING] GPU not available: " << e.what() << std::endl;
        std::cerr << "[PopSift] Falling back to CPU..." << std::endl;
         
        try {
            // Fallback to CPU
            sycl::queue q(sycl::cpu_selector_v);
              
            auto device = q.get_device();
            std::cout << "[PopSift] Using CPU: " 
                      << device.get_info<sycl::info::device::name>() 
                      << std::endl;
             
            return q;
        }
        catch (sycl::exception const& e2) {
            std::cerr << "[PopSift ERROR] No SYCL devices available!" << std::endl;
            std::cerr << "[PopSift ERROR] " << e2.what() << std::endl;
            throw std::runtime_error("Failed to initialize SYCL queue: no GPU or CPU available");
        }
    }
}

Pyramid::Pyramid( const Config& config,
                  int width,
                  int height )
    : _num_octaves( config.octaves )
    , _levels( config.levels + 3 )
    , _assume_initial_blur( config.hasInitialBlur() )
    , _initial_blur( config.getInitialBlur() )
    , _shared_queue( initializeQueue() )  // Initialize with fallback
{
    _octaves = new Octave[_num_octaves];

    int w = width;
    int h = height;

    dct = ExtremaCounters();
    dbuf = ExtremaBuffers();

    _d_extrema_num_blocks = new int[_num_octaves];

    for (int o = 0; o<_num_octaves; o++) {
        _octaves[o].debugSetOctave(o);
        _octaves[o].alloc( config, w, h, _levels, _shared_queue ); 
        w = ceilf(w / 2.0f);
        h = ceilf(h / 2.0f);
    }

    int sz = _num_octaves * h_consts.max_extrema;

    // dbuf.i_ext_dat[0] = new InitialExtremum[sz];
    // dbuf.i_ext_off[0] = new int[sz];
    //
    // for (int o = 1; o<_num_octaves; o++) {
        // dbuf.i_ext_dat[o] = dbuf.i_ext_dat[0] + (o*h_consts.max_extrema);
        // dbuf.i_ext_off[o] = dbuf.i_ext_off[0] + (o*h_consts.max_extrema);
    // }
    // for (int o = _num_octaves; o<MAX_OCTAVES; o++) {
        // dbuf.i_ext_dat[o] = nullptr;
        // dbuf.i_ext_off[o] = nullptr;
    // }

    sz = h_consts.max_extrema;
    // dbuf.extrema  = new Extremum[sz];
    dbuf.features = new Feature[sz];
    dbuf.ext_allocated = sz;

    sz = max( 2 * h_consts.max_extrema, h_consts.max_orientations );
    dbuf.desc            = new Descriptor[sz];
    // dbuf.feat_to_ext_map = new int[sz];
    dbuf.ori_allocated   = sz;
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
    if( numExtrema > dbuf.ext_allocated ) {
        numExtrema = ( ( numExtrema + 1024 ) & ( ~(1024-1) ) );

        // delete [] dbuf.extrema;
        dbuf.extrema.clear();

        delete [] dbuf.features;

        int sz = numExtrema;
        // dbuf.extrema  = new Extremum[sz];
        dbuf.features = new Feature[sz];
        dbuf.ext_allocated = sz;

        numExtrema *= 2;
        if( numExtrema > dbuf.ori_allocated ) {
            delete [] dbuf.desc;
            dbuf.feat_to_ext_map.clear(); // delete [] dbuf.feat_to_ext_map;

            sz = numExtrema;
            dbuf.desc            = new Descriptor[sz];
            // dbuf.feat_to_ext_map = new int[sz];
            dbuf.ori_allocated = sz;
        }
    }
}

Pyramid::~Pyramid()
{    
    // Free extrema management
    if(_d_extrema_num_blocks) {
        delete[] _d_extrema_num_blocks;
        _d_extrema_num_blocks = nullptr;
    }

    // Free descriptor buffers
    if(dbuf.features) {
        delete[] dbuf.features;
        dbuf.features = nullptr;
    }
    
    if(dbuf.desc) {
        delete[] dbuf.desc;
        dbuf.desc = nullptr;
    }

    // Clear vectors
    dbuf.feat_to_ext_map.clear();
    dbuf.extrema.clear();

    // Free all octaves - THIS IS WHERE THE 7.2MB LEAK IS!
    if(_octaves) {
        for(int o = 0; o < _num_octaves; o++) {
            _octaves[o].free();
        }
        delete[] _octaves;
        _octaves = nullptr;
    }
   
}

void Pyramid::step1( const Config& conf, std::shared_ptr<popsift::ImageBase> img )
{
    POP_INFO2( conf.silent(), "enter " << __PRETTY_FUNCTION__ );

    reset_extrema_mgmt( );

    POP_INFO2( conf.silent(), "is image NULL? " << (img->isNull() ? "yes" : "no") );

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
void prep_features(Descriptor* descriptor_base, int up_fac )
{
    printf("Prep features called\n");
    if (!dbuf.features) {
        std::cerr << "[ERROR] dbuf.features is not allocated!" << std::endl;
        return;
    }

    for (int offset = 0; offset < dct.extrema_count_total; ++offset) {
        const Extremum& ext = dbuf.extrema[offset];
        Feature& fet = dbuf.features[offset];

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
        for( ori = 0; ori < num_ori; ori++ ) {
            fet.desc[ori]        = descriptor_base + ( ext.idx_ori + ori );
            fet.orientation[ori] = ext.orientation[ori];
        }
        for( ; ori < ORIENTATION_MAX_COUNT; ori++ ) {
            fet.desc[ori]        = nullptr;
            fet.orientation[ori] = 0;
        }
    }
}

std::unique_ptr<FeaturesHost> Pyramid::get_descriptors( const Config& conf )
{
    const float up_fac = conf.getUpscaleFactor();

    std::unique_ptr<FeaturesHost> features( new FeaturesHost( dct.extrema_count_total, dct.ori_total ) );

    if( dct.extrema_count_total == 0 || dct.ori_total == 0 )
    {
        return features;
    }


    prep_features(features->getDescriptors(), up_fac );

    memcpy( features->getFeatures(),
            dbuf.features,
            dct.extrema_count_total * sizeof(Feature) );

    memcpy( features->getDescriptors(),
            dbuf.desc,
            dct.ori_total * sizeof(Descriptor) );

    return features;
}

void Pyramid::clone_device_descriptors_sub( const Config& conf, std::unique_ptr<FeaturesHost>& features )
{
    const float up_fac = conf.getUpscaleFactor();

    prep_features( features->getDescriptors(), up_fac );

    memcpy( features->getFeatures(),
            dbuf.features,
            dct.extrema_count_total * sizeof(Feature) );

    memcpy( features->getDescriptors(),
            dbuf.desc,
            dct.ori_total * sizeof(Descriptor) );

#if 1
    if( dbuf.feat_to_ext_map.size() != dct.ori_total )
    {
        std::cerr << "feature to ext map has bad size " << dbuf.feat_to_ext_map.size()
                  << ", should be number of orientations " << dct.ori_total
                  << std::endl;
        exit( -1 );
    }
#endif
    features->setReverseMap( dbuf.feat_to_ext_map );
    // memcpy( features->getReverseMap(), dbuf.feat_to_ext_map, dct.ori_total * sizeof(int) );
}

std::unique_ptr<FeaturesHost> Pyramid::clone_device_descriptors( const Config& conf )
{
    std::unique_ptr<FeaturesHost> features( new FeaturesHost( dct.extrema_count_total, dct.ori_total ) );

    clone_device_descriptors_sub( conf, features );

    return features;
}

void Pyramid::reset_extrema_mgmt()
{
    dct = ExtremaCounters();
    memset( _d_extrema_num_blocks, 0, _num_octaves * sizeof(int) );

}

int* Pyramid::getNumberOfBlocks( int octave )
{
    return &_d_extrema_num_blocks[octave];
}

/*
 * Note this is only for debug output. FeaturesHost has functions for final writing.
 */
void Pyramid::writeDescriptor( const Config& conf, ostream& ostr, std::unique_ptr<FeaturesHost>& features, bool really, bool with_orientation )
{
    if( features->getFeatureCount() == 0 ) return;

    const float up_fac = conf.getUpscaleFactor();

    for( int ext_idx = 0; ext_idx<dct.extrema_count_total; ext_idx++ ) {
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
