/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "features.h"
#include "s_image.h"
#include "sift_conf.h"
#include "sift_constants.h"
#include "sift_octave.h"

#include <memory>
#include <iostream>
#include <vector>

namespace popsift {

struct ExtremaCounters
{
    /* BEGIN filled in s_extrema.cc */

    std::vector<InitialExtremum> initial_extrema_in_octave[MAX_OCTAVES];
    std::vector<int>             initial_extrema_offset   [MAX_OCTAVES];

    /* The number of extrema found per octave.
     * Length is initialized to MAX_OCTAVES in Pyramid::find_extrema() */
    std::vector<int> extrema_count_per_octave;
    /* Exclusive prefix sum of extrema_count_per_octave, for later index computations.
     * Final element added for the total sum.
     * Gets it's size during computation of the prefix sum. */
    std::vector<int> extrema_count_prefix_sum;
    /* Number of all extrema found */
    int extrema_count_total;

    /* END filled in s_extrema.cc */

    /* The number of orientation found per octave */
    int ori_ct[MAX_OCTAVES];

    /* Exclusive prefix sum of ori_ct */
    int ori_ps[MAX_OCTAVES];

    int ori_total;
};

struct ExtremaBuffers
{
    Descriptor*      desc;
    int              ext_allocated;
    int              ori_allocated;

    Feature*              features;
    std::vector<Extremum> extrema;
    std::vector<int>      feat_to_ext_map;
    // int*                  feat_to_ext_map;
};

extern ExtremaCounters dct;
extern ExtremaBuffers  dbuf;

class Pyramid
{
    int          _num_octaves;
    int          _levels;
    Octave*      _octaves;

    /* initial blur variables are used for Gauss table computation,
     * not needed on device */
    bool         _assume_initial_blur;
    float        _initial_blur;

    /* used to implement a global barrier per octave */
    int*         _d_extrema_num_blocks;

    /* Storage for device pointers during async extrema detection */
    std::vector<std::pair<InitialExtremum*, int*>> _device_extrema_ptrs;

public:
    enum GaussTableChoice {
        Interpolated_FromPrevious,
        NotInterpolated_FromPrevious,
    };

public:
    Pyramid( const Config& config,
             int     w,
             int     h );
    ~Pyramid( );

    void resetDimensions( const Config& conf, int width, int height );

    /** step 1: load image and build pyramid */
    void step1( const Config& conf, std::shared_ptr<ImageBase> img );

    /** step 2: find extrema, orientations and descriptor */
    void step2( const Config& conf );

    /** step 3: download descriptors */
    std::unique_ptr<FeaturesHost> get_descriptors( const Config& conf );

    /** step 3 (alternative): make copy of descriptors on device side */
    std::unique_ptr<FeaturesHost> clone_device_descriptors( const Config& conf );

    void download_and_save_array( const Config& conf, const char* basename );

    void save_descriptors( const Config& conf, std::unique_ptr<FeaturesHost>& features, const char* basename );

    inline int getNumOctaves() const { return _num_octaves; }
    inline int getNumLevels()  const { return _levels; }

    inline Octave& getOctave(const int o){ return _octaves[o]; }

   /* Store device pointers for async extrema detection cleanup */
   void storeDevicePointers(int octave, InitialExtremum* d_extrema, int* d_count) {
       if(_device_extrema_ptrs.size() <= (size_t)octave) {
           _device_extrema_ptrs.resize(octave + 1);
       }
       _device_extrema_ptrs[octave] = {d_extrema, d_count};
   }
   
   /* Retrieve device pointers for processing results */
   std::pair<InitialExtremum*, int*> getDevicePointers(int octave) {
       return _device_extrema_ptrs[octave];
   }

private:
    void horiz_from_input_image( const Config&              conf,
                                 std::shared_ptr<ImageBase> base );
    void downscale_from_prev_octave( int octave );

    void horiz_from_prev_level( int octave, int level );
    void vert_from_interm( int octave, int level );

    sycl::event dogs_from_blurred( int octave, int max_level );

    void reset_extrema_mgmt( );
    void build_pyramid( const Config& conf, std::shared_ptr<ImageBase> base );
    void find_extrema( const Config& conf );
    void reallocExtrema( int numExtrema );

    int  extrema_filter_grid( const Config& conf, int ext_total ); // called at head of orientation
    void orientation( const Config& conf );

    void descriptors( const Config& conf );

    int* getNumberOfBlocks( int octave );
    void writeDescriptor( const Config& conf, std::ostream& ostr,
                          std::unique_ptr<FeaturesHost>& features,
                          bool really, bool with_orientation );

    void clone_device_descriptors_sub( const Config& conf, std::unique_ptr<FeaturesHost>& features );

};

} // namespace popsift
