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

#include <iostream>
#include <vector>

namespace popsift {

struct ExtremaCounters
{
    /* The number of extrema found per octave */
    int ext_ct[MAX_OCTAVES];
    /* The number of orientation found per octave */
    int ori_ct[MAX_OCTAVES];

    /* Exclusive prefix sum of ext_ct */
    int ext_ps[MAX_OCTAVES];
    /* Exclusive prefix sum of ori_ct */
    int ori_ps[MAX_OCTAVES];

    int ext_total;
    int ori_total;
};

struct ExtremaBuffers
{
    Descriptor*      desc;
    int              ext_allocated;
    int              ori_allocated;
};

struct DevBuffers
{
    InitialExtremum* i_ext_dat[MAX_OCTAVES];
    int*             i_ext_off[MAX_OCTAVES];
    int*             feat_to_ext_map;
    Extremum*        extrema;
    Feature*         features;
};

extern thread_local ExtremaCounters hct;
extern __device__   ExtremaCounters dct;
extern thread_local ExtremaBuffers  hbuf;
extern __device__   ExtremaBuffers  dbuf;
extern thread_local ExtremaBuffers  dbuf_shadow; // just for managing memories
extern __device__   DevBuffers      dobuf;
extern thread_local DevBuffers      dobuf_shadow; // just for managing memories

class Pyramid
{
    int          _num_octaves;
    int          _levels;
    Octave*      _octaves;
    int          _gauss_group;

    /* initial blur variables are used for Gauss table computation,
     * not needed on device */
    bool         _assume_initial_blur;
    float        _initial_blur;

    /* used to implement a global barrier per octave */
    int*         _d_extrema_num_blocks;

    /* the download of converted descriptors should be asynchronous */
    cudaStream_t _download_stream;

public:
    enum GaussTableChoice {
        Interpolated_FromPrevious,
        Interpolated_FromFirst,
        NotInterpolated_FromPrevious,
        NotInterpolated_FromFirst
    };

public:
    Pyramid( const Config& config,
             int     w,
             int     h );
    ~Pyramid( );

    void resetDimensions( const Config& conf, int width, int height );

    /** step 1: load image and build pyramid */
    void step1( const Config& conf, ImageBase* img );

    /** step 2: find extrema, orientations and descriptor */
    void step2( const Config& conf );

    /** step 3: download descriptors */
    FeaturesHost* get_descriptors( const Config& conf );

    /** step 3 (alternative): make copy of descriptors on device side */
    FeaturesDev* clone_device_descriptors( const Config& conf );

    void download_and_save_array( const char* basename );

    void save_descriptors( const Config& conf, FeaturesHost* features, const char* basename );

    inline int getNumOctaves() const { return _num_octaves; }
    inline int getNumLevels()  const { return _levels; }

    inline Octave& getOctave(const int o){ return _octaves[o]; }

private:
    inline void horiz_from_input_image( const Config&    conf,
                                        ImageBase*       base,
					                    int              octave,
					                    cudaStream_t     stream );
    inline void horiz_level_from_input_image( const Config&    conf,
                                              ImageBase*       base,
					                          int              octave,
                                              int              level,
					                          cudaStream_t     stream );
    inline void horiz_all_from_input_image( const Config&    conf,
                                            ImageBase*       base,
                                            int              octave,
                                            int              startlevel,
                                            int              maxlevel,
                                            cudaStream_t     stream );
    inline void downscale_from_prev_octave( int octave, cudaStream_t stream, Config::SiftMode mode );
    inline void horiz_from_prev_level( int octave, int level, cudaStream_t stream, GaussTableChoice useInterpolatedGauss );
    inline void vert_from_interm( int octave, int level, cudaStream_t stream, GaussTableChoice useInterpolatedGauss );
    inline void vert_all_from_interm( int octave,
                                      int start_level,
                                      int max_level,
                                      cudaStream_t stream,
                                      GaussTableChoice useInterpolatedGauss );
    inline void dogs_from_blurred( int octave, int max_level, cudaStream_t stream );

    void make_octave( const Config& conf, ImageBase* base, Octave& oct_obj, cudaStream_t stream, bool isOctaveZero );

    void reset_extrema_mgmt( );
    void build_pyramid( const Config& conf, ImageBase* base );
    void find_extrema( const Config& conf );
    void reallocExtrema( int numExtrema );

    int  extrema_filter_grid( const Config& conf, int ext_total ); // called at head of orientation
    void orientation( const Config& conf );

    void descriptors( const Config& conf );

    void readDescCountersFromDevice( );
    void readDescCountersFromDevice( cudaStream_t s );
    void writeDescCountersToDevice( );
    void writeDescCountersToDevice( cudaStream_t s );
    int* getNumberOfBlocks( int octave );
    void writeDescriptor( const Config& conf, std::ostream& ostr, FeaturesHost* features, bool really, bool with_orientation );

    void clone_device_descriptors_sub( const Config& conf, FeaturesDev* features );

};

} // namespace popsift
