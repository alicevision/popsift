/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "features.h"
#include "filtergrid.h"
#include "s_image.h"
#include "sift_conf.h"
#include "sift_constants.h"
#include "sift_octave.h"
#include "initial_extremum.h"

#include <iostream>
#include <vector>

namespace popsift {

/** Datastructure in managed memory that holds counters for
 *  initially collected extrema and orientations.
 */
struct ExtremaCounters
{
    /* The number of extrema found per octave */
    int ext_ct[MAX_OCTAVES];
    /* The number of orientation found per octave */
    int ori_ct[MAX_OCTAVES];

private:
    /* Exclusive prefix sum of ext_ct */
    int ext_ps[MAX_OCTAVES+1];
    /* Exclusive prefix sum of ori_ct */
    int ori_ps[MAX_OCTAVES+1];

public:
    /** host and device function helper function that updates the exclusive
     *  prefix sum for extrema counts, updates ext_total and returns the total
     *  number of extrema.
     *  Note that exclusive prefix sum on the device could use several threads
     *  but we are talking about a constant size of 20.
     */
    __device__ __host__ inline
    int make_extrema_prefix_sums( )
    {
        ext_ps[0] = 0;
        for( int o=1; o<=MAX_OCTAVES; o++ ) {
            ext_ps[o] = ext_ps[o-1] + ext_ct[o-1];
        }

        return ext_ps[MAX_OCTAVES];
    }

    /** get total number of extrema */
    __device__ __host__ inline
    int getTotalExtrema( ) const
    {
        return ext_ps[MAX_OCTAVES];
    }

    /** in a sorted array of extrema, get the base index for the entries
     *  of this octave's extrema */
    __device__ __host__ inline
    int getExtremaBase( const int& octave ) const
    {
        return ext_ps[octave];
    }

    /** compute the prefix sum and total sum of orientation count per octave */
    __device__ __host__ inline
    int make_orientation_prefix_sums( )
    {
        ori_ps[0] = 0;
        for( int o=1; o<=MAX_OCTAVES; o++ ) {
            ori_ps[o] = ori_ps[o-1] + ori_ct[o-1];
        }

        return ori_ps[MAX_OCTAVES];
    }

    /** get total number of orientations */
    __device__ __host__ inline
    int getTotalOrientations( ) const
    {
        return ori_ps[MAX_OCTAVES];
    }

    /** in a sorted array of orientations, get the base index for the entries
     *  of this octave's orientations */
    __device__ __host__ inline
    int getOrientationBase( const int& octave ) const
    {
        return ori_ps[octave];
    }
};

/** Datastructure in managed memory that allows CPU and GPU to access all
 *  buffers.
 */
struct ExtremaBuffers
{
    /* This part of the struct deals with the descriptors that are
     * finally detected by the algorithm.
     */
    Descriptor*      desc;
    int              ext_allocated;
    int              ori_allocated;

    /* This part of the struct deals with intermediate buffers to find
     * extrema.
     */
    InitialExtremum* i_ext_dat[MAX_OCTAVES];
    int*             i_ext_off[MAX_OCTAVES];
    int*             feat_to_ext_map;
    Extremum*        extrema;
    Feature*         features;

    /** Allocate buffers that hold intermediate and final information about
     *  extrema and if applicable, orientations and descriptors.
     *  Buffers are located in managed memory.
     */
    void init( int num_octave, int max_extrema, int max_orientations );

    /** Release all buffers */
    void uninit( );

    /** Function that allows to resize buffers when all extrema have been
     *  detected and the resulting total is too large.
     */
    void growBuffers( int numExtrema );
};

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

    ExtremaCounters* _ct;
    ExtremaBuffers*  _buf;

    /** A structure that encapsulates everything we need for grid filtering */
    FilterGrid _fg;

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
