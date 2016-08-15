/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

namespace popsift
{

struct Config
{
    Config( );

    enum SiftMode {
        PopSift,
        OpenCV,
        VLFeat,
        Default = PopSift
    };

    enum LogMode {
        None,
        All
    };

    enum ScalingMode {
        ScaleDirect,
        ScaleDefault // Indirect - only working method
    };

    void setMode( SiftMode m );
    void setLogMode( LogMode mode = All );
    void setScalingMode( ScalingMode mode = ScaleDefault );
    void setVerbose( bool on = true );

    void setGaussGroup( int groupsize );
    int  getGaussGroup( ) const;

    void setDownsampling( float v );
    void setOctaves( int v );
    void setLevels( int v );
    void setSigma( float v );
    void setEdgeLimit( float v );
    void setThreshold( float v );
    void setInitialBlur( float blur );
    void setPrintGaussTables( );
    void setDPOrientation( bool on );
    void setDPDescriptors( bool on );

    bool  hasInitialBlur( ) const;
    float getInitialBlur( ) const;

    // computes the actual peak threshold depending on the threshold
    // parameter and the non-augmented number of levels
    float getPeakThreshold() const;

    // print Gauss spans and tables?
    bool ifPrintGaussTables() const;

    // get the SIFT mode for more detailed sub-modes
    SiftMode getSiftMode() const;

    // The number of octaves is chosen freely. If not specified,
    // it is: log_2( min(x,y) ) - 3 - start_sampling
    int      octaves;

    // The number of levels per octave. This is actually the
    // number of inner DoG levels where we can search for
    // feature points. The number of ...
    //
    // This is the non-augmented number of levels, meaning
    // the this is not the number of gauss-filtered picture
    // layers (which is levels+3), but the number of DoG
    // layers in which we can search for extrema.
    int      levels;
    float    sigma;

    // default edge_limit 16.0f from Celebrandil
    // default edge_limit 10.0f from Bemap
    float    _edge_limit;

    /* The input image is stretched by 2^upscale_factor
     * before processing. The factor 1 is default.
     */
    float getUpscaleFactor( ) const {
        return _upscale_factor;
    }

    bool useDPOrientation( ) const {
        return ( _dp_capable & _dp_orientation );
    }

    bool useDPDescriptors( ) const {
        return ( _dp_capable & _dp_descriptors );
    }

private:
    // default threshold 0.0 default of vlFeat
    // default threshold 5.0 / 256.0
    // default threshold 15.0 / 256.0 - it seems our DoG is really small ???
    // default threshold 5.0 from Celebrandil, not happening in our data
    // default threshold 0.04 / (_levels-3.0) / 2.0f * 255
    //                   from Bemap -> 1.69 (makes no sense)
    float    _threshold;

    // determine the image format of the first octave
    // relative to the input image's size (x,y) as follows:
    // (x / 2^start_sampling, y / 2^start_sampling )
    float    _upscale_factor;

public:
    // default LogMode::None
    LogMode  log_mode;

    // default: ScalingMode::DownscaledOctaves
    ScalingMode scaling_mode;

    bool     verbose;

    /* A single Gauss filtering step filters for several levels at once,
     * saving load operations. Accuracy is maintained only for very small
     * numbers.
     */
    int gauss_group_size;

private:
    /* Modes are PopSift, OpenCV and VLFeat.
     * Default is currently identical to PopSift.
     */
    SiftMode _sift_mode;

    /* VLFeat code assumes that an initial input image is partially blurred.
     * This changes the blur computation for the very first level of the first
     * octave, turning it into a special case.
     */
    bool  _assume_initial_blur;
    float _initial_blur;

    /* Call the debug functions in gauss_filter.cu to print Gauss
     * filter width and Gauss tables in use.
     */
    bool _print_gauss_tables;

    /* When these variables are true, CUDA Dynamic Parallelism can be used
     * in the specified run-time stage.
     */
    bool _dp_orientation;
    bool _dp_descriptors;

    /* If the chosen GPU has compute capability below 3.5, then the previous
     * configuration flags are always false.
     */
    bool _dp_capable;
};

}; // namespace popsift

