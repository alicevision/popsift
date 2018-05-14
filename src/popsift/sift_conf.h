/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <string>
#include <iso646.h>

#define MAX_OCTAVES   20
#define MAX_LEVELS    10

#undef USE_DOG_TEX_LINEAR

namespace popsift
{

struct Config
{
    Config( );

    enum GaussMode {
        VLFeat_Compute,
        VLFeat_Relative,
        VLFeat_Relative_All,
        OpenCV_Compute,
        Fixed9,
        Fixed15
    };

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

    /* Modes for descriptor extraction: */
    enum DescMode {
        Loop,        // scan horizontal, extract valid points
        ILoop,       // scan horizontal, extract valid points, interpolate with tex engine
        Grid,        // scan in rotated mode, round pixel address
        IGrid,       // scan in rotated mode, interpolate with tex engine
        NoTile       // variant of IGrid, no duplicate gradiant fetching
    };

    enum NormMode {
        RootSift,   // The L1-inspired norm, gives better matching results
        Classic     // The L2-inspired norm, all descriptors on a hypersphere
    };

    /* To reduce time used in descriptor extraction, some extrema can be filtered
     * immediately after finding them. It is possible to keep those with the largest
     * scale (LargestScaleFirst), smallest scale (SmallestScaleFirst), or a random
     * selection. Note that largest and smallest give a stable result, random does not.
     */
    enum GridFilterMode {
        RandomScale,
        LargestScaleFirst,
        SmallestScaleFirst
    };

    /* A parameter for the PopSift constructor. Determines which data is kept in
     * the Job data structure after processing, which is downloaded to the host,
     * which is invalidated.
     */
    enum ProcessingMode {
        ExtractingMode,
        MatchingMode
    };

    void setGaussMode( const std::string& m );
    void setGaussMode( GaussMode m );
    void setMode( SiftMode m );
    void setLogMode( LogMode mode = All );
    void setScalingMode( ScalingMode mode = ScaleDefault );
    void setVerbose( bool on = true );
    void setDescMode( const std::string& byname );
    void setDescMode( DescMode mode = Loop );

    void setGaussGroup( int groupsize );
    int  getGaussGroup( ) const;

    void setDownsampling( float v );
    void setOctaves( int v );
    void setLevels( int v );
    void setSigma( float v );
    void setEdgeLimit( float v );
    void setThreshold( float v );
    void setInitialBlur( float blur );
    void setMaxExtreme( int m );
    void setPrintGaussTables( );
    void setDPOrientation( bool on );
    void setMaxExtrema( int extrema );
    void setFilterMaxExtrema( int extrema );
    void setFilterGridSize( int sz );
    void setFilterSorting( const std::string& direction );
    void setFilterSorting( GridFilterMode m );

    bool  hasInitialBlur( ) const;
    float getInitialBlur( ) const;

    // computes the actual peak threshold depending on the threshold
    // parameter and the non-augmented number of levels
    float getPeakThreshold() const;

    // print Gauss spans and tables?
    bool ifPrintGaussTables() const;

    // What Gauss filter scan is desired?
    GaussMode getGaussMode( ) const;

    // Call this from the constructor.
    static GaussMode getGaussModeDefault( );

    // Helper functions for the main program's usage string.
    static const char* getGaussModeUsage( );

    // get the SIFT mode for more detailed sub-modes
    SiftMode getSiftMode() const;

    // find out if we should print logging info or not
    LogMode getLogMode() const;

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

    /** Functions related to descriptor normalization: L2-like or RootSift
     */
    void               setNormMode( NormMode m );
    void               setNormMode( const std::string& m );
    void               setNormNode( const std::string& m );
    void               setUseRootSift( bool on ) __attribute__ ((deprecated));
    bool               getUseRootSift( ) const;
    NormMode           getNormMode( NormMode m ) const;
    static NormMode    getNormModeDefault( ); // Call this from the constructor.
    static const char* getNormModeUsage( );  // Helper functions for the main program's usage string.

    /** Functions related to descriptor normalization: multiply with a power of 2
     */
    int  getNormalizationMultiplier( ) const;
    void setNormalizationMultiplier( int mul );

    /* The input image is stretched by 2^upscale_factor
     * before processing. The factor 1 is default.
     */
    inline float getUpscaleFactor( ) const {
        return _upscale_factor;
    }

    int getMaxExtrema( ) const {
        return _max_extrema;
    }

    /* Have we enabled filtering? This is a compile time decision.
     * The reason is that we use Thrust, which increases compile 
     * considerably and can be deactivated at the CMake level when
     * you work on something else.
     */
    bool getCanFilterExtrema() const;

    /* Set the approximate number of extrema whose orientation and descriptor
     * should be computed. Default is -1, which sets the hard limit defined
     * by "number of octaves * getMaxExtrema()".
     */
    int getFilterMaxExtrema( ) const {
        return _filter_max_extrema;
    }

    /* To avoid that grid filtering happens only in a tiny piece of an image,
     * the image is split into getFilterGridSize() X getFilterGridSize() tiles
     * and we allow getFilterMaxExtrema() / getFilterGridSize() extrema in
     * each tile.
     */
    int getFilterGridSize( ) const {
        return _filter_grid_size;
    }

    /* See enum GridFilterMode */
    GridFilterMode getFilterSorting() const {
        return _grid_filter_mode;
    }

    // check if we use direct downscaling from input image
    // for all octaves
    inline ScalingMode getScalingMode() const {
        return _scaling_mode;
    }

    inline DescMode getDescMode() const {
        return _desc_mode;
    }

    bool equal( const Config& other ) const;

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

    // default LogMode::None
    LogMode  _log_mode;

    // default: ScalingMode::DownscaledOctaves
    ScalingMode _scaling_mode;

    // default: DescMode::Loop
    DescMode    _desc_mode;

    // default: RandomScale
    GridFilterMode _grid_filter_mode;

public:
    bool     verbose;

private:
    /* The number of initial extrema that can be discovered in an octave.
     * This parameter changes memory requirements.
     */
    int _max_extrema;

    /* The maximum number of extrema that are returned. There may be
     * several descriptors for each extremum.
     */
    int _filter_max_extrema;

    // Used to achieve an approximation of _max_entrema
    // Subdivide the image in this number of vertical and horizontal tiles,
    // i.e. the grid is actually _grid_size X _grid_size tiles.
    // default: 1
    int  _filter_grid_size;

    /* Modes are computation according to VLFeat or OpenCV,
     * or fixed size. Default is VLFeat mode.
     */
    GaussMode _gauss_mode;

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

    /* OpenMVG requires a normalization named rootSift, the
     * classical L2-inspired mode is also supported.
     */
    NormMode _normalization_mode;

    /* SIFT descriptors are normalized in a final step.
     * The values of the descriptor can also be multiplied
     * by a power of 2 if required.
     * Specify the exponent.
     */
    int _normalization_multiplier;

    /* Call the debug functions in gauss_filter.cu to print Gauss
     * filter width and Gauss tables in use.
     */
    bool _print_gauss_tables;
};

inline bool operator==( const Config& l, const Config& r )
{
    return l.equal( r );
}

inline bool operator!=( const Config& l, const Config& r )
{
    return not l.equal( r );
}

}; // namespace popsift

