#pragma once

namespace popart
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
        DirectDownscaling,
        IndirectUnfilteredDownscaling
    };

    void setMode( SiftMode m );
    void setLogMode( LogMode mode = All );
    void setScalingMode( ScalingMode mode = IndirectUnfilteredDownscaling );
    void setVerbose( bool on = true );

    void setBemapOrientation( );
    bool getBemapOrientation( ) const;

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

    bool  hasInitialBlur( ) const;
    float getInitialBlur( ) const;

    // computes the actual peak threshold depending on the threshold
    // parameter and the non-augmented number of levels
    float getPeakThreshold() const;

    // print Gauss spans and tables?
    bool ifPrintGaussTables() const;

    // get the SIFT mode for more detailed sub-modes
    SiftMode getSiftMode() const;

    // determine the image format of the first octave
    // relative to the input image's size (x,y) as follows:
    // (x / 2^start_sampling, y / 2^start_sampling )
    float    start_sampling;

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

private:
    // default threshold 0.0 default of vlFeat
    // default threshold 5.0 / 256.0
    // default threshold 15.0 / 256.0 - it seems our DoG is really small ???
    // default threshold 5.0 from Celebrandil, not happening in our data
    // default threshold 0.04 / (_levels-3.0) / 2.0f * 255
    //                   from Bemap -> 1.69 (makes no sense)
    float    _threshold;

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

    /* The first orientation code was derived from Bemap OpenCL SIFT.
     * It used double smoothing of the orientation histogram in s_ori.
     * This has been replaced by direct curve fitting according to Lowe.
     * Set this to true for old mode.
     */
    bool _bemap_orientation;

    /* Call the debug functions in gauss_filter.cu to print Gauss
     * filter width and Gauss tables in use.
     */
    bool _print_gauss_tables;
};

}; // namespace popart

