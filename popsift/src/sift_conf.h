#pragma once

namespace popart
{

struct Config
{
    Config( );

    void setModeVLFeat( float sigma = 0.82f );
    void setModeOpenCV( float sigma = 1.6f );

    enum SiftMode {
        OpenCV,
        VLFeat
    }

    enum LogMode {
        None,
        All
    }

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
    int      levels;
    float    sigma;
    float    edge_limit;
    float    threshold;
    SiftMode sift_mode;
    LogMode  log_log;
};

Config::Config( )
    : start_sampling( -1 )
    , octaves( -1 )
    , levels( 3 )
    , sigma( 1.6f )
    , edge_limit( 10.0f )
    , threshold( 10.0f / 256.0f )
    , sift_mode( Config::OpenCV )
    , log_mode( Config::None )
{ }

void Config::setModeVLFeat( float s )
{
    sift_mode = Config::VLFeat;
    sigma     = s;
}

void Config::setModeOpenCV( float s )
{
    sift_mode = Config::OpenCV;
    sigma     = s;
}

}; // namespace popart

