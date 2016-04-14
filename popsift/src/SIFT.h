#pragma once

#include <cuda_runtime.h>

#include "c_util_img.h"
#include "s_pyramid.h"
#include "sift_conf.h"

#define MAX_PIPES 3


/* user parameters */

class PopSift
{
    struct Pipe
    {
        popart::Image*   _inputImage;

        popart::Pyramid* _pyramid;
    };

public:
    /* We support more than 1 streams, but we support only one sigma and one
     * level parameters.
     */
    PopSift( popart::Config config );
    ~PopSift();

public:
    bool init( int pipe, int w, int h );

    void execute( int pipe, imgStream _inp );

    void uninit( int pipe );

private:
    Pipe            _pipe[MAX_PIPES];
    popart::Config& _config;

    // int              _init_octaves;    /* number of octaves */
    // const int        _levels;          /* number of levels */
    // const float      _downscale;       /* downscale by 2^this */
    // const float      _sigma;           /* initial sigma */
    // const float      _threshold;       /* DoG threshold */
    // const float      _edgeLimit;       /* edge threshold */
    // const int        _vlfeat_mode;
    // const bool       _log_to_file;
    // const bool       _verbose;
    // const popart::Config::ScalingMode _downscaling_mode;
    // cudaStream_t          _stream;
};

