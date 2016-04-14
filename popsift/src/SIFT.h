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
    PopSift( const popart::Config& config );
    ~PopSift();

public:
    bool init( int pipe, int w, int h );

    void execute( int pipe, imgStream _inp );

    void uninit( int pipe );

private:
    Pipe           _pipe[MAX_PIPES];
    popart::Config _config;
};

