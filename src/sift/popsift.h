#pragma once

#include <cuda_runtime.h>

#include "sift_conf.h"

#define MAX_PIPES 3


/* user parameters */
namespace popart
{
    class Image;
    class Pyramid;
};

class imgStream;

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

    void execute( int pipe, const unsigned char* imageData ); // const imgStream* _inp );

    inline popart::Pyramid& pyramid(int pipe)
    {
        return *_pipe[pipe]._pyramid;
    }

    void uninit( int pipe );

private:
    Pipe           _pipe[MAX_PIPES];
    popart::Config _config;
};

