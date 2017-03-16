/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cuda_runtime.h>
#include <vector>

#include "sift_conf.h"
#include "sift_extremum.h"

#define MAX_PIPES 3


/* user parameters */
namespace popsift
{
    class Image;
    class Pyramid;
    class Features;
};

class PopSift
{
    struct Pipe
    {
        popsift::Image*   _inputImage;

        popsift::Pyramid* _pyramid;
    };

public:
    /* We support more than 1 streams, but we support only one sigma and one
     * level parameters.
     */
    PopSift( );
    PopSift( const popsift::Config& config );
    ~PopSift();

public:
    /* provide the configuration if you used the PopSift constructor */
    bool configure( const popsift::Config& config );

    bool init( int pipe, int w, int h, bool checktime = false );

    void uninit( int pipe );

#if 1
    popsift::Features* execute( int                  pipe,
                                const unsigned char* imageData,
                                bool                 checktime = false );
#else
    void execute( int                                            pipe,
                  const unsigned char*                           imageData,
                  std::vector<std::vector<popsift::Extremum> >*   extrema = 0,
                  std::vector<std::vector<popsift::Descriptor> >* descs = 0,
                  bool                                           checktime = false );
#endif

    inline popsift::Pyramid& pyramid(int pipe)
    {
        return *_pipe[pipe]._pyramid;
    }

private:
    Pipe            _pipe[MAX_PIPES];
    popsift::Config _config;
};

