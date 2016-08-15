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
    PopSift( const popsift::Config& config );
    ~PopSift();

public:
    bool init( int pipe, int w, int h, bool checktime = false );

    void execute( int                                            pipe,
                  const unsigned char*                           imageData,
                  std::vector<std::vector<popsift::Extremum> >*   extrema = 0,
                  std::vector<std::vector<popsift::Descriptor> >* descs = 0,
                  bool                                           checktime = false );

    inline popsift::Pyramid& pyramid(int pipe)
    {
        return *_pipe[pipe]._pyramid;
    }

    void uninit( int pipe );

private:
    Pipe           _pipe[MAX_PIPES];
    popsift::Config _config;
};

