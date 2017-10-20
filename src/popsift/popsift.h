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
#include <stack>
#include <queue>
#include <future>
#include <boost/thread/thread.hpp>
#include <boost/thread/sync_queue.hpp>

#include "sift_conf.h"
#include "sift_extremum.h"


/* user parameters */
namespace popsift
{
    class Image;
    class Pyramid;
    class Features;
};

class SiftJob
{
    std::promise<popsift::Features*> _p;
    std::future <popsift::Features*> _f;
    int             _w;
    int             _h;
    unsigned char*  _imageData;
    popsift::Image* _img;

public:
    SiftJob( int w, int h, const unsigned char* imageData );
    ~SiftJob( );

    popsift::Features* get() {
        return _f.get();
    }

    void setImg( popsift::Image* img );
    inline popsift::Image* getImg() const { return _img; }

    /** fulfill the promise */
    void setFeatures( popsift::Features* f );
};

class PopSift
{
    struct Pipe
    {
        boost::thread*                     _thread_stage1;
        boost::thread*                     _thread_stage2;
        boost::sync_queue<SiftJob*>        _queue_stage1;
        boost::sync_queue<SiftJob*>        _queue_stage2;
        boost::sync_queue<popsift::Image*> _unused;
        popsift::Image*                    _current;

        popsift::Pyramid*                  _pyramid;
    };

public:
    /* We support more than 1 streams, but we support only one sigma and one
     * level parameters.
     */
    PopSift( );
    PopSift( const popsift::Config& config, popsift::Config::ProcessingMode mode = popsift::Config::ExtractingMode );
    ~PopSift();

public:
    /** Provide the configuration if you used the PopSift default
     *  constructor */
    bool configure( const popsift::Config& config, bool force = false );

    void uninit( );

    SiftJob*  enqueue( int                  w,
                       int                  h,
                       const unsigned char* imageData );

    /** deprecated */
    inline void uninit( int /*pipe*/ ) { uninit(); }

    /** deprecated */
    inline bool init( int /*pipe*/, int w, int h ) {
        _last_init_w = w;
        _last_init_h = h;
        return true;
    }

    /** deprecated */
    inline popsift::Features* execute( int /*pipe*/, const unsigned char* imageData )
    {
        SiftJob* j = enqueue( _last_init_w, _last_init_h, imageData );
        if( !j ) return 0;
        popsift::Features* f = j->get();
        delete j;
        return f;
    }

private:
    bool private_init( int w, int h );
    void uploadImages( );

    /* The following method are alternative worker functions for Jobs submitted by
     * a calling application. The choice of method is made by the mode parameter
     * in the PopSift constructor. */

    /* Worker function: Extract SIFT features and download to host */
    void extractDownloadLoop( );

    /* Worker function: Extract SIFT features, clone results in device memory */
    void matchPrepareLoop( );

private:
    Pipe            _pipe;
    popsift::Config _config;

    /* Keep a copy of the config to avoid unnecessary re-configurations
     * in configure()
     */
    popsift::Config _shadow_config;

    int             _last_init_w; /* to support depreacted interface */
    int             _last_init_h; /* to support depreacted interface */
};

