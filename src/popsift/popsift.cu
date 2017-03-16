/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <fstream>
#include <pthread.h> // for pthread_self

#include "sift_constants.h"
#include "popsift.h"
#include "gauss_filter.h"
#include "common/write_plane_2d.h"
#include "sift_pyramid.h"
#include "sift_extremum.h"
#include "common/assist.h"

using namespace std;

PopSift::PopSift( const popsift::Config& config )
{
    _pipe._unused.push( new popsift::Image );
    _pipe._unused.push( new popsift::Image );
    _pipe._pyramid    = 0;

    configure( config, true );

    _pipe._thread_stage1 = new boost::thread( &PopSift::uploadImages, this );
    _pipe._thread_stage2 = new boost::thread( &PopSift::execute,      this );
}

PopSift::PopSift( )
{
    _pipe._unused.push( new popsift::Image );
    _pipe._unused.push( new popsift::Image );
    _pipe._pyramid    = 0;

    _pipe._thread_stage1 = new boost::thread( &PopSift::uploadImages, this );
    _pipe._thread_stage2 = new boost::thread( &PopSift::execute,      this );
}

PopSift::~PopSift()
{
}

bool PopSift::configure( const popsift::Config& config, bool force )
{
    if( _pipe._pyramid != 0 ) {
        return false;
    }

    _config = config;

    _config.levels = max( 2, config.levels );

    if( force || ( _config  != _shadow_config ) )
    {
        popsift::init_filter( _config,
                              _config.sigma,
                              _config.levels );
        popsift::init_constants(  _config.sigma,
                                  _config.levels,
                                  _config.getPeakThreshold(),
                                  _config._edge_limit,
                                  _config.getMaxExtrema(),
                                  _config.getNormalizationMultiplier() );
    }
    _shadow_config = _config;
    return true;
}

bool PopSift::init( int w, int h )
{
    Pipe& p = _pipe;

    /* up=-1 -> scale factor=2
     * up= 0 -> scale factor=1
     * up= 1 -> scale factor=0.5
     */
    float upscaleFactor = _config.getUpscaleFactor();
    float scaleFactor = 1.0f / powf( 2.0f, -upscaleFactor );

    if( p._pyramid != 0 ) {
        p._pyramid->resetDimensions( ceilf( w * scaleFactor ),
                                     ceilf( h * scaleFactor ) );
        return true;
    }

    if( _config.octaves < 0 ) {
        int oct = _config.octaves;
        oct = max(int (floor( logf( (float)min( w, h ) )
                            / logf( 2.0f ) ) - 3.0f + scaleFactor ), 1);
        _config.octaves = oct;
    }

    p._pyramid = new popsift::Pyramid( _config,
                                       ceilf( w * scaleFactor ),
                                       ceilf( h * scaleFactor ) );

    cudaDeviceSynchronize();

    return true;
}

void PopSift::uninit( )
{
    _pipe._queue_stage1.push( 0 );
    _pipe._thread_stage2->join();
    _pipe._thread_stage1->join();
    delete _pipe._thread_stage2;
    delete _pipe._thread_stage1;

    while( !_pipe._unused.empty() ) {
        popsift::Image* img = _pipe._unused.pull();
        delete img;
    }

    delete _pipe._pyramid;
    _pipe._pyramid    = 0;
}

SiftJob* PopSift::enqueue( int                  w,
                           int                  h,
                           const unsigned char* imageData )
{
    SiftJob* job = new SiftJob( w, h, imageData );
    _pipe._queue_stage1.push( job );
    return job;
}

void PopSift::uploadImages( )
{
    SiftJob* job;
    while( ( job = _pipe._queue_stage1.pull() ) != 0 ) {
        popsift::Image* img = _pipe._unused.pull();
        job->setImg( img );
        _pipe._queue_stage2.push( job );
    }
    _pipe._queue_stage2.push( 0 );
}

void PopSift::execute( )
{
    Pipe& p = _pipe;

    SiftJob* job;
    while( ( job = p._queue_stage2.pull() ) != 0 ) {
        popsift::Image* img = job->getImg();

        init( img->getWidth(), img->getHeight() );

        p._pyramid->step1( _config, img );
        p._unused.push( img );

        p._pyramid->step2( _config );

        popsift::Features* features = p._pyramid->get_descriptors( _config );

        cudaDeviceSynchronize();

        bool log_to_file = ( _config.getLogMode() == popsift::Config::All );
        if( log_to_file ) {
            int octaves = p._pyramid->getNumOctaves();

            // for( int o=0; o<octaves; o++ ) { p._pyramid->download_descriptors( _config, o ); }

            int levels  = p._pyramid->getNumLevels();

            p._pyramid->download_and_save_array( "pyramid" );
            p._pyramid->save_descriptors( _config, features, "pyramid" );
        }

        job->setFeatures( features );
    }
}

SiftJob::SiftJob( int w, int h, const unsigned char* imageData )
    : _w(w)
    , _h(h)
    , _img(0)
{
    _f = _p.get_future();

    _imageData = (unsigned char*)malloc( w*h );
    if( _imageData != 0 ) {
        memcpy( _imageData, imageData, w*h );
    } else {
        cerr << __FILE__ << ":" << __LINE__ << " Memory limitation" << endl
             << "E    Failed to allocate memory for SiftJob" << endl;
        exit( -1 );
    }
}

SiftJob::~SiftJob( )
{
    delete [] _imageData;
}

void SiftJob::setImg( popsift::Image* img )
{
    img->resetDimensions( _w, _h );
    img->load( _imageData );
    _img = img;
}

void SiftJob::setFeatures( popsift::Features* f )
{
    _p.set_value( f );
}

