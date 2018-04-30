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
#include "common/debug_macros.h"
#include "sift_pyramid.h"
#include "sift_extremum.h"
#include "common/assist.h"
#include "features.h"

using namespace std;

PopSift::PopSift( const popsift::Config& config, popsift::Config::ProcessingMode mode, ImageMode imode )
    : _image_mode( imode )
    , _proc_mode( mode )
{
    if( imode == ByteImages )
    {
        _pipe._unused.push( new popsift::Image);
        _pipe._unused.push( new popsift::Image);
    }
    else
    {
        _pipe._unused.push( new popsift::ImageFloat );
        _pipe._unused.push( new popsift::ImageFloat );
    }
    _pipe._pyramid    = 0;

    configure( config, true );

    _pipe._thread_stage1 = new boost::thread( &PopSift::uploadImages, this );
    switch( mode )
    {
    case popsift::Config::ExtractingMode :
        _pipe._thread_stage2 = new boost::thread( &PopSift::extractDownloadLoop, this );
       break;
    case popsift::Config::MatchingMode :
        _pipe._thread_stage2 = new boost::thread( &PopSift::matchPrepareLoop, this );
        break;
    case popsift::Config::RegistrationMode :
        _pipe._thread_stage2 = new boost::thread( &PopSift::registrationPrepareLoop, this );
        break;
    default :
        POP_FATAL( "Programming error: PopSift started with unknown processing moed" );
        break;
    }
}

PopSift::PopSift( ImageMode imode )
    : _image_mode( imode )
{
    if( imode == ByteImages )
    {
        _pipe._unused.push( new popsift::Image);
        _pipe._unused.push( new popsift::Image);
    }
    else
    {
        _pipe._unused.push( new popsift::ImageFloat );
        _pipe._unused.push( new popsift::ImageFloat );
    }
    _pipe._pyramid    = 0;

    _pipe._thread_stage1 = new boost::thread( &PopSift::uploadImages, this );
    _pipe._thread_stage2 = new boost::thread( &PopSift::extractDownloadLoop, this );
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

bool PopSift::private_init( int w, int h )
{
    Pipe& p = _pipe;

    /* up=-1 -> scale factor=2
     * up= 0 -> scale factor=1
     * up= 1 -> scale factor=0.5
     */
    float upscaleFactor = _config.getUpscaleFactor();
    float scaleFactor = 1.0f / powf( 2.0f, -upscaleFactor );

    if( p._pyramid != 0 ) {
        p._pyramid->resetDimensions( _config,
                                     ceilf( w * scaleFactor ),
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
        popsift::ImageBase* img = _pipe._unused.pull();
        delete img;
    }

    delete _pipe._pyramid;
    _pipe._pyramid    = 0;
}

template<class T>
static SiftJob* newJob( int w, int h, const T* imageData, popsift::Config::ProcessingMode mode )
{
    switch( mode )
    {
    case popsift::Config::ExtractingMode :
        return new SiftJob( w, h, imageData );
    case popsift::Config::MatchingMode :
        return new SiftJob( w, h, imageData );
    case popsift::Config::RegistrationMode :
        return new RegistrationJob( w, h, imageData );
    default :
        POP_FATAL( "Programming error: new Job created with unknown processing moed" );
        break;
    }
}

SiftJob* PopSift::enqueue( int                  w,
                           int                  h,
                           const unsigned char* imageData )
{
    if( _image_mode != ByteImages )
    {
        cerr << __FILE__ << ":" << __LINE__ << " Image mode error" << endl
             << "E    Cannot load byte images into a PopSift pipeline configured for float images" << endl;
        exit( -1 );
    }

    SiftJob* job = newJob<unsigned char>( w, h, imageData, _proc_mode );
    _pipe._queue_stage1.push( job );
    return job;
}

SiftJob* PopSift::enqueue( int          w,
                           int          h,
                           const float* imageData )
{
    if( _image_mode != FloatImages )
    {
        cerr << __FILE__ << ":" << __LINE__ << " Image mode error" << endl
             << "E    Cannot load float images into a PopSift pipeline configured for byte images" << endl;
        exit( -1 );
    }

    SiftJob* job = newJob<float>( w, h, imageData, _proc_mode );
    _pipe._queue_stage1.push( job );
    return job;
}

void PopSift::uploadImages( )
{
    SiftJob* job;
    while( ( job = _pipe._queue_stage1.pull() ) != 0 ) {
        popsift::ImageBase* img = _pipe._unused.pull();
        job->setImg( img );
        _pipe._queue_stage2.push( job );
    }
    _pipe._queue_stage2.push( 0 );
}

void PopSift::extractDownloadLoop( )
{
    Pipe& p = _pipe;

    SiftJob* job;
    while( ( job = p._queue_stage2.pull() ) != 0 ) {
        popsift::ImageBase* img = job->getImg();

        private_init( img->getWidth(), img->getHeight() );

        p._pyramid->step1( _config, img );
        p._unused.push( img ); // uploaded input image no longer needed, release for reuse

        p._pyramid->step2( _config );

        popsift::FeaturesHost* features = p._pyramid->get_descriptors( _config );

        cudaDeviceSynchronize();

        bool log_to_file = ( _config.getLogMode() == popsift::Config::All );
        if( log_to_file ) {
            int octaves = p._pyramid->getNumOctaves();

            int levels  = p._pyramid->getNumLevels();

            p._pyramid->download_and_save_array( "pyramid" );
            p._pyramid->save_descriptors( _config, features, "pyramid" );
        }

        job->setFeatures( features );
    }
}

void PopSift::matchPrepareLoop( )
{
    Pipe& p = _pipe;

    SiftJob* job;
    while( ( job = p._queue_stage2.pull() ) != 0 )
    {
        popsift::ImageBase* img = job->getImg();

        private_init( img->getWidth(), img->getHeight() );

        p._pyramid->step1( _config, img );
        p._unused.push( img ); // uploaded input image no longer needed, release for reuse

        p._pyramid->step2( _config );

        popsift::FeaturesDev* features = p._pyramid->clone_device_descriptors( _config );

        cudaDeviceSynchronize();

        job->setFeatures( features );
    }
}

void PopSift::registrationPrepareLoop( )
{
    Pipe& p = _pipe;

    SiftJob* siftjob;
    while( ( siftjob = p._queue_stage2.pull() ) != 0 )
    {
        RegistrationJob* job = dynamic_cast<RegistrationJob*>( siftjob );
        POP_CHECK_NON_NULL( job, "registration loop jobs must have type RegistrationJob" );

        popsift::ImageBase* img = job->getImg();

        private_init( img->getWidth(), img->getHeight() );

        p._pyramid->step1( _config, img );
        p._unused.push( img ); // uploaded input image no longer needed, release for reuse

        p._pyramid->step2( _config );

        popsift::FeaturesDev* features = p._pyramid->clone_device_descriptors( _config );

        cudaDeviceSynchronize();

        job->setPlane( p._pyramid->clone_layer_to_plane2D( 0, 0 ) );

        job->setFeatures( features );
    }
}

