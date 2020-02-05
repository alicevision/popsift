/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <fstream>

#include "popsift.h"
#include "gauss_filter.h"
#include "sift_pyramid.h"

using namespace std;

PopSift::PopSift( const popsift::Config& config, popsift::Config::ProcessingMode mode, ImageMode imode )
    : _image_mode( imode ), _isInit(true)
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
    _pipe._pyramid    = nullptr;

    configure( config, true );

    _pipe._thread_stage1 = new boost::thread( &PopSift::uploadImages, this );
    if( mode == popsift::Config::ExtractingMode )
        _pipe._thread_stage2 = new boost::thread( &PopSift::extractDownloadLoop, this );
    else
        _pipe._thread_stage2 = new boost::thread( &PopSift::matchPrepareLoop, this );
}

PopSift::PopSift( ImageMode imode )
    : _image_mode( imode ), _isInit(true)
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
    _pipe._pyramid    = nullptr;

    _pipe._thread_stage1 = new boost::thread( &PopSift::uploadImages, this );
    _pipe._thread_stage2 = new boost::thread( &PopSift::extractDownloadLoop, this );
}

PopSift::~PopSift()
{
    if(_isInit)
    {
        uninit();
    }
}

bool PopSift::configure( const popsift::Config& config, bool force )
{
    if( _pipe._pyramid != nullptr ) {
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

    if( p._pyramid != nullptr ) {
        p._pyramid->resetDimensions( _config,
                                     ceilf( w * scaleFactor ),
                                     ceilf( h * scaleFactor ) );
        return true;
    }

    if( _config.octaves < 0 ) {
        int oct = max(int (floor( logf( (float)min( w, h ) )
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
    if(!_isInit)
    {
        std::cout << "[warning] Attempt to release resources from an uninitialized instance" << std::endl;
        return;
    }
    _pipe.uninit();

    _isInit = false;
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

    SiftJob* job = new SiftJob( w, h, imageData );
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

    SiftJob* job = new SiftJob( w, h, imageData );
    _pipe._queue_stage1.push( job );
    return job;
}

void PopSift::uploadImages( )
{
    SiftJob* job;
    while( ( job = _pipe._queue_stage1.pull() ) != nullptr ) {
        popsift::ImageBase* img = _pipe._unused.pull();
        job->setImg( img );
        _pipe._queue_stage2.push( job );
    }
    _pipe._queue_stage2.push( nullptr );
}

void PopSift::extractDownloadLoop( )
{
    Pipe& p = _pipe;

    SiftJob* job;
    while( ( job = p._queue_stage2.pull() ) != nullptr ) {
        popsift::ImageBase* img = job->getImg();

        private_init( img->getWidth(), img->getHeight() );

        p._pyramid->step1( _config, img );
        p._unused.push( img ); // uploaded input image no longer needed, release for reuse

        p._pyramid->step2( _config );

        popsift::FeaturesHost* features = p._pyramid->get_descriptors( _config );

        cudaDeviceSynchronize();

        bool log_to_file = ( _config.getLogMode() == popsift::Config::All );
        if( log_to_file ) {
            // int octaves = p._pyramid->getNumOctaves();
            // for( int o=0; o<octaves; o++ ) { p._pyramid->download_descriptors( _config, o ); }
            // int levels  = p._pyramid->getNumLevels();

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
    while( ( job = p._queue_stage2.pull() ) != nullptr ) {
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

SiftJob::SiftJob( int w, int h, const unsigned char* imageData )
    : _w(w)
    , _h(h)
    , _img(nullptr)
{
    _f = _p.get_future();

    _imageData = (unsigned char*)malloc( w*h );
    if( _imageData != nullptr ) {
        memcpy( _imageData, imageData, w*h );
    } else {
        cerr << __FILE__ << ":" << __LINE__ << " Memory limitation" << endl
             << "E    Failed to allocate memory for SiftJob" << endl;
        exit( -1 );
    }
}

SiftJob::SiftJob( int w, int h, const float* imageData )
    : _w(w)
    , _h(h)
    , _img(nullptr)
{
    _f = _p.get_future();

    _imageData = (unsigned char*)malloc( w*h*sizeof(float) );
    if( _imageData != nullptr ) {
        memcpy( _imageData, imageData, w*h*sizeof(float) );
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

void SiftJob::setImg( popsift::ImageBase* img )
{
    img->resetDimensions( _w, _h );
    img->load( _imageData );
    _img = img;
}

popsift::ImageBase* SiftJob::getImg()
{
#ifdef USE_NVTX
    _nvtx_id = nvtxRangeStartA( "inserting image" );
#endif
    return _img;
}

void SiftJob::setFeatures( popsift::FeaturesBase* f )
{
    _p.set_value( f );
#ifdef USE_NVTX
    nvtxRangeEnd( _nvtx_id );
#endif
}

popsift::FeaturesHost* SiftJob::get()
{
    return getHost();
}

popsift::FeaturesBase* SiftJob::getBase()
{
    return _f.get();
}

popsift::FeaturesHost* SiftJob::getHost()
{
    return dynamic_cast<popsift::FeaturesHost*>( _f.get() );
}

popsift::FeaturesDev* SiftJob::getDev()
{
    return dynamic_cast<popsift::FeaturesDev*>( _f.get() );
}

void PopSift::Pipe::uninit()
{
    _queue_stage1.push( nullptr );
    if(_thread_stage2 != nullptr)
    {
        _thread_stage2->join();
        delete _thread_stage2;
        _thread_stage2 = nullptr;
    }
    if(_thread_stage1 != nullptr)
    {
        _thread_stage1->join();
        delete _thread_stage1;
        _thread_stage1 = nullptr;
    }

    while( !_unused.empty() )
    {
        popsift::ImageBase* img = _unused.pull();
        delete img;
    }

    delete _pyramid;
    _pyramid    = nullptr;

}
