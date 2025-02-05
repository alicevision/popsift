/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cmath>
#include <cstring>
#include "popsift.h"

#include "gauss_filter.h"
#include "sift_config.h"
#include "sift_pyramid.h"
#include "common/debug_macros.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace std;

PopSift::PopSift( const popsift::Config& config, popsift::Config::ProcessingMode mode, ImageMode imode )
    : _image_mode( imode )
{
    configure(config);

    if( imode == ByteImages )
    {
        _pipe._unused.push( new popsift::Image );
        _pipe._unused.push( new popsift::Image );
    }
    else
    {
        _pipe._unused.push( new popsift::ImageFloat );
        _pipe._unused.push( new popsift::ImageFloat );
    }

    _pipe._thread_stage1.reset( new std::thread( &PopSift::uploadImages, this ));
    if( mode == popsift::Config::ExtractingMode )
        _pipe._thread_stage2.reset( new std::thread( &PopSift::extractDownloadLoop, this ));
    else
        _pipe._thread_stage2.reset( new std::thread( &PopSift::matchPrepareLoop, this ));
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

    _pipe._thread_stage1.reset( new std::thread( &PopSift::uploadImages, this ));
    _pipe._thread_stage2.reset( new std::thread( &PopSift::extractDownloadLoop, this ));
}

PopSift::~PopSift()
{
    if(_isInit)
    {
        uninit();
    }
}

bool PopSift::configure( const popsift::Config& config, bool /*force*/ )
{
    if( _pipe._pyramid != nullptr ) {
        return false;
    }

    _config = config;
    _config.levels = max( 2, config.levels );

    return true;
}

bool PopSift::applyConfiguration(bool force)
{
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

void PopSift::private_apply_scale_factor( int& w, int& h )
{
    /* up=-1 -> scale factor=2
     * up= 0 -> scale factor=1
     * up= 1 -> scale factor=0.5
     */
    float upscaleFactor = _config.getUpscaleFactor();
    float scaleFactor = 1.0f / powf( 2.0f, -upscaleFactor );

    if( _config.octaves < 0 ) {
        int oct = max(int (floor( logf( (float)min( w, h ) )
                            / logf( 2.0f ) ) - 3.0f + scaleFactor ), 1);
        _config.octaves = oct;
    }

    w = ceilf( w * scaleFactor );
    h = ceilf( h * scaleFactor );
}

bool PopSift::private_init( int w, int h )
{
    Pipe& p = _pipe;

    private_apply_scale_factor( w, h );

    if( p._pyramid != nullptr ) {
        p._pyramid->resetDimensions( _config, w, h );
        return true;
    }

    p._pyramid = new popsift::Pyramid( _config, w, h );

    cudaDeviceSynchronize();

    return true;
}

bool PopSift::private_uninit()
{
    Pipe& p = _pipe;

    delete p._pyramid;
    p._pyramid = nullptr;

    return true;
}

void PopSift::uninit( )
{
    if(!_isInit)
    {
        std::cerr << "[warning] Attempt to release resources from an uninitialized instance" << std::endl;
        return;
    }
    _pipe.uninit();

    _isInit = false;
}

PopSift::AllocTest PopSift::testTextureFit( int width, int height )
{
    return AllocTest::Ok;
}

std::string PopSift::testTextureFitErrorString( AllocTest err, int width, int height )
{
    ostringstream ostr;
    ostr << "?    No error." << endl;
    return ostr.str();
}


SiftJob* PopSift::enqueue( int                  w,
                           int                  h,
                           const unsigned char* imageData )
{
    if( _image_mode != ByteImages )
    {
        stringstream ss;
        ss << "Image mode error" << endl
           << "E    Cannot load byte images into a PopSift pipeline configured for float images";
        POP_FATAL(ss.str());
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
        stringstream ss;
        ss << "Image mode error" << endl
           << "E    Cannot load float images into a PopSift pipeline configured for byte images";
        POP_FATAL(ss.str());
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
    applyConfiguration(true);

    Pipe& p = _pipe;

    SiftJob* job;
    while( ( job = p._queue_stage2.pull() ) != nullptr ) {
        applyConfiguration();

        popsift::ImageBase* img = job->getImg();

        private_init( img->getWidth(), img->getHeight() );

        p._pyramid->step1( _config, img );
        p._unused.push( img ); // uploaded input image no longer needed, release for reuse

        p._pyramid->step2( _config );

        popsift::FeaturesHost* features = p._pyramid->get_descriptors( _config );

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

    private_uninit();
}

void PopSift::matchPrepareLoop( )
{
    applyConfiguration(true);

    Pipe& p = _pipe;

    SiftJob* job;
    while( ( job = p._queue_stage2.pull() ) != nullptr ) {
        popsift::FeaturesDev* features;
        try
        {
            applyConfiguration();

            popsift::ImageBase* img = job->getImg();

            private_init(img->getWidth(), img->getHeight());

            p._pyramid->step1(_config, img);
            p._unused.push(img); // uploaded input image no longer needed, release for reuse

            p._pyramid->step2(_config);

            features = p._pyramid->clone_device_descriptors(_config);
        }
        catch(const std::exception& e)
        {
            job->setError(std::current_exception());
            job->setFeatures(nullptr);
            break;
        }

        job->setFeatures( features );
    }

    private_uninit();
}

SiftJob::SiftJob( int w, int h, const unsigned char* imageData )
    : _w(w)
    , _h(h)
    , _img(nullptr)
{
    _f = _p.get_future();

    _imageData = (unsigned char*)malloc( w*h );
    if( _imageData != nullptr )
    {
        memcpy( _imageData, imageData, w*h );
    }
    else
    {
        stringstream ss;
        ss << "Memory limitation" << endl
           << "E    Failed to allocate memory for SiftJob";
        POP_FATAL(ss.str());
    }
}

SiftJob::SiftJob( int w, int h, const float* imageData )
    : _w(w)
    , _h(h)
    , _img(nullptr)
{
    _f = _p.get_future();

    _imageData = (unsigned char*)malloc( w*h*sizeof(float) );
    if( _imageData != nullptr )
    {
        memcpy( _imageData, imageData, w*h*sizeof(float) );
    }
    else
    {
        stringstream ss;
        ss << "Memory limitation" << endl
           << "E    Failed to allocate memory for SiftJob";
        POP_FATAL(ss.str());
    }
}

SiftJob::~SiftJob( )
{
    free( _imageData );
}

void SiftJob::setImg( popsift::ImageBase* img )
{
    img->resetDimensions( _w, _h );
    img->load( _imageData );
    _img = img;
}

popsift::ImageBase* SiftJob::getImg()
{
    return _img;
}

void SiftJob::setFeatures( popsift::FeaturesBase* f )
{
    _p.set_value( f );
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
    popsift::FeaturesBase* features = _f.get();
    if(this->_err != nullptr) {
        std::rethrow_exception(this->_err);
    }
    return dynamic_cast<popsift::FeaturesDev*>(features);
}

void SiftJob::setError(std::exception_ptr ptr)
{
    this->_err = ptr;
}

void PopSift::Pipe::uninit()
{
    _queue_stage1.push( nullptr );
    if(_thread_stage2 != nullptr)
    {
        _thread_stage2->join();
        _thread_stage2.reset(nullptr);
    }
    if(_thread_stage1 != nullptr)
    {
        _thread_stage1->join();
        _thread_stage1.reset(nullptr);
    }

    while( !_unused.empty() )
    {
        popsift::ImageBase* img = _unused.pull();
        delete img;
    }
}
