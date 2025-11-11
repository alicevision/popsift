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
}

PopSift::PopSift( ImageMode imode )
    : _image_mode( imode )
{
}

PopSift::~PopSift()
{
   // Clean up any remaining jobs in queue_stage1
   while(!_pipe._queue_stage1.empty()) {
       SiftJob* job = _pipe._queue_stage1.front();
       _pipe._queue_stage1.pop();
       delete job;
   }
   
   // Clean up any remaining jobs in queue_stage2
   while(!_pipe._queue_stage2.empty()) {
       SiftJob* job = _pipe._queue_stage2.front();
       _pipe._queue_stage2.pop();
       delete job;
   }
   
   // Delete the pyramid (this will call Pyramid destructor and free all memory)
   if(_pipe._pyramid) {
        delete _pipe._pyramid;
        _pipe._pyramid = nullptr;
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

    return true;
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
    std::cerr << __FILE__ << ":" << __LINE__ << " uploading byte image (" << w << "x" << h << " pixels)" << endl;

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
    std::cerr << __FILE__ << ":" << __LINE__ << " uploading float image (" << w << "x" << h << " pixels)" << endl;

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

void PopSift::processExtract( )
{
    uploadImages();
    extractDownloadLoop();
}

void PopSift::processMatch( )
{
    uploadImages();
    matchPrepareLoop();
}

void PopSift::uploadImages( )
{
    while( !_pipe._queue_stage1.empty() )
    {
        SiftJob* job = _pipe._queue_stage1.front();
        _pipe._queue_stage1.pop();
        _pipe._queue_stage2.push( job );
    }
}

void PopSift::extractDownloadLoop( )
{
    applyConfiguration(true);

    Pipe& p = _pipe;

    while( !_pipe._queue_stage2.empty() )
    {
        SiftJob* job = _pipe._queue_stage2.front();
        _pipe._queue_stage2.pop();

        applyConfiguration();

        std::shared_ptr<popsift::ImageBase> img = job->getImg();

        if( img == NULL )
        {
            std::cerr << __FUNCTION__ << ":" << __LINE__ << " read a job that contains no image" << std::endl;
            exit( -1 );
        }
        if( img->isNull() )
        {
            std::cerr << __FUNCTION__ << ":" << __LINE__ << " read a job that contains a NULL image" << std::endl;
            exit( -1 );
        }

        private_init( img->getWidth(), img->getHeight() );

        p._pyramid->step1( _config, img );

        if( !_config.silent() )
        {
            POP_INFO2( _config.silent(), "Downloading all pyramid images" );
            p._pyramid->download_and_save_array( _config, "pyramid" );
        }

        p._pyramid->step2( _config );

        std::unique_ptr<popsift::FeaturesHost> features = p._pyramid->get_descriptors( _config );

        bool log_to_file = ( _config.getLogMode() == popsift::Config::All );
        if( log_to_file )
        {
            // int octaves = p._pyramid->getNumOctaves();
            // for( int o=0; o<octaves; o++ ) { p._pyramid->download_descriptors( _config, o ); }
            // int levels  = p._pyramid->getNumLevels();

            // p._pyramid->download_and_save_array( "pyramid" );
            p._pyramid->save_descriptors( _config, features, "pyramid" );
        }

        job->setFeatures( features );
    }

    POP_INFO2( _config.silent(), "DEBUG: ExtractDownloadLoop finished" );
}

void PopSift::matchPrepareLoop( )
{
    applyConfiguration(true);

    Pipe& p = _pipe;

    while( !_pipe._queue_stage2.empty() )
    {
        SiftJob* job = _pipe._queue_stage2.front();
        _pipe._queue_stage2.pop();

        std::unique_ptr<popsift::FeaturesHost> features;
        try
        {
            applyConfiguration();

            std::shared_ptr<popsift::ImageBase> img = job->getImg();

            private_init(img->getWidth(), img->getHeight());

            p._pyramid->step1(_config, img);

            p._pyramid->step2(_config);

            features = p._pyramid->clone_device_descriptors(_config);
        }
        catch(const std::exception& e)
        {
            features.release();
            job->setFeatures( features );
            job->setError(std::current_exception());
            break;
        }

        job->setFeatures( features );
    }
}

SiftJob::SiftJob( int w, int h, const unsigned char* imageData )
    : _w(w)
    , _h(h)
    , _img( new popsift::Image(w,h) )
{
    std::cerr << __FILE__ << ":" << __LINE__ << ": enter " << __PRETTY_FUNCTION__ << std::endl;
    _f.swap( _promise );

    std::cerr << __FILE__ << ":" << __LINE__ << ": load image data" << std::endl;
    _img->load( imageData );
    std::cerr << __FILE__ << ":" << __LINE__ << ": is image NULL? " << (_img->isNull() ? "yes" : "no") << std::endl;
}

SiftJob::SiftJob( int w, int h, const float* imageData )
    : _w(w)
    , _h(h)
    , _img( new popsift::ImageFloat(w,h) )
{
    std::cerr << __FILE__ << ":" << __LINE__ << ": enter " << __PRETTY_FUNCTION__ << std::endl;
    _f.swap( _promise );

    std::cerr << __FILE__ << ":" << __LINE__ << ": load image data" << std::endl;
    _img->load( imageData );
    std::cerr << __FILE__ << ":" << __LINE__ << ": is image NULL? " << (_img->isNull() ? "yes" : "no") << std::endl;
}

SiftJob::~SiftJob( )
{
}

std::shared_ptr<popsift::ImageBase> SiftJob::getImg()
{
    std::cerr << __FILE__ << ":" << __LINE__ << ": enter " << __PRETTY_FUNCTION__ << std::endl;
    std::cerr << __FILE__ << ":" << __LINE__ << ": is image NULL? " << (_img->isNull() ? "yes" : "no") << std::endl;
    return _img;
}

void SiftJob::setFeatures( std::unique_ptr<popsift::FeaturesHost>& f )
{
    _promise.swap(f);
}

std::unique_ptr<popsift::FeaturesHost>& SiftJob::get()
{
    if( _err != nullptr ) std::rethrow_exception( _err );
    
    return _promise;
}

void SiftJob::setError(std::exception_ptr ptr)
{
    this->_err = ptr;
}

