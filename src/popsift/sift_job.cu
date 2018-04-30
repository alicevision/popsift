/*
 * Copyright 2018, Simula Research Laboratory
 * Copyright 2018, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "sift_constants.h"
#include "sift_job.h"
#include "s_image.h"
#include "common/assist.h"
#include "sift_features.h"

using namespace std;

/*********************************************************************************
 * SiftJob
 *********************************************************************************/

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

SiftJob::SiftJob( int w, int h, const float* imageData )
    : _w(w)
    , _h(h)
    , _img(0)
{
    _f = _p.get_future();

    _imageData = (unsigned char*)malloc( w*h*sizeof(float) );
    if( _imageData != 0 ) {
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

