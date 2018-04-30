/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <future>

#include "sift_conf.h"
#include "common/plane_2d.h"


#ifdef USE_NVTX
#include <nvToolsExtCuda.h>
#else
#define nvtxRangeStartA(a)
#define nvtxRangeEnd(a)
#endif

/* user parameters */
namespace popsift
{
    class ImageBase;
    class FeaturesBase;
    class FeaturesHost;
    class FeaturesDev;

}; // namespace popsift

class SiftJob
{
    std::promise<popsift::FeaturesBase*> _p;
    std::future <popsift::FeaturesBase*> _f;
    int                 _w;
    int                 _h;
    unsigned char*      _imageData;
    popsift::ImageBase* _img;
#ifdef USE_NVTX
    nvtxRangeId_t       _nvtx_id;
#endif

public:
    /** Constructor for byte images, value range 0..255 */
    SiftJob( int w, int h, const unsigned char* imageData );

    /** Constructor for float images, value range [0..1[ */
    SiftJob( int w, int h, const float* imageData );

    virtual ~SiftJob( );

    popsift::FeaturesHost* get();    // should be deprecated, same as getHost()
    popsift::FeaturesBase* getBase();
    popsift::FeaturesHost* getHost();
    popsift::FeaturesDev*  getDev();

    void setImg( popsift::ImageBase* img );
    popsift::ImageBase* getImg();

    /** fulfill the promise */
    void setFeatures( popsift::FeaturesBase* f );
};

class RegistrationJob : public SiftJob
{
    /* Some data from the Pyramid must be retained for registration.
     * We don't know yet what that is, this is experimental.
     */
    popsift::Plane2D<float>* _blurred_input;

public:
    /** Constructor for byte images, value range 0..255 */
    RegistrationJob( int w, int h, const unsigned char* imageData );

    /** Constructor for float images, value range [0..1[ */
    RegistrationJob( int w, int h, const float* imageData );

    virtual ~RegistrationJob( );

    void setPlane( popsift::Plane2D<float>* plane );

    inline popsift::Plane2D<float>* getPlane() const { return _blurred_input; }
};

