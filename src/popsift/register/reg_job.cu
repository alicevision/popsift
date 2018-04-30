/*
 * Copyright 2018, Simula Research Laboratory
 * Copyright 2018, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "reg_job.h"
// #include "sift_constants.h"
// #include "sift_job.h"
// #include "s_image.h"
// #include "common/assist.h"
// #include "sift_features.h"

using namespace std;

/*********************************************************************************
 * RegistrationJob
 *********************************************************************************/

RegistrationJob::RegistrationJob( int w, int h, const unsigned char* imageData )
    : SiftJob( w, h, imageData )
    , _blurred_input( 0 )
{ }

RegistrationJob::RegistrationJob( int w, int h, const float* imageData )
    : SiftJob( w, h, imageData )
    , _blurred_input( 0 )
{ }

RegistrationJob::~RegistrationJob( )
{
    _blurred_input->freeDev();
    delete _blurred_input;
}

void RegistrationJob::setPlane( popsift::Plane2D<float>* plane )
{
    if( _blurred_input ) delete _blurred_input;
    _blurred_input = plane;
}

