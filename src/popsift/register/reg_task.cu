/*
 * Copyright 2018, Simula Research Laboratory
 * Copyright 2018, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
// #include <fstream>
// #include <iostream>
// #include <iomanip>

#include "reg_task.h"
#include "reg_job.h"
#include "../popsift.h"
#include "../common/debug_macros.h"
#include "../sift_features.h"

using namespace std;

/*********************************************************************************
 * TaskRegister
 *********************************************************************************/

void TaskRegister::loop( )
{
    registrationPrepareLoop();
}

void TaskRegister::registrationPrepareLoop( )
{
    SiftJob* siftjob;
    while( ( siftjob = _op->getNextJob() ) != 0 )
    {
        popsift::FeaturesDev* features;

        RegistrationJob* job = dynamic_cast<RegistrationJob*>( siftjob );
        POP_CHECK_NON_NULL( job, "registration loop jobs must have type RegistrationJob" );

        popsift::ImageBase* img = job->getImg();
        _op->uploadImageFromJob( img );
        _op->returnImageToPool( img );
        _op->findKeypoints( );
        features = _op->cloneFeaturesOnDevice( );

        job->setPlane( _op->cloneLayerOnDevice( 0, 0 ) );

        job->setFeatures( features );
    }
}


SiftJob* TaskRegister::newJob( int w, int h, const unsigned char* imageData )
{
    return new RegistrationJob( w, h, imageData );
}

SiftJob* TaskRegister::newJob( int w, int h, const float* imageData )
{
    return new RegistrationJob( w, h, imageData );
}

