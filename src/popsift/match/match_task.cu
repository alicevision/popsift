/*
 * Copyright 2018, Simula Research Laboratory
 * Copyright 2018, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "match_task.h"
#include "../sift_job.h"
#include "../popsift.h"
#include "../sift_features.h"

using namespace std;

/*********************************************************************************
 * TaskMatch
 *********************************************************************************/

void TaskMatch::loop( )
{
    matchPrepareLoop();
}

void TaskMatch::matchPrepareLoop( )
{
    SiftJob* job;
    while( ( job = _op->getNextJob() ) != 0 )
    {
        popsift::FeaturesDev* features;

        popsift::ImageBase* img = job->getImg();
        _op->uploadImageFromJob( img );
        _op->returnImageToPool( img );
        _op->findKeypoints( );
        features = _op->cloneFeaturesOnDevice( );

        job->setFeatures( features );
    }
}

