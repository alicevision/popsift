/*
 * Copyright 2018, Simula Research Laboratory
 * Copyright 2018, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "sift_task.h"

/*********************************************************************************
 * TaskExtract
 *********************************************************************************/

class TaskExtract : public Task
{
public:
    TaskExtract( popsift::Config& config ) : Task(config) { }

    virtual void loop();

    virtual SiftJob* newJob( int w, int h, const unsigned char* imageData );
    virtual SiftJob* newJob( int w, int h, const float*         imageData );
private:
    /* Worker function: Extract SIFT features and download to host */
    void extractDownloadLoop( );

    /* If Config asks for logging All, we write descriptors
     * to files.
     */
    void logDescriptors( popsift::FeaturesHost* features );

    void writeDescriptors( std::ostream& ostr, popsift::FeaturesHost* features, bool really, bool with_orientation );
};

/*********************************************************************************
 * TaskMatch
 *********************************************************************************/

class TaskMatch : public TaskExtract
{
public:
    TaskMatch( popsift::Config& config ) : TaskExtract(config) { }

    virtual void loop();
private:
    /* Worker function: Extract SIFT features, clone results in device memory */
    void matchPrepareLoop( );
};

