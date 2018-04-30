/*
 * Copyright 2018, Simula Research Laboratory
 * Copyright 2018, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "../sift_task_extract.h"

/*********************************************************************************
 * TaskMatch
 *********************************************************************************/

class TaskMatch : public TaskExtract
{
public:
    TaskMatch( const popsift::Config& config ) : TaskExtract(config) { }

    virtual void loop();
private:
    /* Worker function: Extract SIFT features, clone results in device memory */
    void matchPrepareLoop( );
};

