/*
 * Copyright 2018, Simula Research Laboratory
 * Copyright 2018, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "sift_conf.h"

/* user parameters */
namespace popsift
{
    class FeaturesBase;
    class FeaturesHost;
    class FeaturesDev;

}; // namespace popsift

class PopSift;
class SiftJob;

/*********************************************************************************
 * Task
 *********************************************************************************/

class Task
{
public:
    Task( popsift::Config& config ) : _config(config) { }

    virtual void loop() = 0;

    inline void setOperator( PopSift* op ) { _op = op; }

    virtual SiftJob* newJob( int w, int h, const unsigned char* imageData ) = 0;
    virtual SiftJob* newJob( int w, int h, const float*         imageData ) = 0;
protected:
    PopSift*         _op;
    popsift::Config& _config;
};

