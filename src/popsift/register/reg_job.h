/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "../sift_job.h"
#include "../common/plane_2d.h"

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

