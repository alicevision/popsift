/*
 * Copyright 2018, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "../sift_features.h"
#include "../common/plane_2d.h"

namespace popsift {

class Registration
{
public:
    Registration( );

    void setKeypointsA( const FeaturesDev* p );
    void setKeypointsB( const FeaturesDev* p );
    void setPlaneA( const Plane2D<float>* p );
    void setPlaneB( const Plane2D<float>* p );

    void compute( );

private:
    void private_makeTexture( cudaTextureObject_t& tex, const Plane2D<float>* plane );
    void private_destroyTexture( cudaTextureObject_t& tex );

    void private_makeMatches( int3*& match_a_to_b, int& good_match_len, int*& good_match_a );
    void private_destroyMatches( int3* match_a_to_b, int* good_match_a );

private:
    const FeaturesDev*    _keypt_a;
    const FeaturesDev*    _keypt_b;
    const Plane2D<float>* _plane_a;
    const Plane2D<float>* _plane_b;
};

} // namespace popsift
