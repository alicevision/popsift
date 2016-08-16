/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "sift_extremum.h"

namespace popsift {

Features::Features( )
    : _desc_buffer( 0 )
{ }

Features::~Features( )
{
    _features.clear();
    delete [] _desc_buffer;
}

std::ostream& operator<<( std::ostream& ostr, const Features& feature )
{
    std::vector<Feature>::const_iterator it  = feature.begin();
    std::vector<Feature>::const_iterator end = feature.end();

    for( ; it!=end; it++ ) {
        const Feature& f = *it;
        ostr << f;
    }

    return ostr;
}

std::ostream& operator<<( std::ostream& ostr, const Feature& feature )
{
    float sigval =  1.0f / ( feature.sigma * feature.sigma );

    for( int ori=0; ori<feature.num_descs; ori++ ) {
        ostr << feature.xpos << " " << feature.ypos << " "
             << sigval << " 0 " << sigval << " ";
        for( int i=0; i<128; i++ ) {
            ostr << feature.desc[ori]->features[i] << " ";
        }
        ostr << std::endl;
    }
    return ostr;
}

} // namespace popsift
