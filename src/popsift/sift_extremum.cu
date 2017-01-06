/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "sift_extremum.h"
#include <iomanip>

namespace popsift {

Features::Features( )
    : _desc_buffer( 0 )
{ }

Features::~Features( )
{
    _features.clear();
    delete [] _desc_buffer;
}

void Features::print( std::ostream& ostr, bool write_as_uchar ) const
{
    std::vector<Feature>::const_iterator it  = _features.begin();
    std::vector<Feature>::const_iterator end = _features.end();

    for( ; it!=end; it++ ) {
        const Feature& f = *it;
        f.print( ostr, write_as_uchar );
    }
}

std::ostream& operator<<( std::ostream& ostr, const Features& feature )
{
    feature.print( ostr, false );
    return ostr;
}

void Feature::print( std::ostream& ostr, bool write_as_uchar ) const
{
    float sigval =  1.0f / ( sigma * sigma );

    for( int ori=0; ori<num_descs; ori++ ) {
        ostr << xpos << " " << ypos << " "
             << sigval << " 0 " << sigval << " ";
        if( write_as_uchar ) {
            for( int i=0; i<128; i++ ) {
                ostr << roundf(desc[ori]->features[i]) << " ";
            }
        } else {
            ostr << std::setprecision(3);
            for( int i=0; i<128; i++ ) {
                ostr << desc[ori]->features[i] << " ";
            }
            ostr << std::setprecision(6);
        }
        ostr << std::endl;
    }
}

std::ostream& operator<<( std::ostream& ostr, const Feature& feature )
{
    feature.print( ostr, false );
    return ostr;
}

} // namespace popsift
