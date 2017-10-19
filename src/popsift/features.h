/*
 * Copyright 2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <iostream>
#include <vector>

#include "sift_constants.h"

namespace popsift {

struct Descriptor; // float features[128];

/* This is a data structure that is returned to a calling program.
 * The xpos/ypos information in feature is scale-adapted.
 */
struct Feature
{
    int         debug_octave;
    float       xpos;
    float       ypos;
    float       sigma;   // scale;
    int         num_ori; // number of this extremum's orientations
                         // remaining entries in desc are 0
    float       orientation[ORIENTATION_MAX_COUNT];
    Descriptor* desc[ORIENTATION_MAX_COUNT];

    void print( std::ostream& ostr, bool write_as_uchar ) const;
};

std::ostream& operator<<( std::ostream& ostr, const Feature& feature );

class Features
{
    int          _num_ext;
    int          _num_ori;

public:
    Features( );
    virtual~ Features( );

    inline int     size() const                { return _num_ext; }
    inline int     getFeatureCount() const     { return _num_ext; }
    inline int     getDescriptorCount() const  { return _num_ori; }

    inline void    setFeatureCount( int num_ext )    { _num_ext = num_ext; }
    inline void    setDescriptorCount( int num_ori ) { _num_ori = num_ori; }
};

/* This is a data structure that is returned to a calling program.
 * _ori is a transparent flat memory holding descriptors
 * that are referenced by the extrema.
 *
 * Note that the current data structures do not allow to match
 * Descriptors in the transparent array with their extrema except
 * for brute force.
 */
class HostFeatures : public Features
{
    Feature*     _ext;
    Descriptor*  _ori;

public:
    HostFeatures( );
    HostFeatures( int num_ext, int num_ori );
    virtual ~HostFeatures( );

    typedef Feature*       F_iterator;
    typedef const Feature* F_const_iterator;

    inline F_iterator       begin()       { return _ext; }
    inline F_const_iterator begin() const { return _ext; }
    inline F_iterator       end()         { return &_ext[size()]; }
    inline F_const_iterator end() const   { return &_ext[size()]; }

    void reset( int num_ext, int num_ori );
    void pin( );
    void unpin( );

    inline Feature*    getFeatures()    { return _ext; }
    inline Descriptor* getDescriptors() { return _ori; }

    void print( std::ostream& ostr, bool write_as_uchar ) const;

protected:
    friend class Pyramid;
};

std::ostream& operator<<( std::ostream& ostr, const HostFeatures& feature );

class DeviceFeatures : public Features
{
    Feature*     _ext;
    Descriptor*  _ori;
    int*         _rev; // the reverse map from descriptors to extrema

public:
    DeviceFeatures( );
    DeviceFeatures( int num_ext, int num_ori );
    virtual ~DeviceFeatures( );

    void reset( int num_ext, int num_ori );

    void match( DeviceFeatures* other );

    inline Feature*    getFeatures()    { return _ext; }
    inline Descriptor* getDescriptors() { return _ori; }
    inline int*        getReverseMap()  { return _rev; }
};

} // namespace popsift
