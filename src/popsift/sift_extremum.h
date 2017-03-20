/*
 * Copyright 2016, Simula Research Laboratory
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

struct InitialExtremum
{
    float xpos;
    float ypos;
    int   lpos;  // extremum refined into this level
    float sigma; // scale;
};

/* This is an internal data structure.
 * For performance reasons, it would be appropriate to split
 * the first 4 values from the rest of this structure. Right
 * now, descriptor computation is a bigger concern.
 */
struct Extremum
{
    float xpos;
    float ypos;
    int   lpos;  // extremum refined into this level
    float sigma; // scale;

    int   octave;  // belonging to this octave
    int   num_ori; // number of this extremum's orientations
    int   idx_ori; // exclusive prefix sum of the layer's orientations
    float orientation[ORIENTATION_MAX_COUNT];
};

/* This is a data structure that is returned to a calling program.
 * This is the SIFT descriptor itself.
 */
struct Descriptor
{
    float features[128];
};

/* This is a data structure that is returned to a calling program.
 * The xpos/ypos information in feature is scale-adapted.
 * Note that these data structures are filled host-size in a loop
 * because addresses are not shared between CPU and GPU.
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

/* This is a data structure that is returned to a calling program.
 * _desc_buffer is a transparent flat memory holding descriptors
 * that are referenced by the extrema.
 *
 * Note that the current data structures do no allow to match
 * Descriptors in the transparent array with there extrema except
 * for brute force.
 */
class Features
{
    Feature*     _ext;
    Descriptor*  _ori;
    int          _num_ext;
    int          _num_ori;

public:
    Features( );
    Features( int num_ext, int num_ori );
    ~Features( );

    typedef Feature*       F_iterator;
    typedef const Feature* F_const_iterator;

    inline F_iterator       begin()       { return _ext; }
    inline F_const_iterator begin() const { return &_ext[_num_ext]; }
    inline F_iterator       end()         { return _ext; }
    inline F_const_iterator end() const   { return &_ext[_num_ext]; }

    void reset( int num_ext, int num_ori );
    void pin( );
    void unpin( );

    inline int     size() const                { return _num_ext; }
    inline int     getFeatureCount() const     { return _num_ext; }
    inline int     getDescriptorCount() const  { return _num_ori; }

    inline Feature*    getFeatures()    { return _ext; }
    inline Descriptor* getDescriptors() { return _ori; }

    void print( std::ostream& ostr, bool write_as_uchar ) const;

protected:
    friend class Pyramid;
};

std::ostream& operator<<( std::ostream& ostr, const Features& feature );

} // namespace popsift
