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

#include <thrust/device_vector.h>

#include "sift_constants.h"

namespace popsift {

struct Descriptor; // float features[128];

/* This is a data structure that is returned to a calling program.
 * The xpos/ypos information in feature is scale-adapted.
 */
struct Feature
{
    int         octave;  // octave where keypoint was found
                         // technically not needed, could be re-computed from sigma
    float       xpos;    // xpos in input frame
    float       ypos;    // ypos in input frame
    float       scale;   // scale computed from sigma;
    int         num_ori; // number of this extremum's orientations
                         // remaining entries in desc are 0
    float       orientation[ORIENTATION_MAX_COUNT];
    Descriptor* desc[ORIENTATION_MAX_COUNT];

    void print( std::ostream& ostr, bool write_as_uchar ) const;
};

std::ostream& operator<<( std::ostream& ostr, const Feature& feature );

class FeaturesBase
{
    int          _num_ext;
    int          _num_ori;

public:
    FeaturesBase( );
    virtual~ FeaturesBase( );

    inline int     size() const                { return _num_ext; }
    inline int     getFeatureCount() const     { return _num_ext; }
    inline int     getDescriptorCount() const  { return _num_ori; }

    inline void    setFeatureCount( int num_ext )    { _num_ext = num_ext; }
    inline void    setDescriptorCount( int num_ori ) { _num_ori = num_ori; }
};

/** This is a data structure that is returned to a calling program.
 *  _ori is a transparent flat memory holding descriptors
 *  that are referenced by the extrema.
 *
 *  Note that the current data structures do not allow to match
 *  Descriptors in the transparent array with their extrema except
 *  for brute force.
 *
 *  Note: FeaturesHost is typedef'd to its older name Features
 */
class FeaturesHost : public FeaturesBase
{
    Feature*     _ext;
    Descriptor*  _ori;

public:
    FeaturesHost( );
    FeaturesHost( int num_ext, int num_ori );
    virtual ~FeaturesHost( );

    typedef Feature*       F_iterator;
    typedef const Feature* F_const_iterator;

    inline F_iterator       begin()       { return _ext; }
    inline F_const_iterator begin() const { return _ext; }
    inline F_iterator       end()         { return &_ext[size()]; }
    inline F_const_iterator end() const   { return &_ext[size()]; }

    /** Allocate or re-allocate the host-side memory for keypoints and
     *  descriptors. The memory is memaligned regular memory that must
     *  be pinned for efficient transfer from CUDA.
     */
    void reset( int num_ext, int num_ori );

    /** Pin keypoint and descriptor memory.
     *  Failure to pin results in a warning since transfer can still happen,
     *  but slower.
     */
    void pin( );

    /** Unpin keypoint and descriptor memory */
    void unpin( );

    inline Feature*       getFeatures()               { return _ext; }
    inline const Feature& getFeature( int idx ) const { return _ext[idx]; }
    inline Descriptor*    getDescriptors()            { return _ori; }

    void print( std::ostream& ostr, bool write_as_uchar ) const;

protected:
    friend class Pyramid;
};

/** This typedef exists only for backwards compatibility */
typedef FeaturesHost Features;

std::ostream& operator<<( std::ostream& ostr, const FeaturesHost& feature );

class FeaturesDev : public FeaturesBase
{
    Feature*     _ext;
    Descriptor*  _ori;
    int*         _rev; // the reverse map from descriptors to extrema

public:
    FeaturesDev( );
    FeaturesDev( int num_ext, int num_ori );
    virtual ~FeaturesDev( );

    void reset( int num_ext, int num_ori );

    /* Find best matches in keypoint list other for all keypoints in this.
     * Allocate dev-side structures, match, print, delete structures.
     * This function is not suitable for further processing.
     * No transfer of descriptors to the host side.
     */
    void matchAndPrint( const FeaturesDev* other ) const;

    /* Like matchAndPrint, but removing the not accepted matches before
     * printing.
     */
    void matchAndPrintAccepted( const FeaturesDev* other ) const;

    /* Allocate a dev-side table that will contain matching results
     */
    int3* match_AllocMatchTable( ) const;

    /* Free dev-side table that was created by match_AllocMatchTable.
     */
    void match_freeMatchTable( int3* table ) const;

    /* Fill dev-side table with tuples of (best match, 2nd best match, accept)
     * where accept is true when the ratio of best and 2nd best is < 0.8
     */
    void match_computeMatchTable( int3* match_matrix, const FeaturesDev* other ) const;

    /* Find the accepted matches in the match_matrix.
     * Return a vector of indices that have been accepted, and return the
     * length of this array in l_accepted_len.
     */
    int* match_getAcceptedIndex( const int3* match_matrix, int& l_accepted_len ) const;

    void match_getAcceptedDescriptorMatchesFromMatrix(
                int3*                        match_matrix,
                thrust::device_vector<int2>& accepted_matches ) const;

    void match_getAcceptedFeatureMatchesFromMatrix(
                int3*                        match_matrix,
                thrust::device_vector<int2>& accepted_matches,
                int*                         l_fem,
                int*                         r_fem ) const;

    /** Return a Thrust vector containing matching index pointing into the Descriptor
     *  tables of "this" and "that".
     *  These matches are accepted matches using the ratio test.
     *  For each match, x is an index in this Feature list, y is an index in the best
     *  match in the other feature list.
     */
    void match_getAcceptedDescriptorMatches(
                thrust::device_vector<int2>& matches,
                const FeaturesDev*           that ) const;

    /** Return a Thrust vector containing matching index pointing into the Feature
     *  tables of "this" and "that".
     *  These matches are accepted matches using the ratio test.
     *  For each match, x is an index in this Feature list, y is an index in the best
     *  match in the other feature list.
     *  The vector is sorted by left indices and unique.
     */
    void match_getAcceptedFeatureMatches(
                thrust::device_vector<int2>& matches,
                const FeaturesDev*           that ) const;

    /* Release the map containing the indices to accepted matches.
     */
    void match_freeAcceptedIndex( int* index ) const;

    /* A debug function to find out whether feature lists of two identical
     * inputs are actually identical, as they should be.
     */
    void checkIdentity( const FeaturesDev* other ) const;

    inline Feature*    getFeatures()    const { return _ext; }
    inline Descriptor* getDescriptors() const { return _ori; }
    inline int*        getReverseMap()  const { return _rev; }
};

} // namespace popsift

