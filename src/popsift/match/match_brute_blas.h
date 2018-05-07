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

#include "../sift_features.h"
#include "../common/cublas_matrix.h"

namespace popsift {

struct Descriptor; // float features[128];

/* This is a data structure that is returned to a calling program.
 * The xpos/ypos information in feature is scale-adapted.
 */
class BruteForceBlasMatcher
{
    const FeaturesDev* _l;
    const FeaturesDev* _r;
    CuFortMatrix       _table;

public:
    BruteForceBlasMatcher( const FeaturesDev* l, const FeaturesDev* r );

    /* Find best matches in keypoint list other for all keypoints in this.
     * Allocate dev-side structures, match, print, delete structures.
     * This function is not suitable for further processing.
     * No transfer of descriptors to the host side.
     */
    void matchAndPrint( );

    /* Like matchAndPrint, but removing the not accepted matches before
     * printing.
     */
    void matchAndPrintAccepted( );

    /* Allocate a dev-side table that will contain matching results
     */
    int3* match_AllocMatchTable( ) const;

    /* Free dev-side table that was created by match_AllocMatchTable.
     */
    void match_freeMatchTable( int3* table ) const;

    /* Fill dev-side table with tuples of (best match, 2nd best match, accept)
     * where accept is true when the ratio of best and 2nd best is < 0.8
     */
    void match_computeMatchTable( int3* match_matrix);

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
                thrust::device_vector<int2>& matches );

    /** Return a Thrust vector containing matching index pointing into the Feature
     *  tables of "this" and "that".
     *  These matches are accepted matches using the ratio test.
     *  For each match, x is an index in this Feature list, y is an index in the best
     *  match in the other feature list.
     *  The vector is sorted by left indices and unique.
     */
    void match_getAcceptedFeatureMatches(
                thrust::device_vector<int2>& matches );

    /* Release the map containing the indices to accepted matches.
     */
    void match_freeAcceptedIndex( int* index ) const;

    /* A debug function to find out whether feature lists of two identical
     * inputs are actually identical, as they should be.
     */
    void checkIdentity( ) const;
};

} // namespace popsift

