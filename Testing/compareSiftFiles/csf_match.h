#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "csf_feat.h"

/** Parameters of the geometric matching.
 */
struct MatchParams
{
    /* A feature of the second file is a candidate for a feature of the first
     * file if their positions are at most max(radius_min, radius_sigma*sigma)
     * pixels apart, where sigma is the scale of the first feature.
     */
    float radius_sigma = 0.5f;
    float radius_min   = 1.0f;
    /* ... and if their scales differ by at most this factor. */
    float max_scale_ratio = 1.5f;
};

/** One pair of matched features, indices into the two feature lists.
 */
struct MatchPair
{
    int   a;
    int   b;
    float pos_dist;
    float ori_diff; // b - a in degrees, wrapped to [-180,180)
};

/** Pairs the features of l_a and l_b by geometry, not by descriptor.
 *
 * SIFT emits several features with different orientations at the same
 * position, so candidates are ranked by position distance first and by
 * orientation difference second. The assignment is one-to-one and greedy,
 * smallest distance first.
 */
std::vector<MatchPair> matchFeatures( const std::vector<feat_t>& l_a,
                                      const std::vector<feat_t>& l_b,
                                      const MatchParams&         param );

/** The descriptor nearest neighbour of one feature in another list.
 */
struct NearestDesc
{
    int   index  = -1;          // into the other list, -1 if it is empty
    float dist   = INFINITY;    // descriptor distance to it
    float second = INFINITY;    // descriptor distance to the second nearest
};

/** For every feature of l_query, the feature of l_other with the most similar
 * descriptor, by descDistance(). Brute force, parallel where available.
 */
std::vector<NearestDesc> nearestDescriptors( const std::vector<feat_t>& l_query,
                                             const std::vector<feat_t>& l_other );

/** Writes the nearest neighbours of l_query in l_other to path, one row per
 * feature of l_query: both features, their descriptor distance and that of
 * the second nearest, their position distance and orientation difference,
 * and geo_index, the geometric match of the query feature in l_other (-1 if
 * it has none). pairs_query_first tells whether MatchPair::a indexes l_query.
 * @return false if the file cannot be written
 */
bool writeNearestFile( const std::string&              path,
                       const std::vector<feat_t>&      l_query,
                       const std::vector<feat_t>&      l_other,
                       const std::vector<NearestDesc>& nearest,
                       const std::vector<MatchPair>&   pairs,
                       bool                            pairs_query_first );

/** Writes matched.csv, only_a.csv and only_b.csv into dir.
 * @return false if a file cannot be written
 */
bool writeMatchFiles( const std::string&            dir,
                      const std::vector<feat_t>&    l_a,
                      const std::vector<feat_t>&    l_b,
                      const std::vector<MatchPair>& pairs );

/** Signed orientation difference b - a in degrees, wrapped to [-180,180).
 */
float signedAngleDiffDegrees( float ori_a, float ori_b );
