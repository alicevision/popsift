#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "csf_feat.h"

/** Parameters of the pairing by position, and the tolerances of the two
 * agreement levels above it.
 */
struct PairParams
{
    /* A keypoint of the second file is a candidate for a keypoint of the
     * first file if their positions are at most max(radius_min,
     * radius_sigma*sigma) pixels apart, where sigma is the scale of the first
     * keypoint ...
     */
    float radius_sigma = 0.5f;
    float radius_min   = 1.0f;
    /* ... and if their scales differ by at most this factor. */
    float max_scale_ratio = 1.5f;
    /* A co-located pair has the same orientation if the orientations differ
     * by at most this many degrees ...
     */
    float ori_tolerance_deg = 2.0f;
    /* ... and also a similar descriptor if the descriptor distance is at
     * most this. 12 is a little more than sqrt(128): every one of the 128
     * byte values may differ by 1.
     */
    float desc_tolerance = 12.0f;
};

/** How far two co-located keypoints agree. Every level includes the ones
 * before it.
 *
 * Position:    same position and scale, but the orientations differ. The
 *              orientation computation differs; nothing can be concluded
 *              about the descriptor extraction.
 * Orientation: also the same orientation, but the descriptors differ. A
 *              strong hint at a difference in the descriptor extraction.
 * Descriptor:  also a similar descriptor.
 */
enum class Agreement
{
    Position,
    Orientation,
    Descriptor
};

/** The name of an agreement level in the CSV files.
 */
const char* agreementName( Agreement level );

/* SIFT gives one position and scale several orientations, each its own
 * keypoint. When one pair at a position agrees further than another, the
 * other one is a minor problem: an orientation or descriptor that only one
 * run has, or that the greedy pairing did not pair with its counterpart.
 * KeypointPair::best_at_position is the highest level of all pairs whose
 * keypoint of a has the same position and scale.
 */

/** One pair of co-located keypoints, indices into the two keypoint lists.
 */
struct KeypointPair
{
    int       a;
    int       b;
    float     pos_dist;
    float     ori_diff;  // b - a in degrees, wrapped to [-180,180)
    float     desc_dist;        // descDistance(), set by classifyAgreement()
    Agreement level;            // set by classifyAgreement()
    Agreement best_at_position; // set by classifyAgreement()
};

/** Pairs the keypoints of l_a and l_b by position and scale, not by
 * descriptor. This tells whether two implementations or parameter sets find
 * the same extrema in their pyramids.
 *
 * SIFT emits several keypoints with different orientations at the same
 * position, so candidates are ranked by position distance first and by
 * orientation difference second. The assignment is one-to-one and greedy,
 * smallest distance first. The result is sorted by a, with level Position.
 */
std::vector<KeypointPair> matchKeypointsByPosition( const std::vector<feat_t>& l_a,
                                                    const std::vector<feat_t>& l_b,
                                                    const PairParams&          param );

/** Sets desc_dist and level of every pair: Orientation if the orientations
 * differ by at most param.ori_tolerance_deg, Descriptor if in addition the
 * descriptor distance is at most param.desc_tolerance. Then best_at_position,
 * the highest level among the pairs whose keypoint of a has exactly the same
 * position and scale.
 */
void classifyAgreement( std::vector<KeypointPair>& pairs,
                        const std::vector<feat_t>& l_a,
                        const std::vector<feat_t>& l_b,
                        const PairParams&          param );

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
void nearestDescriptors( std::vector<NearestDesc>& nearest_out,
                         const std::vector<feat_t>& l_query,
                         const std::vector<feat_t>& l_other );

/** Writes the nearest neighbours of l_query in l_other to path, one row per
 * feature of l_query: both features, their descriptor distance and that of
 * the second nearest, their position distance and orientation difference,
 * and colocated_index, the keypoint of l_other co-located with the query
 * keypoint (-1 if it has none). pairs_query_first tells whether
 * KeypointPair::a indexes l_query.
 * @return false if the file cannot be written
 */
bool writeNearestFile( const std::string&               path,
                       const std::vector<feat_t>&       l_query,
                       const std::vector<feat_t>&       l_other,
                       const std::vector<NearestDesc>&  nearest,
                       const std::vector<KeypointPair>& pairs,
                       bool                             pairs_query_first );

/** Writes colocated.csv, one row per co-located pair with its agreement
 * level, and only_a.csv and only_b.csv, the keypoints without a co-located
 * keypoint in the other file, into dir. Sorted by the index in l_a.
 * @return false if a file cannot be written
 */
bool writePairFiles( const std::string&               dir,
                     const std::vector<feat_t>&       l_a,
                     const std::vector<feat_t>&       l_b,
                     const std::vector<KeypointPair>& pairs );

/** Writes desc_bins.csv into dir: for every one of the 128 descriptor
 * elements, the number of pairs, the sum of the differences b - a and the
 * sum of their absolute values. Once over all co-located pairs ("colocated")
 * and once over those with the same orientation ("same_orientation").
 * @return false if the file cannot be written
 */
bool writeDescBinsFile( const std::string&               dir,
                        const std::vector<feat_t>&       l_a,
                        const std::vector<feat_t>&       l_b,
                        const std::vector<KeypointPair>& pairs );

/** Signed orientation difference b - a in degrees, wrapped to [-180,180).
 */
float signedAngleDiffDegrees( float ori_a, float ori_b );
