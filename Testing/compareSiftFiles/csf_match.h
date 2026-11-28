#pragma once

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "csf_feat.h"

/** Parameters of the pairing by position, and the tolerances of the two
 * agreement levels above it. No values are validated; the ranges below are
 * the meaningful ones.
 */
struct PairParams
{
    /** A keypoint b of the second file is a candidate for a keypoint a of the
     * first file if their positions are at most max(radius_min,
     * radius_sigma*a.sigma) pixels apart (inclusive) ... Both >= 0.
     */
    float radius_sigma = 0.5f;
    float radius_min   = 1.0f; ///< see radius_sigma

    /** ... and if 1/max_scale_ratio <= b.sigma/a.sigma <= max_scale_ratio.
     * >= 1; not checked if either sigma is 0.
     */
    float max_scale_ratio = 1.5f;

    /** A co-located pair has the same orientation if |ori_diff| is at most
     * this many degrees (inclusive) ...
     */
    float ori_tolerance_deg = 2.0f;

    /** ... and also a similar descriptor if the descriptor distance is at
     * most this (inclusive). 12 is a little more than sqrt(128): every one of
     * the 128 byte values may differ by 1.
     */
    float desc_tolerance = 12.0f;
};

/** How far two co-located keypoints agree. Every level includes the ones
 * before it; the order of the enumerators is part of the contract.
 */
enum class Agreement
{
    /** Same position and scale, but the orientations differ. The orientation
     * computation differs; nothing can be concluded about the descriptor.
     */
    Position,

    /** Also the same orientation, but the descriptors differ. A strong hint
     * at a difference in the descriptor extraction.
     */
    Orientation,

    /** All values including descriptor are similar.
     */
    Descriptor
};

/* SIFT gives one position and scale several orientations, each its own
 * keypoint. When one pair at a position agrees further than another, the
 * other one is a minor problem: an orientation or descriptor that only one
 * run has, or that the greedy pairing did not pair with its counterpart.
 * KeypointPair::best_at_position is the highest level of all pairs whose
 * keypoint of a has the same position and scale.
 */

/** One pair of co-located keypoints.
 */
struct KeypointPair
{
    int       a;         ///< index into the first list
    int       b;         ///< index into the second list
    float     pos_dist;  ///< position distance in pixels
    float     ori_diff;  ///< b - a in degrees, wrapped to [-180,180)
    float     desc_dist; ///< descDistance(); NaN until classifyAgreement()
    Agreement level;     ///< Position until classifyAgreement()
    Agreement best_at_position; ///< set by classifyAgreement()
};

/** Pairs the keypoints of l_A and l_B by position and scale (see
 * PairParams), not by descriptor. This tells whether two implementations or
 * parameter sets find the same extrema in their pyramids.
 *
 * SIFT emits several keypoints with different orientations at the same
 * position, so candidates are ranked by position distance, then by
 * |orientation difference|, then by a, then by b. The assignment is
 * one-to-one and greedy in that order. Not symmetric in l_A and l_B, because
 * the radius uses the sigma of the l_A keypoint.
 * @param[in] l_A   first keypoint list
 * @param[in] l_B   second keypoint list
 * @param[in] param pairing parameters
 * @return the pairs sorted by a, each a and each b at most once, with
 *         desc_dist NaN and level Position
 */
std::vector<KeypointPair> matchKeypointsByPosition( const feat_vector_t&       l_A,
                                                    const feat_vector_t&       l_B,
                                                    const PairParams&          param );

/** Sets desc_dist = descDistance() and level of every pair: Orientation if
 * |ori_diff| <= param.ori_tolerance_deg, Descriptor if in addition
 * desc_dist <= param.desc_tolerance, otherwise Position.
 * Then best_at_position, the highest level among the pairs whose keypoint of
 * has exactly the same position and scale.
 *
 * @param[in,out] pairs result of matchKeypointsByPosition( l_A, l_B, ... )
 * @param[in]     l_A   first keypoint list
 * @param[in]     l_B   second keypoint list
 * @param[in]     param tolerances
 */
void classifyAgreement( std::vector<KeypointPair>& pairs,
                        const feat_vector_t&       l_A,
                        const feat_vector_t&       l_B,
                        const PairParams&          param );

/** The descriptor nearest neighbour of one feature in another list.
 */
struct NearestDesc
{
    size_t index  = 0;        ///< index into the other list
    float  dist   = INFINITY; ///< descriptor distance to it
    float  second = INFINITY; ///< distance to the second nearest; INFINITY if there is none
};

/** For every feature of l_query, the feature of l_other with the smallest
 * descDistance(); the lower index wins ties. Brute force, parallel where
 * available.
 * @param[out] nearest_out cleared, then one entry per feature of l_query
 * @param[in]  l_query     features to search for
 * @param[in]  l_other     features to search in
 * @pre l_query and l_other are not empty (asserted)
 * @pre all descriptor values are finite
 */
void nearestDescriptors( std::vector<NearestDesc>& nearest_out,
                         const feat_vector_t&      l_query,
                         const feat_vector_t&      l_other );

/** Writes the nearest neighbours of l_A in l_B to path, one row per
 * feature of l_A:
 *   q_index,q_x,q_y,q_sigma,q_ori_deg,
 *   n_index,n_x,n_y,n_sigma,n_ori_deg,
 *   desc_dist,second_desc_dist,pos_dist,ori_diff_deg,colocated_index
 * note:
 *   colocated_index is the keypoint of l_B paired with the query
 *   keypoint, or -1.
 *   directory for path must exist, existing files are overwritten
 *
 * @param[in] path              output file
 * @param[in] l_A               query features
 * @param[in] l_B               searched features
 * @param[in] nearest_of_A      result of nearestDescriptors( ..., l_A, l_B )
 * @param[in] pairs             pairs between the two lists
 * @param[in] pairs_query_first true if KeypointPair::a indexes l_A,
 *                              false if KeypointPair::b does
 * @return false if the file cannot be opened or written (a partial file may remain)
 */
bool writeNearestFile( const std::filesystem::path&     path,
                       const feat_vector_t&             l_A,
                       const feat_vector_t&             l_B,
                       const std::vector<NearestDesc>&  nearest_of_A,
                       const std::vector<KeypointPair>& pairs,
                       bool                             pairs_query_first );

/** Writes several CSV files into dir, overwriting old files:
 *
 * colocated.csv: one row per pair in the order of pairs:
 *   a_index,a_x,a_y,a_sigma,a_ori_deg,
 *   b_index,b_x,b_y,b_sigma,b_ori_deg,
 *   dx,dy,pos_dist,scale_ratio,ori_diff_deg,desc_dist,agreement
 * note:
 *   (dx = b.x - a.x, scale_ratio = b.sigma/a.sigma or 0 if a.sigma is 0,
 *   agreement is position, orientation or descriptor);
 *
 * only_a.csv and only_b.csv, the unpaired keypoints of l_A and l_B in index
 *   order: index,x,y,sigma,ori_deg
 *
 * @param[in] dir   existing output directory
 * @param[in] l_A   first keypoint list
 * @param[in] l_B   second keypoint list
 * @param[in] pairs classified pairs between l_A and l_B
 * @return false as soon as one file cannot be opened or written; files
 *         before it remain
 */
bool writePairFiles( const std::filesystem::path&     dir,
                     const feat_vector_t&             l_A,
                     const feat_vector_t&             l_B,
                     const std::vector<KeypointPair>& pairs );

/** Writes desc_bins.csv into dir, overwriting old files:
 *
 * subset,element,pairs,sum_diff,sum_abs_diff
 *
 * For each of the DescSize elements, the number of pairs and the sums of
 * b.desc - a.desc and of its absolute value, over all pairs (subset
 * colocated) and over the pairs with level Orientation or Descriptor (subset
 * same_orientation). Always 2*DescSize rows.
 *
 * @param[in] dir   existing output directory
 * @param[in] l_A   first keypoint list
 * @param[in] l_B   second keypoint list
 * @param[in] pairs classified pairs between l_A and l_B
 * @return false if the file cannot be opened or written
 */
bool writeDescBinsFile( const std::filesystem::path&     dir,
                        const feat_vector_t&             l_A,
                        const feat_vector_t&             l_B,
                        const std::vector<KeypointPair>& pairs );
