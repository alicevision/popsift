#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>

#include "csf_config.h"

// Execution policy argument for the parallel algorithms. Without them
// (see cmake/CheckParallelExecution.cmake) the algorithms run sequentially.
#if CSF_IS_DEFINED(CSF_HAVE_PAR_EXECUTION)
#include <execution>
#define PAR_UNSEQ std::execution::par_unseq,
#else
#define PAR_UNSEQ
#endif

/// Number of elements of a SIFT descriptor.
constexpr int DescSize = 128;

/// A descriptor, or DescSize per-element values such as sums of differences.
using desc_t = std::array<float,DescSize>;

/// Factor from radians to degrees.
constexpr float RAD2DEG = 180.0f / 3.14159265358979323846f;

class feat_t;

/** A list of features, such as all features of one descriptor file.
 * Indices into it are int: main() rejects inputs with more than
 * std::numeric_limits<int>::max() features, so every function that takes a
 * feat_vector_t may assume that bound.
 */
using feat_vector_t = std::vector<feat_t>;

/** One keypoint with its descriptor, as read from one line of a descriptor
 * file.
 */
class feat_t
{
public:
    /** true: descriptor distances are L2 (square root of the sum of squared
     * differences); false: L1 (sum of absolute differences). Default true.
     * Global state read by all distance functions; set it with
     * setUseL2Distance() before any of them run, not concurrently.
     */
    static bool _use_l2_distance;

    float  x;     ///< column in pixels
    float  y;     ///< row in pixels
    float  sigma; ///< scale in pixels
    float  ori;   ///< orientation in radians, in [0, 2*pi]
    desc_t desc;  ///< descriptor values

    /** Builds a feature from the numbers of one line.
     * Layout for num == DescSize+4: x y sigma ori desc[DescSize]; ori is
     * wrapped into [0, 2*pi].
     * Layout for num == DescSize+5: x y a b c desc[DescSize] (affine format);
     * sigma = sqrt(1/a), or 0 if a == 0; b and c are ignored; ori = 0.
     * @param[in] num   DescSize+4 or DescSize+5
     * @param[in] input at least num values
     * @pre ori is finite (otherwise wrapping does not terminate)
     */
    feat_t( int num, const std::vector<float>& input );

    /** Prints "(x,y) sigma=S ori=O" followed by the DescSize descriptor
     * values, separated by spaces, without a newline.
     * @param[in,out] ostr output stream
     */
    void print( std::ostream& ostr ) const;

    /** Prints the distances between this and all descriptors in l_B,
     * marking the best (closest) match.
     * First prints a line "==========", then one line per feature r of l_B with
     * descDistance(*this,r), " MIN " for the feature smallest distance,
     * l2PixDist(r), |sigma - r.sigma| and signedAngleDiffDegrees(r).
     * @param[in,out] ostr output stream
     * @param[in]     l_B  features to compare with
     * @return the smallest descriptor distance, or 0 if l_B is empty
     */
    float compareAll( std::ostream&              ostr,
                      const feat_vector_t&       l_B ) const;

    /** Prints to ostr, without a newline: "desc dist d MIN pixdist P
     * scaledist S angledist A" for this feature and r. If dstr is not null,
     * also prints the DescSize differences desc[i] - r.desc[i] as one line
     * to *dstr and adds them to desc_stats[i].
     * @param[in,out] ostr       output stream
     * @param[in,out] dstr       stream for descriptor differences, or nullptr
     * @param[in]     r          the best match of this feature
     * @param[in]     d          descriptor distance to r, printed as given
     * @param[in,out] desc_stats sums of the differences; unchanged if dstr is null
     */
    void printBestMatch( std::ostream&       ostr,
                         std::ostream*       dstr,
                         const feat_t&       r,
                         float               d,
                         desc_t&             desc_stats ) const;

    /** Sets _use_l2_distance.
     * @param[in] onoff true for L2, false for L1
     */
    static void setUseL2Distance( bool onoff );

    /** @param[in] other second feature
     * @return Euclidean distance of the two positions in pixels
     */
    float l2PixDist( const feat_t& other ) const;

    /** @param[in] other second feature
     * @return other.ori - ori in degrees, wrapped into [-180, 180)
     */
    float signedAngleDiffDegrees( const feat_t& other ) const;
};

/** Reads all lines of f_one with addFeat() and appends the features to l_one.
 * Reading continues after a malformed line; each one is reported on cerr.
 * @param[in,out] l_one features are appended; not cleared first
 * @param[in,out] f_one input stream, read to its end
 * @return l_one.size() after reading, or -1 if any line is malformed
 * @note The result is undefined above std::numeric_limits<int>::max()
 *       features; main() checks that bound only after reading.
 */
int readFeats( feat_vector_t& l_one,
               std::ifstream&       f_one );

/** Parses one line of whitespace-separated floats and appends a feature.
 * @param[in,out] features the new feature is appended
 * @param[in]     line     one line of a descriptor file
 * @return true if the line holds DescSize+4 or DescSize+5 numbers, or no
 *         numbers at all (empty or whitespace only, nothing appended);
 *         false otherwise, with features unchanged. A wrong count is also
 *         reported on cerr.
 */
bool addFeat( feat_vector_t& features, const std::string& line );

/** Descriptor distance, L2 or L1 as set by feat_t::setUseL2Distance().
 * @param[in] l first feature
 * @param[in] r second feature
 * @return the distance, >= 0, symmetric; NaN if a descriptor holds NaN
 */
float descDistance( const feat_t& l, const feat_t& r );

/** The sum behind descDistance(): of squared differences for L2, of absolute
 * differences for L1. Stops early once the partial sum exceeds bound
 * (checked every 16 elements), which speeds up nearest neighbour searches.
 * @param[in] l     first feature
 * @param[in] r     second feature
 * @param[in] bound INFINITY for the full sum
 * @return the full sum if it is <= bound, otherwise some partial sum > bound
 */
float descDistanceSum( const feat_t& l, const feat_t& r, float bound );

/** Turns a full descDistanceSum() into the distance descDistance() returns.
 * @param[in] sum a sum from descDistanceSum()
 * @return sqrt(sum) for L2, sum for L1
 * @pre sum >= 0 (asserted)
 */
float descDistanceFromSum( float sum );
