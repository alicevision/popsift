#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// Execution policy argument for the parallel algorithms. libstdc++ 9 runs
// <execution> on TBB and does not compile without the TBB headers, so it
// is not included there. Standard libraries without the parallel
// algorithms (e.g. libc++ before 17) do not define __cpp_lib_execution.
// In both cases the algorithms run sequentially.
#if defined(__GLIBCXX__) && _GLIBCXX_RELEASE < 10 && !__has_include(<tbb/blocked_range.h>)
#define PAR_UNSEQ
#else
#include <execution>
#if defined(__cpp_lib_execution) && __cpp_lib_execution >= 201603L
#define PAR_UNSEQ std::execution::par_unseq,
#else
#define PAR_UNSEQ
#endif
#endif

const int DescSize = 128;

typedef std::vector<float> desc_t;

static const float RAD2DEG = 180.0f / 3.14159265358979323846f;

class feat_t
{
public:
    /* Indicator to compute descriptor distance as L2 (square root of sum of squares)
     * or as average absolute difference. Changed by setL2Distance().
     */
    static bool _use_l2_distance;

    float  x;
    float  y;
    float  sigma;
    float  ori;
    desc_t desc;

    feat_t( int num, const std::vector<float>& input );

    void print( std::ostream& ostr ) const;

    /**
     * Print the distances between this and all descriptors in l_B, marking
     * the best (closest) match.
     * @note The distance metric is determined by _use_l2_distance.
     * @return the descriptor distance of the best match
     */
    float compareAll( std::ostream&              ostr,
                      const std::vector<feat_t>& l_B ) const;

    /**
     * Print one line about the best match r at descriptor distance d. If dstr
     * is given, print the element-wise descriptor differences to it and add
     * them to desc_stats.
     */
    void printBestMatch( std::ostream&       ostr,
                         std::ostream*       dstr,
                         const feat_t&       r,
                         float               d,
                         std::vector<float>& desc_stats ) const;

    static void setUseL2Distance( bool onoff );

    /** Compute L2 distance between two features.
     */
    float l2PixDist( const feat_t& other ) const;

    /** Compute rotation distance in degree between two features.
     */
    float signedAngleDiffDegrees( const feat_t& other ) const;
};

/** Read all feature lines of a file into l_one.
 * @return the number of features read, or -1 if a line is malformed
 */
int readFeats( std::vector<feat_t>& l_one,
               std::ifstream&       f_one );

/** Parse one line. Empty lines are skipped and return true.
 * @return false if the line is not a feature with 4 or 5 values plus a descriptor
 */
bool addFeat( std::vector<feat_t>& features, const std::string& line );

/** Descriptor distance of two features, L2 or L1 as set by setL2Distance().
 */
float descDistance( const feat_t& l, const feat_t& r );

/** The sum behind descDistance(): of squares for L2, of absolute values for
 * L1. Stops early and returns a value > bound once the partial sum exceeds
 * bound, which makes nearest neighbour searches much faster.
 */
float descDistanceSum( const feat_t& l, const feat_t& r, float bound );

/** Turns a descDistanceSum() into the distance descDistance() returns.
 */
float descDistanceFromSum( float sum );

