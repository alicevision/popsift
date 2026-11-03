#pragma once

#include <execution>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// Execution policy argument for the parallel algorithms. Standard libraries
// without them (e.g. libc++ before 17) do not define __cpp_lib_execution;
// the algorithms then run sequentially.
#if defined(__cpp_lib_execution) && __cpp_lib_execution >= 201603L
#define PAR_UNSEQ std::execution::par_unseq,
#else
#define PAR_UNSEQ
#endif

const int DescSize = 128;

typedef std::vector<float> desc_t;

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
     * Print the distances between this and all descriptors in l_one, marking
     * the best (closest) match.
     * @note The distance metric is determined by _use_l2_distance.
     * @return the descriptor distance of the best match
     */
    float compareAll( std::ostream&              ostr,
                      const std::vector<feat_t>& l_one ) const;

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

    static void setL2Distance( bool onoff );
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

/** Absolute difference of two orientations in degrees, wrapped to [0,180].
 */
float angleDistDegrees( float ori_a, float ori_b );

