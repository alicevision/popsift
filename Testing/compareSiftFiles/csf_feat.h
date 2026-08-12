#pragma once

#include <iostream>
#include <fstream>
#include <vector>

const int DescSize = 128;

typedef std::vector<float> desc_t;

class feat_t
{
    /* Indicator to compute descriptor distance as L2 (square root of sum of squares)
     * or as average absolute difference. Changed by setL2Distance().
     */
    static bool _use_l2_distance;

public:
    float  x;
    float  y;
    float  sigma;
    float  ori;
    desc_t desc;

    feat_t( int num, const std::vector<float>& input );

    void print( std::ostream& ostr ) const;

    /**
     * Find best descriptors match between this and all descriptors in l_one.
     * @note The distance metric is determined by _use_l2_distance.
     */
    // Returns the descriptor distance of the best (closest) match, so that
    // callers can aggregate a pass/fail metric across all features.
    float compareBestMatch( std::ostream&              ostr,
                            std::ostream*              dstr,
                            const std::vector<feat_t>& l_one,
                            std::vector<float>&        desc_stats,
                            bool                       minOnly ) const;

    static void setL2Distance( bool onoff );
};

int readFeats( std::vector<feat_t>& l_one,
               std::ifstream&       f_one );
bool addFeat( std::vector<feat_t>& features, char* line );

