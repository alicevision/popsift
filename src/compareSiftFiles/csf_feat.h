#pragma once

#include <iostream>
#include <fstream>
#include <vector>

#define DESC_SIZE 128

typedef std::vector<float> desc_t;

class feat_t
{
public:
    static bool _use_l2_distance;

    float  x;
    float  y;
    float  sigma;
    float  ori;
    desc_t desc;

    feat_t( int num, const std::vector<float>& input );

    void print( std::ostream& ostr ) const;

    void compareBestMatch( std::ostream&              ostr,
                           std::ostream*              dstr,
                           const std::vector<feat_t>& l_one,
                           std::vector<float>&        desc_stats,
                           bool                       minOnly ) const;

    static void setL2Distance( bool onoff );
};

int readFeats( std::vector<feat_t>& l_one,
               std::ifstream&       f_one );
bool addFeat( std::vector<feat_t>& features, char* line );

