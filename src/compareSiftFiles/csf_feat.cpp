#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "csf_feat.h"

#if __cplusplus >= 201703L
#define PAR_UNSEQ std::execution::par_unseq,
#else
#define PAR_UNSEQ
#endif

#undef HAVE_STD_TRANSFORM_REDUCE

using namespace std;

const float M_PI2 = 2.0f * 3.14159265358979323846f;

bool feat_t::_use_l2_distance = true;

inline static float sum_of_squares( const feat_t& l, const feat_t& r )
{
#ifdef HAVE_STD_TRANSFORM_REDUCE
/* From C++17 numeric:
template<class ExecutionPolicy,
         class ForwardIt1, class ForwardIt2, class T, class BinaryOp1, class BinaryOp2>
T transform_reduce(ExecutionPolicy&& policy,
                   ForwardIt1 first1, ForwardIt1 last1, ForwardIt2 first2,
                   T init, BinaryOp1 binary_op1, BinaryOp2 binary_op2);
*/
    const float sum = std::transform_reduce(
                                     PAR_UNSEQ
                                     l.desc, l.desc+DESC_SIZE,
                                     r.desc,
                                     0.0f,
                                     std::plus<>(),
                                     [](float l, float r){ float v=l-r; return (v*v); },
                                     std::plus );
#else
    float sum = 0.0f;

    for( int i=0; i<DESC_SIZE; i++ )
    {
        const float val = l.desc[i] - r.desc[i];
        sum += ( val * val );
    }
#endif
    return sum;
}

static float dist( const feat_t& l, const feat_t& r )
{
    if( feat_t::_use_l2_distance )
    {
        const float sum = sum_of_squares( l, r );
        return sqrtf( sum );
    }
    else
    {
#ifdef HAVE_STD_TRANSFORM_REDUCE
        const float sum = std::transform_reduce(
                                     PAR_UNSEQ
                                     l.desc, l.desc+DESC_SIZE,
                                     r.desc,
                                     0.0f,
                                     std::plus<>(),
                                     [](float l, float r){ float v=l-r; return fabsf(v); },
                                     std::plus );
#else
        float sum = 0.0f;

        for( int i=0; i<DESC_SIZE; i++ )
        {
            const float val = l.desc[i] - r.desc[i];
            sum += fabsf( val );
        }
#endif
        return sum / DESC_SIZE;
    }
}

int readFeats( vector<feat_t>& l_one, ifstream& f_one )
{
    char buffer[1024];
    int  lines_read;
    
    lines_read = 0;
    while( f_one.good() )
    {
        f_one.getline( buffer, 1024 );
        if( f_one.good() )
        {
            bool success = addFeat( l_one, buffer );
            if( success )
            {
                lines_read++;
            }
        }
    }
    return lines_read;
}

bool addFeat( vector<feat_t>& features, char* line )
{
    vector<float> values(5+DESC_SIZE); // 4 or 5 values followed by DESC_SIZE desc values

    int i = 0;
    istringstream s( line );
    while( s >> values[i] )
    {
        i++;
    }

    // cerr << "Found " << i << " floats in line" << endl;
    features.emplace_back( i, values );

    if( i == 0 ) return false;
    return true;
}

feat_t::feat_t( int num, const vector<float>& input )
    : desc(DESC_SIZE)
{
    auto it = input.begin();
    auto to = desc.begin();
    if( num == DESC_SIZE+4 )
    {
        x     = *it++;
        y     = *it++;
        sigma = *it++;
        ori   = *it++;
        for( int i=0; i<DESC_SIZE; i++ ) *to++ = *it++;
    }
    else if( num == DESC_SIZE+5 )
    {
        float odbss;
        x     = *it++;
        y     = *it++;
        odbss = *it++;
        sigma = odbss == 0.0f ? 0.0f : sqrtf( 1.0f / odbss );
        ori   = 0.0f;
        it++;
        it++;
        for( int i=0; i<DESC_SIZE; i++ ) *to++ = *it++;
    }
    else
    {
        cerr << "The keypoint line contains an unexpected number of floats (" << num << ")" << endl;
        return;
    }
}

void feat_t::print( ostream& ostr ) const
{
    ostr << "(" << x << "," << y << ")";
    ostr << " sigma=" << sigma << " ori=" << ori;
    for( auto it : desc )
    {
        ostr << " " << it;
    }
}

struct Find2Best
{
    float min1 { std::numeric_limits<float>::infinity() };
    float min2 { std::numeric_limits<float>::infinity() };

    int   idx1;
    int   idx2;

    inline void update( const int& idx, const float& val )
    {
        if( val < min1 )
        {
            min2 = min1;
            idx2 = idx1;
            min1 = val;
            idx1 = idx;
        }
        else if( val < min2 )
        {
            min2 = val;
            idx2 = idx;
        }
    }

    void takeSqrt( )
    {
        min1 = sqrtf( min1 );
        min2 = sqrtf( min2 );
    }
};

void feat_t::compareBestMatch( ostream& ostr, ostream* dstr, const vector<feat_t>& l_one, vector<float>& desc_stats,  bool minOnly ) const
{
    const int l_one_sz = l_one.size();

    if( !minOnly ) ostr << "==========" << endl;

    const feat_t& left( *this );

    if( minOnly )
    {
        Find2Best find2best;

        for( int j=0; j<l_one_sz; j++ )
        {
            const float val =  sum_of_squares( left, l_one[j] );
            find2best.update( j, val );
        }

        find2best.takeSqrt();

        const feat_t& r = l_one[find2best.idx1];

        ostr << "desc dist " << find2best.min1
             << " MIN"
             << " pixdist " << sqrtf( (x-r.x)*(x-r.x) + (y-r.y)*(y-r.y) )
             << " angledist " << fabsf( ori/M_PI2*360.0f - r.ori/M_PI2*360.0f );

        if( dstr )
        {
            auto right = r.desc;
            for( int i=0; i<DESC_SIZE; i++ )
            {
                const float d = left.desc[i] - right[i];
                (*dstr) << d << " ";
                desc_stats[i] += d;
            }
            (*dstr) << endl;
        }

        ostr << " 2best " << find2best.min2 << endl;
    }
    else
    {
        vector<float> distances( l_one_sz );

        for( int j=0; j<l_one_sz; j++ )
        {
            distances[j] = dist( left, l_one[j] );
        }

        auto m = min_element( PAR_UNSEQ
                              distances.begin(),
                              distances.end() );

        auto it = distances.begin();

        for( int j=0; j<l_one_sz; j++ )
        {
            const feat_t& r = l_one[j];

            ostr << "desc dist " << *it;
            if( it == m )
                ostr << " MIN ";
            else
                ostr << "     ";
            it++;
            ostr << " pixdist " << sqrtf( (x-r.x)*(x-r.x) + (y-r.y)*(y-r.y) )
                 << " angledist " << fabsf( ori/M_PI2*360.0f - r.ori/M_PI2*360.0f )
                 << endl;
        }
    }
}

void feat_t::setL2Distance( bool onoff )
{
    _use_l2_distance = onoff;
}

