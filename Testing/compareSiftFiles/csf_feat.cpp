#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>

#include "csf_feat.h"

#undef HAVE_STD_TRANSFORM_REDUCE

using namespace std;

const float M_PI2 = 2.0f * 3.14159265358979323846f;

bool feat_t::_use_l2_distance = true;

float descDistance( const feat_t& l, const feat_t& r )
{
    return descDistanceFromSum( descDistanceSum( l, r, INFINITY ) );
}

float descDistanceSum( const feat_t& l, const feat_t& r, float bound )
{
    // The bound is checked once per block, so that the compiler can still
    // vectorize the inner loop.
    const int Block = 16;
    const float* a = l.desc.data();
    const float* b = r.desc.data();
    float sum = 0.0f;

    for( int blk=0; blk<DescSize; blk+=Block )
    {
        if( feat_t::_use_l2_distance )
        {
            for( int i=blk; i<blk+Block; i++ )
            {
                const float val = a[i] - b[i];
                sum += ( val * val );
            }
        }
        else
        {
            for( int i=blk; i<blk+Block; i++ )
            {
                sum += fabsf( a[i] - b[i] );
            }
        }
        if( sum > bound ) break;
    }
    return sum;
}

float descDistanceFromSum( float sum )
{
    return feat_t::_use_l2_distance ? sqrtf( sum ) : sum / DescSize;
}

int readFeats( vector<feat_t>& l_one, ifstream& f_one )
{
    string line;
    int    line_no = 0;
    bool   malformed = false;

    // A line holds 132 or 133 numbers, often well over 1000 characters.
    while( getline( f_one, line ) )
    {
        line_no++;
        if( ! addFeat( l_one, line ) )
        {
            cerr << "Line " << line_no << " is not a feature" << endl;
            malformed = true;
        }
    }
    return malformed ? -1 : int( l_one.size() );
}

bool addFeat( vector<feat_t>& features, const string& line )
{
    vector<float> values; // 4 or 5 values followed by DescSize desc values
    values.reserve( 5+DescSize );

    istringstream s( line );
    float v;
    while( s >> v )
    {
        values.push_back( v );
    }
    if( ! s.eof() ) return false; // something that is not a number

    const int num = values.size();
    if( num == 0 ) return true;   // empty line
    if( num != DescSize+4 && num != DescSize+5 )
    {
        cerr << "The keypoint line contains an unexpected number of floats (" << num << ")" << endl;
        return false;
    }

    features.emplace_back( num, values );

    return true;
}

float angleDistDegrees( float ori_a, float ori_b )
{
    float d = fmodf( fabsf( ori_a - ori_b ), M_PI2 );
    if( d > M_PI2 / 2.0f ) d = M_PI2 - d;
    return d / M_PI2 * 360.0f;
}

feat_t::feat_t( int num, const vector<float>& input )
    : desc(DescSize)
{
    auto it = input.begin();
    auto to = desc.begin();
    if( num == DescSize+4 )
    {
        x     = *it++;
        y     = *it++;
        sigma = *it++;
        ori   = *it++;
        for( int i=0; i<DescSize; i++ ) *to++ = *it++;

        /* make sure orientation is between 0 and M_PI2 */
        while( ori < 0 ) ori += M_PI2;
        while( ori > M_PI2 ) ori -= M_PI2;
    }
    else // num == DescSize+5, checked by addFeat
    {
        float odbss;
        x     = *it++;
        y     = *it++;
        odbss = *it++;
        sigma = odbss == 0.0f ? 0.0f : sqrtf( 1.0f / odbss );
        ori   = 0.0f;
        it++;
        it++;
        for( int i=0; i<DescSize; i++ ) *to++ = *it++;
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

float feat_t::compareAll( ostream& ostr, const vector<feat_t>& l_one ) const
{
    vector<float> distances( l_one.size() );
    transform( PAR_UNSEQ l_one.begin(), l_one.end(), distances.begin(),
               [this]( const feat_t& r ) { return descDistance( *this, r ); } );

    const auto m = min_element( distances.begin(), distances.end() );

    ostr << "==========" << endl;
    for( size_t i = 0; i < l_one.size(); i++ )
    {
        const feat_t& r = l_one[i];
        ostr << "desc dist " << distances[i]
             << ( i == size_t( m - distances.begin() ) ? " MIN " : "     " )
             << " pixdist " << sqrtf( (x-r.x)*(x-r.x) + (y-r.y)*(y-r.y) )
             << " scaledist " << fabsf( sigma - r.sigma )
             << " angledist " << angleDistDegrees( ori, r.ori )
             << endl;
    }
    return m == distances.end() ? 0.0f : *m;
}

void feat_t::printBestMatch( ostream& ostr, ostream* dstr, const feat_t& r, float d, vector<float>& desc_stats ) const
{
    ostr << "desc dist " << d
         << " MIN"
         << " pixdist " << sqrtf( (x-r.x)*(x-r.x) + (y-r.y)*(y-r.y) )
         << " scaledist " << fabsf( sigma - r.sigma )
         << " angledist " << angleDistDegrees( ori, r.ori );

    if( dstr )
    {
        for( int i=0; i<DescSize; i++ )
        {
            const float diff = desc[i] - r.desc[i];
            (*dstr) << diff << " ";
            desc_stats[i] += diff;
        }
        (*dstr) << endl;
    }
}

void feat_t::setL2Distance( bool onoff )
{
    _use_l2_distance = onoff;
}

