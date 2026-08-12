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

typedef std::pair<float, const vector<feat_t>::const_iterator> dist_loc_t;

static dist_loc_t dist( const feat_t& l, const vector<feat_t>::const_iterator r )
{
    if( feat_t::_use_l2_distance )
    {
        float sum = 0.0f;

        for( int i=0; i<DescSize; i++ )
        {
            const float val = l.desc[i] - r->desc[i];
            sum += ( val * val );
        }
        return sqrtf( sum );
    }
    else
    {
        float sum = 0.0f;

        for( int i=0; i<DescSize; i++ )
        {
            const float val = l.desc[i] - r->desc[i];
            sum += fabsf( val );
        }
        return dist_loc_t( sum / DescSize, r );
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
    vector<float> values(5+DescSize); // 4 or 5 values followed by DescSize desc values

    int i = 0;
    istringstream s( line );
    while( s >> values[i] )
    {
        i++;
    }

    if( i == 0 ) return false;

    // cerr << "Found " << i << " floats in line" << endl;
    features.emplace_back( i, values );

    return true;
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
    else if( num == DescSize+5 )
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

float feat_t::compareBestMatch( ostream& ostr, ostream* dstr, const vector<feat_t>& l_one, vector<float>& desc_stats,  bool minOnly ) const
{
    const int l_one_sz = l_one.size();

    // vector<float> distances( l_one_sz );
    vector<dist_loc_t> distances;
    distances.reserve( l_one.size() );

    if( !minOnly ) ostr << "==========" << endl;

    const feat_t& left( *this );

    for( auto lit = l_one.begin(); lit != l_one.end(); lit++ )
    {
        distances.emplace_back( dist( left, lit ) );
    }

    auto m = min_element( PAR_UNSEQ
                          distances.begin(),
                          distances.end(),
                          []( const dist_loc_t& a, const dist_loc_t& b ) { return a.first < b.first; } );

    if( minOnly )
    {
        auto r = m->second;

        float                second   = INFINITY;
        const vector<float>& left     = desc;

        ostr << "desc dist " << m->first
             << " MIN"
             << " pixdist " << sqrtf( (x-r->x)*(x-r->x) + (y-r->y)*(y-r->y) )
             << " scaledist " << fabsf( sigma - r->sigma )
             << " angledist " << fabsf( ori/M_PI2*360.0f - r->ori/M_PI2*360.0f );

        if( dstr )
        {
            auto right = r->desc;
            for( int i=0; i<DescSize; i++ )
            {
                const float d = left[i] - right[i];
                (*dstr) << d << " ";
                desc_stats[i] += d;
            }
            (*dstr) << endl;
        }
    }
    else
    {
        for( auto it=distance.begin(); it!=distances.end(); it++ )
        {
            const feat_t& r = *it->second;

            ostr << "desc dist " << it->first;
            if( it == m )
                ostr << " MIN ";
            else
                ostr << "     ";
            it++;
            ostr << " pixdist " << sqrtf( (x-r.x)*(x-r.x) + (y-r.y)*(y-r.y) )
                 << " scaledist " << fabsf( sigma - r.sigma )
                 << " angledist " << fabsf( ori/M_PI2*360.0f - r.ori/M_PI2*360.0f )
                 << endl;
        }
    }

    return *m;
}

void feat_t::setL2Distance( bool onoff )
{
    _use_l2_distance = onoff;
}

