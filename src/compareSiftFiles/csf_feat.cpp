#include <sstream>
#include <cmath>
#include <algorithm>

#include "csf_feat.h"

using namespace std;

const float M_PI2 = 2.0f * 3.14159265358979323846f;

bool feat_t::_use_l2_distance = true;

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
    vector<float> values(5+128); // 4 or 5 values followed by 128 desc values

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
    : desc(128)
{
    auto it = input.begin();
    auto to = desc.begin();
    if( num == 132 )
    {
        x     = *it++;
        y     = *it++;
        sigma = *it++;
        ori   = *it++;
        for( int i=0; i<128; i++ ) *to++ = *it++;
    }
    else if( num == 133 )
    {
        float odbss;
        x     = *it++;
        y     = *it++;
        odbss = *it++;
        sigma = odbss == 0.0f ? 0.0f : sqrtf( 1.0f / odbss );
        ori   = 0.0f;
        it++;
        it++;
        for( int i=0; i<128; i++ ) *to++ = *it++;
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

void feat_t::compareBestMatch( ostream& ostr, ostream* dstr, const vector<feat_t>& l_one, vector<float>& desc_stats,  bool minOnly ) const
{
    vector<float> distances;
    distances.reserve( l_one.size() );

    if( !minOnly ) ostr << "==========" << endl;
    for( auto r : l_one )
    {
        float v = dist( r );
        distances.push_back( v );
    }

    auto it = distances.begin();

    auto m = min_element( distances.begin(), distances.end() );

    float second = INFINITY;

    for( auto r : l_one )
    {
        if( minOnly )
        {
            if( it == m )
            {
                ostr << "desc dist " << *it
                     << " MIN"
                     << " pixdist " << sqrtf( (x-r.x)*(x-r.x) + (y-r.y)*(y-r.y) )
                     << " angledist " << fabsf( ori/M_PI2*360.0f - r.ori/M_PI2*360.0f );

                if( dstr )
                {
                    auto left  = desc.begin();
                    auto right = r.desc.begin();
                    for( int i=0; i<128; i++, left++, right++ )
                    {
                        float diff = *left - *right;
                        (*dstr) << diff << " ";
                        desc_stats[i] += diff;
                    }
                    (*dstr) << endl;
                }
            }
            else if( *it > *m )
            {
                second = min<float>( second, *it );
            }
            it++;
        }
        else
        {
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
    if( minOnly )
    {
        ostr << " 2best " << second << endl;
    }
}

float feat_t::dist( const feat_t& r ) const
{
    if( _use_l2_distance )
    {
        float sum = 0.0f;
        auto it_r = r.desc.begin();
        for( auto l : desc )
        {
            float val = l - *it_r++;
            sum += ( val * val );
        }
        return sqrtf( sum );
    }
    else
    {
        float sum = 0.0f;
        auto it_r = r.desc.begin();
        for( auto l : desc )
        {
            float val = l - *it_r++;
            sum += fabsf( val );
        }
        return sum / 128;
    }
}

void feat_t::setL2Distance( bool onoff )
{
    _use_l2_distance = onoff;
}

