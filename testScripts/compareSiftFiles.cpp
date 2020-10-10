#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

using namespace std;

const float M_PI2 = 2.0f * 3.14159265358979323846f;

typedef vector<float> desc_t;

struct feat_t
{
    float  x;
    float  y;
    float  sigma;
    float  ori;
    desc_t desc;

    feat_t( int num, const vector<float>& input );

    void print( ostream& ostr ) const;

    void compareBestMatch( const vector<feat_t>& r, bool minOnly ) const;

private:
    float dist( const feat_t& r ) const;
};

int readFeats( vector<feat_t>& l_one, fstream& f_one );
bool addFeat( vector<feat_t>& features, char* line );
void usage( char* name );

bool use_l2_distance = true;

int main( int argc, char* argv[] )
{
    bool briefinfo = true;

    char* name = argv[0];
    while( argc > 3 )
    {
        if( argv[1][0] == '-' )
        {
            if( !strcmp( argv[1], "-v" ) )
            {
                briefinfo = false;
            }
            else if( !strcmp( argv[1], "-l1" ) )
            {
                use_l2_distance = false;
            }
            else
            {
                break;
            }
            argc--;
            argv++;
        }
    }
    if( argc != 3 ) usage( name );

    fstream f_one( argv[1], fstream::in );
    fstream f_two( argv[2], fstream::in );

    if( ! f_one.good() )
    {
        cerr << "File " << argv[1] << " is not open." << endl;
        exit( -1 );
    }
    if( ! f_two.good() )
    {
        cerr << "File " << argv[2] << " is not open." << endl;
        exit( -1 );
    }

    vector<feat_t> l_one;
    vector<feat_t> l_two;

    int lines_read;

    lines_read = readFeats( l_one, f_one );
    cerr << "Read " << lines_read << " lines from " << argv[1] << endl;
    lines_read = readFeats( l_two, f_two );
    cerr << "Read " << lines_read << " lines from " << argv[2] << endl;

#if 0
    for( auto l : l_one )
    {
        l.print( cout );
        cout << endl;
    }
    for( auto l : l_two )
    {
        l.print( cout );
        cout << endl;
    }
#endif

    int len = l_one.size();
    int ct = 0;
    float nextpercent = 10;
    for( auto l : l_one )
    {
        l.compareBestMatch( l_two, briefinfo );
        ct++;
        if( float(ct * 100) / len >= float(nextpercent) )
        {
            cerr << nextpercent << "% " <<  ct << endl;
            nextpercent += 10;
        }
    }
}

void usage( char* name )
{
    cerr << "Usage: " << name << " [options] <descriptorfile> <descriptorfile>" << endl
         << "       compute the L1 and L2 distance between the descriptors of" << endl
         << "       closest coordinate pairs. When a coordinate has 2 several" << endl
         << "       orientations, the closest distance is reported." << endl
         << "       Summary information at the end." << endl
         << "Options:" << endl
         << "       -v : verbose, longer output" << endl
         << "       -l1 : use L1 for distance computation instead of L2" << endl
         << endl;
    exit( 0 );
}

int readFeats( vector<feat_t>& l_one, fstream& f_one )
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

void feat_t::compareBestMatch( const vector<feat_t>& l_one, bool minOnly ) const
{
    vector<float> distances;
    distances.reserve( l_one.size() );

    if( !minOnly ) cout << "==========" << endl;
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
                cout << "desc dist " << *it
                     << " MIN"
                     << " pixdist " << sqrtf( (x-r.x)*(x-r.x) + (y-r.y)*(y-r.y) )
                     << " angledist " << fabsf( ori/M_PI2*360.0f - r.ori/M_PI2*360.0f );
            }
            else if( *it > *m )
            {
                second = min<float>( second, *it );
            }
            it++;
        }
        else
        {
            cout << "desc dist " << *it;
            if( it == m )
                cout << " MIN ";
            else
                cout << "     ";
            it++;
            cout << " pixdist " << sqrtf( (x-r.x)*(x-r.x) + (y-r.y)*(y-r.y) )
                << " angledist " << fabsf( ori/M_PI2*360.0f - r.ori/M_PI2*360.0f )
                << endl;
        }
    }
    if( minOnly )
    {
        cout << " 2.best " << second << endl;
    }
}

float feat_t::dist( const feat_t& r ) const
{
    if( use_l2_distance )
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

