#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

#include <boost/program_options.hpp>

#include "csf_feat.h"
#include "csf_options.h"

using namespace std;

typedef vector<float> desc_t;

void readFile( vector<feat_t>& vec, const std::string& filename );

int main( int argc, char* argv[] )
{
    Parameters param;

    parseargs( argc, argv, param );

    vector<feat_t> l_one;
    vector<feat_t> l_two;

    readFile( l_one, param.input[0] );
    readFile( l_two, param.input[1] );

    ostream* outfile = &cout;
    if( ! param.outfile_name.empty() )
    {
        ostream* o = new ofstream( param.outfile_name );
        if( o->good() )
        {
            outfile = o;
        }
    }

    ostream* descfile = 0;
    if( ! param.descfile_name.empty() )
    {
        ostream* o = new ofstream( param.descfile_name );
        if( o->good() )
        {
            descfile = o;
        }
    }

    int len = l_one.size();
    int ct = 0;
    float nextpercent = 10;

    vector<float> desc_stats( 128, 0.0f );

    for( auto l : l_one )
    {
        ostream* print_dists = param.descfile_verbose ? descfile : 0;

        l.compareBestMatch( *outfile, print_dists, l_two, desc_stats, param.briefinfo );
        ct++;
        if( float(ct * 100) / len >= float(nextpercent) )
        {
            cerr << nextpercent << "% " <<  ct << endl;
            nextpercent += 10;
        }
    }

    if( descfile )
    {
        int sz = l_one.size();
        (*descfile) << "========== Summary Stats ==========" << endl
                    << "Average values:" << endl
                    << setprecision(3);
        for( int i=0; i<128; i++ )
        {
            if( i%32==0  ) (*descfile) << "X=0 | ";
            if( i%32==8  ) (*descfile) << "X=1 |  ";
            if( i%32==16 ) (*descfile) << "X=2 |   ";
            if( i%32==24 ) (*descfile) << "X=3 |    ";
            desc_stats[i] /= sz;
            (*descfile) << setw(8) << desc_stats[i] << " ";
            if( i%8==7 ) (*descfile) << endl;
        }
        (*descfile) << endl;
    }

    if( ! param.outfile_name.empty() )
    {
        delete outfile;
    }
}

void readFile( vector<feat_t>& vec, const std::string& filename )
{
    ifstream infile( filename );

    if( ! infile.good() )
    {
        cerr << "File " << filename << " is not open." << endl;
        exit( -1 );
    }

    int lines_read = readFeats( vec, infile );
    cerr << "Read " << lines_read << " lines from " << filename << endl;
}

