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

/** Read a file containing SIFT descriptors
 */
void readFile( vector<feat_t>& vec, const std::string& filename );

/** Write the average descriptor differences to file
 * @param file The output file
 * @param stats A 128-float vector containing the sum of differences
 * @param sz The number of samples that have been collected in stats
 */
void writeSummary( ostream& file, const vector<float>& stats, int sz );

/** Open a new stream or return the default stream. The default stream is
 *  returned if the name is empty or opening the stream fails.
 * @param name A string containing the name of the file to open or empty
 * @param default_stream The default stream to return
 */
ostream* openOutputFile( const string& name, ostream* default_stream );

int main( int argc, char* argv[] )
{
    Parameters param;

    parseargs( argc, argv, param );

    vector<feat_t> l_one;
    vector<feat_t> l_two;

    readFile( l_one, param.input[0] );
    readFile( l_two, param.input[1] );

    ostream* outfile  = openOutputFile( param.outfile_name, &cout );
    ostream* descfile = openOutputFile( param.descfile_name, nullptr );

    int len = l_one.size();
    int ct = 0;
    float nextpercent = 10;

    vector<float> desc_stats( 128, 0.0f );

    ostream* print_dists = param.descfile_verbose ? descfile : 0;

    for( auto l : l_one )
    {
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
        writeSummary( *descfile, desc_stats, l_one.size() );
    }

    if( ! param.outfile_name.empty() )
    {
        delete outfile;
    }
}

void writeSummary( ostream& descfile, const vector<float>& desc_stats, int sz )
{
    descfile << "========== Summary Stats ==========" << endl
             << "Average values:" << endl
             << setprecision(3);
    for( int i=0; i<128; i++ )
    {
        if( i%32==0  ) descfile << "X=0 | ";
        if( i%32==8  ) descfile << "X=1 |  ";
        if( i%32==16 ) descfile << "X=2 |   ";
        if( i%32==24 ) descfile << "X=3 |    ";
        float d = desc_stats[i] / sz;
        descfile << setw(8) << d << " ";
        if( i%8==7 ) descfile << endl;
    }
    descfile << endl;
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

ostream* openOutputFile( const string& outfile_name, ostream* default_stream )
{
    ostream* outfile = default_stream;
    if( outfile_name.empty() ) return outfile;

    ostream* o = new ofstream( outfile_name );
    if( o->good() )
    {
        outfile = o;
    }
    else
    {
        delete o;
    }

    return outfile;
}

