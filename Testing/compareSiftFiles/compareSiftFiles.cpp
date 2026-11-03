#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdlib>

#include <boost/program_options.hpp>

#include "csf_feat.h"
#include "csf_match.h"
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

    // The most expensive part: every descriptor of the first file against
    // every descriptor of the second, in parallel where available.
    const vector<NearestDesc> nearest = nearestDescriptors( l_one, l_two );

    if( ! param.match_dir.empty() )
    {
        MatchParams mp;
        mp.radius_sigma    = param.match_radius_sigma;
        mp.radius_min      = param.match_radius_min;
        mp.max_scale_ratio = param.match_max_scale_ratio;
        const vector<MatchPair> pairs = matchFeatures( l_one, l_two, mp );
        if( ! writeMatchFiles( param.match_dir, l_one, l_two, pairs ) )
        {
            cerr << "Cannot write the match files into " << param.match_dir << endl;
            return EXIT_FAILURE;
        }
        if( ! writeDescBinsFile( param.match_dir, l_one, l_two, pairs,
                                 param.same_pose_px, param.same_pose_deg ) )
        {
            cerr << "Cannot write desc_bins.csv into " << param.match_dir << endl;
            return EXIT_FAILURE;
        }
        const string base = param.match_dir + "/";
        if( ! writeNearestFile( base + "desc_nn_a.csv", l_one, l_two,
                                nearest, pairs, true )
         || ! writeNearestFile( base + "desc_nn_b.csv", l_two, l_one,
                                nearestDescriptors( l_two, l_one ), pairs, false ) )
        {
            cerr << "Cannot write the descriptor nearest neighbours into " << param.match_dir << endl;
            return EXIT_FAILURE;
        }
        cerr << "Matched " << pairs.size() << " of " << l_one.size() << " features in "
             << param.input[0] << " and " << l_two.size() << " in " << param.input[1] << endl;
    }

    // One empty input is always a difference. Both empty is only accepted
    // for images where no features are expected.
    if( l_one.empty() != l_two.empty() )
    {
        cerr << "FAIL: " << ( l_one.empty() ? param.input[0] : param.input[1] )
             << " contains no features, but "
             << ( l_one.empty() ? param.input[1] : param.input[0] ) << " does" << endl;
        return EXIT_FAILURE;
    }
    if( l_one.empty() )
    {
        if( ! param.allow_empty )
        {
            cerr << "FAIL: neither input contains features "
                    "(use --allow-empty if that is expected)" << endl;
            return EXIT_FAILURE;
        }
        cerr << "Average best-match descriptor distance: 0" << endl;
        return EXIT_SUCCESS;
    }

    ostream* outfile  = openOutputFile( param.outfile_name, &cout );
    ostream* descfile = openOutputFile( param.descfile_name, nullptr );

    const int len = l_one.size();

    vector<float> desc_stats( DescSize, 0.0f );

    ostream* print_dists = param.descfile_verbose ? descfile : 0;

    float sum_min_dist = 0.0f;

    for( int i = 0; i < len; i++ )
    {
        if( param.briefinfo )
        {
            l_one[i].printBestMatch( *outfile, print_dists, l_two[nearest[i].index],
                                     nearest[i].dist, desc_stats );
            *outfile << endl;
            sum_min_dist += nearest[i].dist;
        }
        else
        {
            sum_min_dist += l_one[i].compareAll( *outfile, l_two );
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

    const float avg_min_dist = ( len > 0 ) ? ( sum_min_dist / len ) : 0.0f;
    cerr << "Average best-match descriptor distance: " << avg_min_dist << endl;

    if( param.max_avg_desc_dist >= 0.0f && avg_min_dist > param.max_avg_desc_dist )
    {
        cerr << "FAIL: average best-match descriptor distance " << avg_min_dist
             << " exceeds threshold " << param.max_avg_desc_dist << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
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
    if( lines_read < 0 )
    {
        cerr << "File " << filename << " contains lines that are not features." << endl;
        exit( EXIT_FAILURE );
    }
    cerr << "Read " << lines_read << " features from " << filename << endl;
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

