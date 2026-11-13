#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <limits>

#include <boost/program_options.hpp>

#include "csf_feat.h"
#include "csf_match.h"
#include "csf_options.h"

using namespace std;

using desc_t = vector<float>;

/** Read a file containing SIFT descriptors
 */
void readFile( vector<feat_t>& vec, const std::string& filename );

/** Write the average descriptor differences to file
 * @param file The output file
 * @param stats A 128-float vector containing the sum of differences
 * @param sz The number of samples that have been collected in stats
 */
void writeSummary( ostream& file, const vector<float>& stats, int sz );

/** Open a new stream or return the default stream. nullptr is
 *  returned if the name is empty or opening the stream fails.
 * @param name A string containing the name of the file to open or empty
 */
static ostream* openOutputFile( const string& name );

static float distanceSumBrief( const Parameters& param, ostream* outfile, const vector<feat_t>& l_A, const vector<feat_t>& l_B, const vector<NearestDesc>& nearest_A_B );

static float distanceSum( ostream* outfile, const vector<feat_t>& l_A, const vector<feat_t>& l_B );

static bool writeFiles( const Parameters& param,
                        const vector<NearestDesc>& nearest_A_B,
                        const vector<NearestDesc>& nearest_B_A,
                        const vector<feat_t>& l_A,
                        const vector<feat_t>& l_B )
{
    if( param.pair_directory.empty() ) return true;

    /* writeFiles is never called if l_A or l_B is empty */
    assert( !l_A.empty() );
    assert( !l_B.empty() );

    PairParams pp;
    pp.radius_sigma      = param.pair_radius_sigma;
    pp.radius_min        = param.pair_radius_min;
    pp.max_scale_ratio   = param.pair_max_scale_ratio;
    pp.ori_tolerance_deg = param.ori_tolerance_deg;
    pp.desc_tolerance    = param.desc_tolerance;
    vector<KeypointPair> pairs = matchKeypointsByPosition( l_A, l_B, pp );
    classifyAgreement( pairs, l_A, l_B, pp );

    if( ! writePairFiles( param.pair_directory, l_A, l_B, pairs ) )
    {
            cerr << "Cannot write the pair files into " << param.pair_directory << endl;
            return false;
    }
    if( ! writeDescBinsFile( param.pair_directory, l_A, l_B, pairs ) )
    {
            cerr << "Cannot write desc_bins.csv into " << param.pair_directory << endl;
            return false;
    }
    if( ! writeNearestFile( param.pair_directory / "desc_nn_a.csv",
                            l_A, l_B, nearest_A_B, pairs, true ) ||
        ! writeNearestFile( param.pair_directory / "desc_nn_b.csv",
                            l_B, l_A, nearest_B_A, pairs, false ) )
    {
            cerr << "Cannot write the descriptor nearest neighbours into " << param.pair_directory << endl;
            return false;
    }

    const auto at_least = [&]( Agreement level ) {
        return count_if( pairs.begin(), pairs.end(),
                         [level]( const KeypointPair& p ) { return p.level >= level; } );
    };
    // Pairs below level whose position has a pair that reaches it.
    const auto minor = [&]( Agreement level ) {
        return count_if( pairs.begin(), pairs.end(),
                         [level]( const KeypointPair& p ) {
                             return p.level < level && p.best_at_position >= level;
                         } );
    };
    cerr << "Co-located " << pairs.size() << " of " << l_A.size() << " keypoints in "
         << param.input[0] << " and " << l_B.size() << " in " << param.input[1] << endl
         << "  same orientation (<= " << pp.ori_tolerance_deg << " deg): "
         << at_least( Agreement::Orientation ) << endl
         << "  also similar descriptor (<= " << pp.desc_tolerance << "): "
         << at_least( Agreement::Descriptor ) << endl
         << "  orientation differs, but another keypoint at the same position agrees: "
         << minor( Agreement::Orientation ) << endl
         << "  descriptor differs, but another keypoint at the same position agrees: "
         << minor( Agreement::Descriptor ) << endl;

    return true;
}

int main( int argc, char* argv[] )
{
    Parameters param;

    parseargs( argc, argv, param );

    vector<feat_t> l_A;
    vector<feat_t> l_B;

    readFile( l_A, param.input[0] );
    readFile( l_B, param.input[1] );

    /* If both inputs are empty, report this as an error if not explicitly
     * permitted by a command line switch.
     * Always exit.
     */
    if( l_A.empty() && l_B.empty() )
    {
        if( !param.allow_empty )
        {
            cerr << "FAIL: neither input contains features "
                    "(use --allow-empty if that is expected)" << endl;
            return EXIT_FAILURE;
        }
        cerr << "Average best-match descriptor distance: 0" << endl;
        return EXIT_SUCCESS;
    }

    /* If only one input is empty, it is always an error.
     */
    if( l_A.empty() || l_B.empty() )
    {
        cerr << "FAIL: " << ( l_A.empty() ? param.input[0] : param.input[1] )
             << " contains no features, but "
             << ( l_A.empty() ? param.input[1] : param.input[0] ) << " does" << endl;
        return EXIT_FAILURE;
    }

    if( l_A.size() > std::numeric_limits<int>::max() ||
        l_B.size() > std::numeric_limits<int>::max() )
    {
        cerr << "FAIL: one of the files contains more than " << std::numeric_limits<int>::max()
             << "descriptors." << std::endl
             << "This tool does not support that and must be modified before continuing." << std::endl;
        return EXIT_FAILURE;
    }

    // The most expensive part: every descriptor of one first file against
    // every descriptor of the other, in parallel where available.
    vector<NearestDesc> nearest_A_B;
    vector<NearestDesc> nearest_B_A;
    nearestDescriptors( nearest_A_B, l_A, l_B );
    nearestDescriptors( nearest_B_A, l_B, l_A );

    bool written = writeFiles( param, nearest_A_B, nearest_B_A, l_A, l_B );
    if( !written )
        return EXIT_FAILURE;

    ostream* outfile  = openOutputFile( param.outfile_name );

    const float sum_min_dist = ( param.briefinfo )
                             ? distanceSumBrief( param, outfile?outfile:&cout, l_A, l_B, nearest_A_B )
                             : distanceSum( outfile?outfile:&cout, l_A, l_B );

    delete outfile;

    const float avg_min_dist = sum_min_dist / l_A.size();
    cerr << "Average best-match descriptor distance: " << avg_min_dist << endl;

    if( param.max_avg_desc_dist >= 0.0f && avg_min_dist > param.max_avg_desc_dist )
    {
        cerr << "FAIL: average best-match descriptor distance " << avg_min_dist
             << " exceeds threshold " << param.max_avg_desc_dist << endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

static float distanceSumBrief( const Parameters& param, ostream* outfile, const vector<feat_t>& l_A, const vector<feat_t>& l_B, const vector<NearestDesc>& nearest_A_B )
{
    if( l_B.size() == 0 ) return INFINITY;

    const int len = static_cast<int>( std::min<size_t>( l_A.size(), std::numeric_limits<int>::max() ) );

    ostream*      descfile = openOutputFile( param.descfile_name );
    vector<float> desc_stats( DescSize, 0.0f );

    ostream* print_distances = param.descfile_verbose ? descfile : nullptr;

    float sum_min_dist = 0.0f;

    for( int i = 0; i < len; ++i ) // for loops over signed integer types are faster
    {
        l_A[i].printBestMatch( *outfile, print_distances, l_B[nearest_A_B[i].index],
                               nearest_A_B[i].dist, desc_stats );
        *outfile << endl;
        sum_min_dist += nearest_A_B[i].dist;
    }

    if( descfile )
    {
        writeSummary( *descfile, desc_stats, l_A.size() );
        delete descfile;
    }

    return sum_min_dist;
}

static float distanceSum( ostream* outfile, const vector<feat_t>& l_A, const vector<feat_t>& l_B )
{
    auto sum_min_dist{0.0f};
    for( const auto& feature : l_A )
        sum_min_dist += feature.compareAll( *outfile, l_B );
    return sum_min_dist;
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

static ostream* openOutputFile( const string& outfile_name )
{
    if( outfile_name.empty() ) return nullptr;

    ostream* o = new ofstream( outfile_name );
    if( o->good() )
    {
        return o;
    }
    else
    {
        delete o;
    }

    return nullptr;
}

