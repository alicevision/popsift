#include <algorithm>
#include <cmath>
#include <cassert>
#include <fstream>
#include <filesystem>
#include <numeric>
#include <ostream>
#include <string_view>

#include "csf_match.h"

using namespace std;

namespace {

struct Candidate
{
    int   a;
    int   b;
    float pos_dist;
    float abs_ori;
    float ori_diff;
};

}

/** The name of an agreement level in the CSV files.
 */
static constexpr std::string_view agreementName( Agreement level )
{
    switch( level )
    {
    case Agreement::Position:    return "position";
    case Agreement::Orientation: return "orientation";
    case Agreement::Descriptor:  return "descriptor";
    }
    return "?";
}

vector<KeypointPair> matchKeypointsByPosition( const vector<feat_t>& l_a,
                                               const vector<feat_t>& l_b,
                                               const PairParams&     param )
{
    // Indices of l_b sorted by x, so that the candidates of a feature are
    // found by a range search instead of a scan over all of l_b.
    vector<int> by_x( l_b.size() );
    iota( by_x.begin(), by_x.end(), 0 );
    sort( by_x.begin(), by_x.end(),
          [&]( int i, int j ) { return l_b[i].x < l_b[j].x; } );

    vector<Candidate> cand;
    for( int ia = 0; ia < int(l_a.size()); ia++ )
    {
        const feat_t& fa = l_a[ia];
        const float   r  = max( param.radius_min, param.radius_sigma * fa.sigma );

        auto it = lower_bound( by_x.begin(), by_x.end(), fa.x - r,
                               [&]( int i, float v ) { return l_b[i].x < v; } );
        for( ; it != by_x.end() && l_b[*it].x <= fa.x + r; ++it )
        {
            const feat_t& fb = l_b[*it];
            const float   d = fa.l2PixDist( fb );
            if( d > r ) continue;

            if( fa.sigma > 0.0f && fb.sigma > 0.0f )
            {
                const float ratio = fb.sigma / fa.sigma;
                if( ratio > param.max_scale_ratio || ratio < 1.0f / param.max_scale_ratio ) continue;
            }

            const float od = fa.signedAngleDiffDegrees( fb );
            cand.push_back( Candidate{ ia, *it, d, fabsf(od), od } );
        }
    }

    // Position first. Features at the same position (several orientations
    // of one keypoint) are then paired by orientation.
    sort( cand.begin(), cand.end(),
          []( const Candidate& l, const Candidate& r ) {
              if( l.pos_dist != r.pos_dist ) return l.pos_dist < r.pos_dist;
              if( l.abs_ori  != r.abs_ori  ) return l.abs_ori  < r.abs_ori;
              if( l.a != r.a ) return l.a < r.a;
              return l.b < r.b;
          } );

    vector<bool>      used_a( l_a.size(), false );
    vector<bool>      used_b( l_b.size(), false );
    vector<KeypointPair> pairs;
    for( const Candidate& c : cand )
    {
        if( used_a[c.a] || used_b[c.b] ) continue;
        used_a[c.a] = true;
        used_b[c.b] = true;
        pairs.push_back( KeypointPair{ c.a, c.b, c.pos_dist, c.ori_diff,
                                       NAN, Agreement::Position } );
    }

    sort( pairs.begin(), pairs.end(),
          []( const KeypointPair& l, const KeypointPair& r ) { return l.a < r.a; } );
    return pairs;
}

void classifyAgreement( vector<KeypointPair>& pairs,
                        const vector<feat_t>& l_a,
                        const vector<feat_t>& l_b,
                        const PairParams&     param )
{
    for( KeypointPair& p : pairs )
    {
        p.desc_dist = descDistance( l_a[p.a], l_b[p.b] );
        p.level     = Agreement::Position;
        if( fabsf( p.ori_diff ) <= param.ori_tolerance_deg )
        {
            p.level = ( p.desc_dist <= param.desc_tolerance ) ? Agreement::Descriptor
                                                              : Agreement::Orientation;
        }
    }
}

void nearestDescriptors( std::vector<NearestDesc>& nearest,
                         const std::vector<feat_t>& l_query,
                         const std::vector<feat_t>& l_other )
{

    nearest.clear();
    nearest.resize( l_query.size() );

    // comparison should never reach this if one of the lists is empty
    assert( !l_query.empty() );
    assert( !l_other.empty() );

    const int   otherlen = static_cast<int>( l_other.size() );
    vector<int> idx( l_query.size() );
    iota( idx.begin(), idx.end(), 0 );
    for_each( PAR_UNSEQ idx.begin(), idx.end(), [&]( int iq ) {
        // Sums instead of distances, so that a candidate can be dropped as
        // soon as it is farther than the second nearest so far.
        size_t index  = 0;
        float  best   = INFINITY;
        float  second = INFINITY;
        for( int io = 0; io < otherlen; ++io )
        {
            const float s = descDistanceSum( l_query[iq], l_other[io], second );
            if( s < best )
            {
                second = best;
                best   = s;
                index  = static_cast<size_t>( io );
            }
            else if( s < second )
            {
                second = s;
            }
        }
        assert( best != INFINITY );
        nearest[iq] = NearestDesc{ index,
                                   descDistanceFromSum( best ),
                                   descDistanceFromSum( second ) };
    } );
}

static void writeFeature( std::ostream& f, int index, const feat_t& ft )
{
    f << csf::format( "{},{:.6g},{:.6g},{:.6g},{:.6g}", index, ft.x, ft.y, ft.sigma, ft.ori * RAD2DEG );
}

static bool writeUnpaired( const std::filesystem::path& path, const vector<feat_t>& l, const vector<bool>& used )
{
    std::ofstream f( path );
    if( !f ) return false;

    f << "index,x,y,sigma,ori_deg\n";
    for( int i = 0; i < int(l.size()); i++ )
    {
        if( used[i] ) continue;
        writeFeature( f, i, l[i] );
        f << "\n";
    }
    return !f.fail();
}

bool writeNearestFile( const std::filesystem::path& path,
                       const vector<feat_t>&        l_query,
                       const vector<feat_t>&        l_other,
                       const vector<NearestDesc>&   nearest,
                       const vector<KeypointPair>&  pairs,
                       bool                         pairs_query_first )
{
    vector<int> colocated( l_query.size(), -1 );
    for( const KeypointPair& p : pairs )
    {
        if( pairs_query_first ) colocated[p.a] = p.b;
        else                    colocated[p.b] = p.a;
    }

    std::ofstream f( path );
    if( !f ) return false;

    f << "q_index,q_x,q_y,q_sigma,q_ori_deg,"
         "n_index,n_x,n_y,n_sigma,n_ori_deg,"
         "desc_dist,second_desc_dist,pos_dist,ori_diff_deg,colocated_index\n";
    for( int iq = 0; iq < int(l_query.size()); iq++ )
    {
        const NearestDesc& n = nearest[iq];
        const feat_t& fq = l_query[iq];
        const feat_t& fn = l_other[n.index];
        writeFeature( f, iq, fq );
        f << ",";
        writeFeature( f, n.index, fn );
        f << csf::format( ",{:.6g},{:.6g},{:.6g},{:.6g},{}\n",
                          n.dist, n.second,
                          fq.l2PixDist(fn), // sqrtf( (fn.x-fq.x)*(fn.x-fq.x) + (fn.y-fq.y)*(fn.y-fq.y) ),
                          fq.signedAngleDiffDegrees(fn),
                          colocated[iq] );
    }
    return !f.fail();
}

bool writePairFiles( const std::filesystem::path& base,
                     const vector<feat_t>&        l_a,
                     const vector<feat_t>&        l_b,
                     const vector<KeypointPair>&  pairs )
{
    std::ofstream f( base / "colocated.csv" );
    if( !f ) return false;

    f << "a_index,a_x,a_y,a_sigma,a_ori_deg,"
         "b_index,b_x,b_y,b_sigma,b_ori_deg,"
         "dx,dy,pos_dist,scale_ratio,ori_diff_deg,desc_dist,agreement\n";

    vector<bool> used_a( l_a.size(), false );
    vector<bool> used_b( l_b.size(), false );
    for( const KeypointPair& p : pairs )
    {
        const feat_t& fa = l_a[p.a];
        const feat_t& fb = l_b[p.b];
        used_a[p.a] = true;
        used_b[p.b] = true;
        writeFeature( f, p.a, fa );
        f << ",";
        writeFeature( f, p.b, fb );
        f << csf::format( ",{:.6g},{:.6g},{:.6g},{:.6g},{:.6g},{:.6g},{}\n",
                          fb.x - fa.x, fb.y - fa.y, p.pos_dist,
                          fa.sigma > 0.0f ? fb.sigma / fa.sigma : 0.0f,
                          p.ori_diff,
                          p.desc_dist,
                          agreementName( p.level ) ); // data() is a hack
    }
    if( f.fail() ) return false;

    return writeUnpaired( base / "only_a.csv", l_a, used_a )
        && writeUnpaired( base / "only_b.csv", l_b, used_b );
}

bool writeDescBinsFile( const std::filesystem::path& base,
                        const vector<feat_t>&        l_a,
                        const vector<feat_t>&        l_b,
                        const vector<KeypointPair>&  pairs )
 {
    // [subset][element]: subset 0 is all co-located pairs, 1 those with the
    // same orientation
    long   count[2]             = { 0, 0 };
    double sum[2][DescSize]     = {};
    double sum_abs[2][DescSize] = {};
    for( const KeypointPair& p : pairs )
    {
        const bool same = p.level != Agreement::Position;
        for( int s = 0; s < ( same ? 2 : 1 ); s++ )
        {
            count[s]++;
            for( int i = 0; i < DescSize; i++ )
            {
                const double d = l_b[p.b].desc[i] - l_a[p.a].desc[i];
                sum[s][i]     += d;
                sum_abs[s][i] += fabs( d );
            }
        }
    }

    std::ofstream f( base / "desc_bins.csv" );
    if( !f ) return false;

    f << "subset,element,pairs,sum_diff,sum_abs_diff\n";
    constexpr std::string_view names[2] = { "colocated", "same_orientation" };

    for( int s = 0; s < 2; s++ )
    {
        for( int i = 0; i < DescSize; i++ )
        {
            f << csf::format( "{},{},{},{:.9g},{:.9g}\n", names[s], i, count[s], sum[s][i], sum_abs[s][i] );
        }
    }
    return !f.fail();
}

