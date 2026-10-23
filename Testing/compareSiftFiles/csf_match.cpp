#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>

#include "csf_match.h"

using namespace std;

static const float RAD2DEG = 180.0f / 3.14159265358979323846f;

float signedAngleDiffDegrees( float ori_a, float ori_b )
{
    float d = fmodf( ( ori_b - ori_a ) * RAD2DEG, 360.0f );
    if( d <  -180.0f ) d += 360.0f;
    if( d >=  180.0f ) d -= 360.0f;
    return d;
}

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

vector<MatchPair> matchFeatures( const vector<feat_t>& l_a,
                                 const vector<feat_t>& l_b,
                                 const MatchParams&    param )
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
            const float   dx = fb.x - fa.x;
            const float   dy = fb.y - fa.y;
            const float   d  = sqrtf( dx*dx + dy*dy );
            if( d > r ) continue;

            if( fa.sigma > 0.0f && fb.sigma > 0.0f )
            {
                const float ratio = fb.sigma / fa.sigma;
                if( ratio > param.max_scale_ratio || ratio < 1.0f / param.max_scale_ratio ) continue;
            }

            const float od = signedAngleDiffDegrees( fa.ori, fb.ori );
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
    vector<MatchPair> pairs;
    for( const Candidate& c : cand )
    {
        if( used_a[c.a] || used_b[c.b] ) continue;
        used_a[c.a] = true;
        used_b[c.b] = true;
        pairs.push_back( MatchPair{ c.a, c.b, c.pos_dist, c.ori_diff } );
    }

    sort( pairs.begin(), pairs.end(),
          []( const MatchPair& l, const MatchPair& r ) { return l.a < r.a; } );
    return pairs;
}

vector<NearestDesc> nearestDescriptors( const vector<feat_t>& l_query,
                                        const vector<feat_t>& l_other )
{
    vector<NearestDesc> nearest( l_query.size() );
    vector<int>         idx( l_query.size() );
    iota( idx.begin(), idx.end(), 0 );
    for_each( PAR_UNSEQ idx.begin(), idx.end(), [&]( int iq ) {
        // Sums instead of distances, so that a candidate can be dropped as
        // soon as it is farther than the second nearest so far.
        float best   = INFINITY;
        float second = INFINITY;
        int   index  = -1;
        for( int io = 0; io < int(l_other.size()); io++ )
        {
            const float s = descDistanceSum( l_query[iq], l_other[io], second );
            if( s < best )
            {
                second = best;
                best   = s;
                index  = io;
            }
            else if( s < second )
            {
                second = s;
            }
        }
        nearest[iq] = NearestDesc{ index, descDistanceFromSum( best ),
                                   descDistanceFromSum( second ) };
    } );
    return nearest;
}

static void writeFeature( FILE* f, int index, const feat_t& ft )
{
    fprintf( f, "%d,%.6g,%.6g,%.6g,%.6g", index, ft.x, ft.y, ft.sigma, ft.ori * RAD2DEG );
}

static bool writeUnmatched( const string& path, const vector<feat_t>& l, const vector<bool>& used )
{
    FILE* f = fopen( path.c_str(), "w" );
    if( !f ) return false;
    fprintf( f, "index,x,y,sigma,ori_deg\n" );
    for( int i = 0; i < int(l.size()); i++ )
    {
        if( used[i] ) continue;
        writeFeature( f, i, l[i] );
        fprintf( f, "\n" );
    }
    return fclose( f ) == 0;
}

bool writeNearestFile( const string&              path,
                       const vector<feat_t>&      l_query,
                       const vector<feat_t>&      l_other,
                       const vector<NearestDesc>& nearest,
                       const vector<MatchPair>&   pairs,
                       bool                       pairs_query_first )
{
    vector<int> geo( l_query.size(), -1 );
    for( const MatchPair& p : pairs )
    {
        if( pairs_query_first ) geo[p.a] = p.b;
        else                    geo[p.b] = p.a;
    }

    FILE* f = fopen( path.c_str(), "w" );
    if( !f ) return false;
    fprintf( f, "q_index,q_x,q_y,q_sigma,q_ori_deg,"
                "n_index,n_x,n_y,n_sigma,n_ori_deg,"
                "desc_dist,second_desc_dist,pos_dist,ori_diff_deg,geo_index\n" );
    for( int iq = 0; iq < int(l_query.size()); iq++ )
    {
        const NearestDesc& n = nearest[iq];
        if( n.index < 0 ) continue;
        const feat_t& fq = l_query[iq];
        const feat_t& fn = l_other[n.index];
        writeFeature( f, iq, fq );
        fprintf( f, "," );
        writeFeature( f, n.index, fn );
        fprintf( f, ",%.6g,%.6g,%.6g,%.6g,%d\n",
                 n.dist, n.second,
                 sqrtf( (fn.x-fq.x)*(fn.x-fq.x) + (fn.y-fq.y)*(fn.y-fq.y) ),
                 signedAngleDiffDegrees( fq.ori, fn.ori ),
                 geo[iq] );
    }
    return fclose( f ) == 0;
}

bool writeMatchFiles( const string&            dir,
                      const vector<feat_t>&    l_a,
                      const vector<feat_t>&    l_b,
                      const vector<MatchPair>& pairs )
{
    const string base = dir.empty() ? string() : dir + "/";

    FILE* f = fopen( ( base + "matched.csv" ).c_str(), "w" );
    if( !f ) return false;
    fprintf( f, "a_index,a_x,a_y,a_sigma,a_ori_deg,"
                "b_index,b_x,b_y,b_sigma,b_ori_deg,"
                "dx,dy,pos_dist,scale_ratio,ori_diff_deg,desc_dist\n" );

    vector<bool> used_a( l_a.size(), false );
    vector<bool> used_b( l_b.size(), false );
    for( const MatchPair& p : pairs )
    {
        const feat_t& fa = l_a[p.a];
        const feat_t& fb = l_b[p.b];
        used_a[p.a] = true;
        used_b[p.b] = true;
        writeFeature( f, p.a, fa );
        fprintf( f, "," );
        writeFeature( f, p.b, fb );
        fprintf( f, ",%.6g,%.6g,%.6g,%.6g,%.6g,%.6g\n",
                 fb.x - fa.x, fb.y - fa.y, p.pos_dist,
                 fa.sigma > 0.0f ? fb.sigma / fa.sigma : 0.0f,
                 p.ori_diff,
                 descDistance( fa, fb ) );
    }
    if( fclose( f ) != 0 ) return false;

    return writeUnmatched( base + "only_a.csv", l_a, used_a )
        && writeUnmatched( base + "only_b.csv", l_b, used_b );
}

bool writeDescBinsFile( const string&            dir,
                        const vector<feat_t>&    l_a,
                        const vector<feat_t>&    l_b,
                        const vector<MatchPair>& pairs,
                        float                    same_pose_px,
                        float                    same_pose_deg )
{
    // [subset][element]: subset 0 is all matched pairs, 1 the same pose
    long   count[2]             = { 0, 0 };
    double sum[2][DescSize]     = {};
    double sum_abs[2][DescSize] = {};
    for( const MatchPair& p : pairs )
    {
        const bool same = p.pos_dist <= same_pose_px && fabsf( p.ori_diff ) <= same_pose_deg;
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

    const string base = dir.empty() ? string() : dir + "/";
    FILE* f = fopen( ( base + "desc_bins.csv" ).c_str(), "w" );
    if( !f ) return false;
    fprintf( f, "subset,element,pairs,sum_diff,sum_abs_diff\n" );
    const char* names[2] = { "matched", "same_pose" };
    for( int s = 0; s < 2; s++ )
    {
        for( int i = 0; i < DescSize; i++ )
        {
            fprintf( f, "%s,%d,%ld,%.9g,%.9g\n", names[s], i, count[s], sum[s][i], sum_abs[s][i] );
        }
    }
    return fclose( f ) == 0;
}
