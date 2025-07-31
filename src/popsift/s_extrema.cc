/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/clamp.h"
#include "common/debug_macros.h"
#include "common/grid.h"
#include "s_solve.h"
#include "sift_constants.h"
#include "sift_pyramid.h"

#include <cstdio>
#include <cmath>
#include <numeric>
#include <sstream>
#include <iterator>

namespace popsift{

static
inline void extremum_cmp( float val, float f, uint32_t& gt, uint32_t& lt, uint32_t mask )
{
    gt |= ( ( val > f ) ? mask : 0 );
    lt |= ( ( val < f ) ? mask : 0 );
}

#define TX(dx,dy,dz) obj.get( z+dz, y+dy, x+dx )

static
inline bool is_extremum( Plane2D_float& obj,
                         int x, int y, int z )
{
    uint32_t gt = 0;
    uint32_t lt = 0;

    const float val0 = TX( 0, 1, 1 );
    const float val2 = TX( 2, 1, 1 );
    const float val  = TX( 1, 1, 1 );

    // bit indeces for neighbours:
    //     7 0 1    0x80 0x01 0x02
    //     6   2 -> 0x40      0x04
    //     5 4 3    0x20 0x10 0x08
    // upper layer << 24 ; own layer << 16 ; lower layer << 8
    // 1st group: left and right neigbhour
    extremum_cmp( val, val0, gt, lt, 0x00400000 ); // ( 0x01<<6 ) << 16
    extremum_cmp( val, val2, gt, lt, 0x00040000 ); // ( 0x01<<2 ) << 16

    if( ( gt != 0x00440000 ) && ( lt != 0x00440000 ) ) return false;

    // 2nd group: requires a total of 8 128-byte reads
    extremum_cmp( val, TX(0,0,1), gt, lt, 0x00800000 ); // ( 0x01<<7 ) << 16
    extremum_cmp( val, TX(0,2,1), gt, lt, 0x00200000 ); // ( 0x01<<5 ) << 16
    extremum_cmp( val, TX(0,0,0), gt, lt, 0x80000000 ); // ( 0x01<<6 ) << 24
    extremum_cmp( val, TX(0,2,0), gt, lt, 0x40000000 ); // ( 0x01<<6 ) << 24
    extremum_cmp( val, TX(0,1,0), gt, lt, 0x20000000 ); // ( 0x01<<6 ) << 24
    extremum_cmp( val, TX(0,0,2), gt, lt, 0x00008000 ); // ( 0x01<<6 ) <<  8
    extremum_cmp( val, TX(0,1,2), gt, lt, 0x00004000 ); // ( 0x01<<6 ) <<  8
    extremum_cmp( val, TX(0,2,2), gt, lt, 0x00002000 ); // ( 0x01<<6 ) <<  8

    if( ( gt != 0xe0e4e000 ) && ( lt != 0xe0e4e000 ) ) return false;

    // 3rd group: remaining 2 cache misses in own layer
    extremum_cmp( val, TX(1,0,1), gt, lt, 0x00010000 ); // ( 0x01<<0 ) << 16
    extremum_cmp( val, TX(2,0,1), gt, lt, 0x00020000 ); // ( 0x01<<1 ) << 16
    extremum_cmp( val, TX(1,2,1), gt, lt, 0x00100000 ); // ( 0x01<<4 ) << 16
    extremum_cmp( val, TX(2,2,1), gt, lt, 0x00080000 ); // ( 0x01<<3 ) << 16

    if( ( gt != 0xe0ffe000 ) && ( lt != 0xe0ffe000 ) ) return false;

    // 4th group: 3 cache misses higher layer
    extremum_cmp( val, TX(1,0,0), gt, lt, 0x01000000 ); // ( 0x01<<0 ) << 24
    extremum_cmp( val, TX(2,0,0), gt, lt, 0x02000000 ); // ( 0x01<<1 ) << 24
    extremum_cmp( val, TX(1,1,0), gt, lt, 0x00000004 ); // ( 0x01<<2 )
    extremum_cmp( val, TX(2,1,0), gt, lt, 0x04000000 ); // ( 0x01<<2 ) << 24
    extremum_cmp( val, TX(1,2,0), gt, lt, 0x10000000 ); // ( 0x01<<4 ) << 24
    extremum_cmp( val, TX(2,2,0), gt, lt, 0x08000000 ); // ( 0x01<<3 ) << 24

    if( ( gt != 0xffffe004 ) && ( lt != 0xffffe004 ) ) return false;

    // 5th group: 3 cache misss lower layer
    extremum_cmp( val, TX(1,0,2), gt, lt, 0x00000100 ); // ( 0x01<<0 ) <<  8
    extremum_cmp( val, TX(2,0,2), gt, lt, 0x00000200 ); // ( 0x01<<1 ) <<  8
    extremum_cmp( val, TX(1,1,2), gt, lt, 0x00000001 ); // ( 0x01<<0 )
    extremum_cmp( val, TX(2,1,2), gt, lt, 0x00000400 ); // ( 0x01<<2 ) <<  8
    extremum_cmp( val, TX(1,2,2), gt, lt, 0x00001000 ); // ( 0x01<<4 ) <<  8
    extremum_cmp( val, TX(2,2,2), gt, lt, 0x00000800 ); // ( 0x01<<3 ) <<  8

    if( ( gt != 0xffffff05 ) && ( lt != 0xffffff05 ) ) return false;

    return true;
}

template<int sift_mode>
class ModeFunctions
{
public:
    /* refine
     * returns 0 : continue looping
     *         1 : break loop and succeed
     */
    inline 
    int refine( float3& d, int3& n, int width, int height, int maxlevel, bool last_it );
};

template<>
class ModeFunctions<Config::RefineInLevel>
{
public:
    inline 
    int refine( float3& d, int3& n, int width, int height, int maxlevel, bool last_it ) const
    {
        if( last_it ) return 0;

        int2 t;

        t.x = ((d.x >=  0.6f && n.x < width-2) ?  1 : 0 )
            + ((d.x <= -0.6f && n.x > 1)       ? -1 : 0 );

        t.y = ((d.y >=  0.6f && n.y < height-2)  ?  1 : 0 )
            + ((d.y <= -0.6f && n.y > 1)         ? -1 : 0 );

        if( t.x == 0 && t.y == 0 ) {
            // no more changes
            return 1;
        }

        n.x += t.x;
        n.y += t.y;
        // n.z += t.z; - VLFeat is not changing levels !!!

        return 0;
    }
};

template<>
class ModeFunctions<Config::RefineInOctave>
{
public:
    inline 
    int refine( float3& d, int3& n, int width, int height, int maxlevel, bool last_it ) const
    {
        if( last_it ) return 0;

        int3 t;

        t.x = ((d.x >=  0.6f && n.x < width-2) ?  1 : 0 )
            + ((d.x <= -0.6f && n.x > 1)       ? -1 : 0 );

        t.y = ((d.y >=  0.6f && n.y < height-2)  ?  1 : 0 )
            + ((d.y <= -0.6f && n.y > 1)         ? -1 : 0 );

        t.z = ((d.z >=  0.6f && n.z < maxlevel-1)  ?  1 : 0 )
            + ((d.z <= -0.6f && n.z > 1)           ? -1 : 0 );

        if( t.x == 0 && t.y == 0 && t.z == 0 ) {
            // no more changes
            return 1;
        }

        n.x += t.x;
        n.y += t.y;
        n.z += t.z;

        return 0;
    }
};

inline static
bool first_contrast_ok( const float val )
{
    return ( fabsf( val ) >= 1.6f * h_consts.threshold );
}

/** verify() checks whether a refine position is outside the image boundaries or
 *  outside the DoG boundaries.
 *  returns true  : values after refine make sense
 *          false : they do not
 */
inline static
bool verify( float xn, float yn, float sn, int width, int height, int maxlevel )
{
    // reject if outside of image bounds or far outside DoG bounds
    return ( ( xn < 0.0f ||
               xn > width - 1.0f ||
               yn < 0.0f ||
               yn > height - 1.0f ||
               sn < -0.0f ||
               sn > maxlevel ) ? false
                               : true );
}

template<int sift_mode>
static inline
bool find_extrema_in_dog_sub( const int3&      g,
                              Plane2D_float&   dog,
                              int              this_octave,
                              int              width,
                              int              height,
                              uint32_t         maxlevel,
                              float            w_grid_divider,
                              float            h_grid_divider,
                              int              grid_width,
                              InitialExtremum& ec)
{
    const bool no_extrema_reporting = true;

    ec.xpos    = 0.0f;
    ec.ypos    = 0.0f;
    ec.lpos    = 0;
    ec.sigma   = 0.0f;

    /*
     * First consideration: extrema cannot be found on any outermost edge,
     * one pixel on the left, right, upper, lower edge will never qualify.
     * Also, the upper and lower DoG layer will never qualify. So there is
     * no reason for selecting any of those pixel for the center of a 3x3x3
     * region.
     * Instead, I use groups of 32x4 threads that read from a 34x34x3 area,
     * but implicitly, they fetch * 64x4+2x3 floats (bad luck).
     * To find maxima, compare first on the left edge of the 3x3x3 cube, ie.
     * a 1x3x3 area. If the rightmost 2 threads of a warp (x==30 and 3==31)
     * are not extreme w.r.t. to the left slice, 8 fetch operations.
     */
    // const int block_x = g.blockIdx.x * 32;
    // const int block_y = g.blockIdx.y * g.blockDim.y;
    // const int block_z = g.blockIdx.z;
    // const int y       = block_y + g.threadIdx.y + 1;
    // const int x       = block_x + g.threadIdx.x + 1;
    // const int level   = block_z + 1;
    const int x     = g.x + 1;
    const int y     = g.y + 1;
    const int level = g.z + 1;

    const float val = dog.get( level, y, x );

    ModeFunctions<sift_mode> f;
    if( ! first_contrast_ok( val ) ) return false;

    if( ! is_extremum( dog, x-1, y-1, level-1 ) ) {
        // if( this_octave==0 && level==2 && x==14 && y==73 ) printf("But I fail\n");
        return false;
    } else {
        POP_INFO2( no_extrema_reporting, "Found an extremum in octave " << this_octave << " at (" << x << ", " << y << ", " << level << ")" );
    }

    float3 D; // Dx Dy Ds
    float3 DD; // Dxx Dyy Dss
    float3 DX; // Dxy Dxs Dys
    float3 d; // dx dy ds

    float v = val;

    int3 n = make_int3( x, y, level ); // nj ni ns

    int32_t iter = 0;

#define MAX_ITERATIONS 5

    do {
        iter++;

        // const int z = level - 1;
        /* compute gradient */
        const float x2y1z1 = dog.get( n.z,   n.y  , n.x+1 );
        const float x0y1z1 = dog.get( n.z,   n.y  , n.x-1 );
        const float x1y2z1 = dog.get( n.z,   n.y+1, n.x   );
        const float x1y0z1 = dog.get( n.z,   n.y-1, n.x   );
        const float x1y1z2 = dog.get( n.z+1, n.y  , n.x   );
        const float x1y1z0 = dog.get( n.z-1, n.y  , n.x   );
        // D.x = 0.5f * ( x2y1z1 - x0y1z1 );
        // D.y = 0.5f * ( x1y2z1 - x1y0z1 );
        // D.z = 0.5f * ( x1y1z2 - x1y1z0 );
        D.x = scalbnf( x2y1z1 - x0y1z1, -1 );
        D.y = scalbnf( x1y2z1 - x1y0z1, -1 );
        D.z = scalbnf( x1y1z2 - x1y1z0, -1 );

        /* compute Hessian */
        const float x1y1z1 = dog.get( n.z, n.y, n.x );
        // DD.x = x2y1z1 + x0y1z1 - 2.0f * x1y1z1;
        // DD.y = x1y2z1 + x1y0z1 - 2.0f * x1y1z1;
        // DD.z = x1y1z2 + x1y1z0 - 2.0f * x1y1z1;
        DD.x = x2y1z1 + x0y1z1 - scalbnf( x1y1z1, 1 );
        DD.y = x1y2z1 + x1y0z1 - scalbnf( x1y1z1, 1 );
        DD.z = x1y1z2 + x1y1z0 - scalbnf( x1y1z1, 1 );

        const float x0y0z1 = dog.get( n.z  , n.y-1, n.x-1 );
        const float x0y1z0 = dog.get( n.z-1, n.y  , n.x-1 );
        const float x0y1z2 = dog.get( n.z+1, n.y  , n.x-1 );
        const float x0y2z1 = dog.get( n.z  , n.y+1, n.x-1 );
        const float x1y0z0 = dog.get( n.z-1, n.y-1, n.x   );
        const float x1y0z2 = dog.get( n.z+1, n.y-1, n.x   );
        const float x1y2z0 = dog.get( n.z-1, n.y+1, n.x   );
        const float x1y2z2 = dog.get( n.z+1, n.y+1, n.x   );
        const float x2y0z1 = dog.get( n.z  , n.y-1, n.x+1 );
        const float x2y1z0 = dog.get( n.z-1, n.y  , n.x+1 );
        const float x2y1z2 = dog.get( n.z+1, n.y  , n.x+1 );
        const float x2y2z1 = dog.get( n.z  , n.y+1, n.x+1 );
        // DX.x = 0.25f * ( x2y2z1 + x0y0z1 - x0y2z1 - x2y0z1 );
        // DX.y = 0.25f * ( x2y1z2 + x0y1z0 - x0y1z2 - x2y1z0 );
        // DX.z = 0.25f * ( x1y2z2 + x1y0z0 - x1y2z0 - x1y0z2 );
        DX.x = scalbnf( x2y2z1 + x0y0z1 - x0y2z1 - x2y0z1, -2 );
        DX.y = scalbnf( x2y1z2 + x0y1z0 - x0y1z2 - x2y1z0, -2 );
        DX.z = scalbnf( x1y2z2 + x1y0z0 - x1y2z0 - x1y0z2, -2 );

        float3 b;
        float A[3][3];

        /* Solve linear system. */
        A[0][0] = DD.x;
        A[1][1] = DD.y;
        A[2][2] = DD.z;
        A[1][0] = A[0][1] = DX.x;
        A[2][0] = A[0][2] = DX.y;
        A[2][1] = A[1][2] = DX.z;

        b.x = -D.x;
        b.y = -D.y;
        b.z = -D.z;

        if(!solve(A, b)) {
            d.x = 0;
            d.y = 0;
            d.z = 0;
            break ;
        }

        d = b;

        /* If the translation of the keypoint is big, move the keypoint
         * and re-iterate the computation. Otherwise we are all set.
         */
        const int retval = f.refine( d, n, width, height, maxlevel, iter==MAX_ITERATIONS );

        if( retval == 1 ) {
            break;
        }
    }
    while( iter < MAX_ITERATIONS ); /* go to next iter */

    if( d.x >= 1.5f || d.y >= 1.5f || d.z >= 1.5f ) {
        // excessive pixel movement in at least dimension, reject
        POP_INFO2( no_extrema_reporting, "Failed due to excessive repositioning" );
        return false;
    }

    const float xn      = n.x + d.x;
    const float yn      = n.y + d.y;
    const float sn      = n.z + d.z;

    if( ! verify( xn, yn, sn, width, height, maxlevel ) ) {
        POP_INFO2( no_extrema_reporting, "Failed due to optimum outside image plane" );
        return false;
    }

    // float contr   = v + 0.5f * (D.x * d.x + D.y * d.y + D.z * d.z);
    const float contr   = v + scalbnf( D.x * d.x + D.y * d.y + D.z * d.z , -1 );
    const float tr      = DD.x + DD.y;
    const float det     = DD.x * DD.y - DX.x * DX.x;
    const float edgeval = tr * tr / det;

    /* negative determinant => curvatures have different signs -> reject it */
    if (det <= 0.0f) {
        POP_INFO2( no_extrema_reporting, "Failed due to saddle-shaped optimum" );
        return false;
    }

    /* accept-reject extremum */
    // if( fabsf(contr) < (h_consts.threshold*2.0f) )
    if( fabsf(contr) < scalbnf( h_consts.threshold, 1 ) )
    {
        POP_INFO2( no_extrema_reporting, "Failed because contrast threshold exceeded" );
        return false;
    }

    /* reject condition: tr(H)^2/det(H) < (r+1)^2/r */
    if( edgeval >= (h_consts.edge_limit+1.0f)*(h_consts.edge_limit+1.0f)/h_consts.edge_limit ) {
        POP_INFO2( no_extrema_reporting, "Failed because edge threshold exceeded" );
        return false;
    }

    ec.xpos      = xn;
    ec.ypos      = yn;
    ec.lpos      = (int)roundf(sn);
    ec.sigma     = h_consts.sigma0 * pow(h_consts.sigma_k, sn); // * 2;
    ec.cell      = floorf( yn / h_grid_divider ) * grid_width + floorf( xn / w_grid_divider );
        // const float sigma_k = powf(2.0f, 1.0f / levels );

    POP_INFO2( no_extrema_reporting, "Succeeded" );
    return true;
}


template<int sift_mode>
static
void find_extrema_in_dog( const int3&    g,
                          Plane2D_float& dog,
                          int            octave,
                          int            width,
                          int            height,
                          const uint32_t maxlevel,
                          const float    w_grid_divider,
                          const float    h_grid_divider,
                          const int      grid_width )
{
    const bool no_extrema_reporting = false;

    std::vector<InitialExtremum>& i_extrema = dct.initial_extrema_in_octave[octave];

    POP_INFO2( no_extrema_reporting, "initial extrema values for octave " << octave );
    for( int z=0; z<g.z; z++ )
    {
        for( int y=0; y<g.y; y++ )
        {
            for( int x=0; x<g.x; x++ )
            {
                InitialExtremum ec;
                ec.ignore = false;

                int3 gi( x, y, z );

                bool indicator = find_extrema_in_dog_sub<sift_mode>( gi,
                                                                     dog,
                                                                     octave,
                                                                     width,
                                                                     height,
                                                                     maxlevel,
                                                                     w_grid_divider,
                                                                     h_grid_divider,
                                                                     grid_width,
                                                                     ec );

                if( indicator )
                {
                    // store the initial extremum in an array
                    i_extrema.emplace_back( ec );
                }
            }
        }

        std::vector<int>& i_ext_off = dct.initial_extrema_offset[octave];

        i_ext_off.resize( i_extrema.size() );

        for( int w_idx=0; w_idx<i_extrema.size(); w_idx++ )
        {
            i_extrema[w_idx].write_index = w_idx;
            i_ext_off[w_idx]             = w_idx;
        }


        POP_INFO2( no_extrema_reporting, "Number of extrema in octave " << octave << " after level " << z << ": " << i_extrema.size() );
    }

    dct.extrema_count_per_octave[octave] = i_extrema.size();

    POP_INFO2( no_extrema_reporting, "final extrema count in octave " << octave << ": " << dct.extrema_count_per_octave[octave] );
}

void Pyramid::find_extrema( const Config& conf )
{
    POP_INFO2( false, "Enter " << __FUNCTION__ );

    dct.extrema_count_per_octave.resize( MAX_OCTAVES );

    for( int octave=0; octave<_num_octaves; octave++ )
    {
        Octave&      oct_obj = _octaves[octave];

        int*  extrema_num_blocks = getNumberOfBlocks( octave );

        int cols = oct_obj.getWidth();
        int rows = oct_obj.getHeight();

        int3 g( cols, rows, _levels-3 );
        // Grid g;
        // g.setBlockDim( 32, 4 );
        // g.setGridDim( grid_divide( cols, g.blockDim.x ), grid_divide( rows, g.blockDim.y ), _levels - 3 );

        int*  num_blocks      = extrema_num_blocks;

        switch( conf.getSiftMode() )
        {
        case Config::RefineInLevel :
                find_extrema_in_dog<Config::RefineInLevel>
                    ( g,
                      oct_obj.getDog( ),
                      octave,
                      cols,
                      rows,
                      _levels-1,
                      oct_obj.getWGridDivider(),
                      oct_obj.getHGridDivider(),
                      conf.getFilterGridSize() );
                break;
        default :
                find_extrema_in_dog<Config::RefineInOctave>
                    ( g,
                      oct_obj.getDog( ),
                      octave,
                      cols,
                      rows,
                      _levels-1,
                      oct_obj.getWGridDivider(),
                      oct_obj.getHGridDivider(),
                      conf.getFilterGridSize() );
                break;
        }
    }

    POP_INFO2( false, "found extrema in all octaves" );

    /* Copy the extreme count for every octave from the (already initialized)
     * array extrema_count_per_octave to the (uninitialized) array extrema_count_prefix_sum. */
    dct.extrema_count_prefix_sum.resize( dct.extrema_count_per_octave.size() + 1 );

    auto it = dct.extrema_count_prefix_sum.begin();
    *it = 0;
    it++;

    /* Compute the exclusive prefix sum on the array extrema_count_prefix_sum, but add
     * the total sum in the last element. Easier to achieve with an inclusive_scan. */
    std::inclusive_scan( dct.extrema_count_per_octave.begin(),
                         dct.extrema_count_per_octave.end(),
                         it );

    /* Store the total number of orientations and the total number of
     * extrema as well. */
    dct.extrema_count_total = dct.extrema_count_prefix_sum.back();

    std::ostringstream debug_ostr;
    debug_ostr << "Extrema per octave:" << std::endl;
    std::copy( dct.extrema_count_per_octave.begin(),
               dct.extrema_count_per_octave.end(),
               std::ostream_iterator<int>(debug_ostr, " ") );
    debug_ostr << std::endl
          << "Extrema prefix sum per octave:" << std::endl;
    std::copy( dct.extrema_count_prefix_sum.begin(),
               dct.extrema_count_prefix_sum.end(),
               std::ostream_iterator<int>(debug_ostr, " ") );
    debug_ostr << std::endl
          << "Extrema prefix sum per octave: "
          << dct.extrema_count_total
          << std::endl;
    POP_INFO2( false, debug_ostr.str() );
}

} // namespace popsift

