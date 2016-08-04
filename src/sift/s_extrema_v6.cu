#include "sift_pyramid.h"
#include "sift_constants.h"
#include "s_sigma.h"
#include "s_solve.h"
#include "debug_macros.h"
#include "assist.h"
#include "clamp.h"
#include <cuda_runtime.h>
#include <stdio.h>

#undef PRINT_EXTREMA_DEBUG_INFO

#undef DEBUG_MEGA_LOOP

namespace popart{

/*************************************************************
 * V6 (with dog array): device side
 *************************************************************/

template<int HEIGHT>
__device__
static
inline uint32_t extrema_count( int indicator, int* extrema_counter )
{
    uint32_t mask = __ballot( indicator ); // bitfield of warps with results

    int ct = __popc( mask );          // horizontal reduce

    int write_index;
    if( threadIdx.x == 0 ) {
        // atomicAdd returns the old value, we consider this the based
        // index for this thread's write operation
        write_index = atomicAdd( extrema_counter, ct );
    }
    // broadcast from thread 0 to all threads in warp
    write_index = __shfl( write_index, 0 );

    // this thread's offset: count only bits below the bit of the own
    // thread index; this provides the 0 result and every result up to ct
    write_index += __popc( mask & ((1 << threadIdx.x) - 1) );

    return write_index;
}

__device__
static
inline void extremum_cmp( float val, float f, uint32_t& gt, uint32_t& lt, uint32_t mask )
{
    gt |= ( ( val > f ) ? mask : 0 );
    lt |= ( ( val < f ) ? mask : 0 );
}


#define TX(dx,dy,dz) tex2DLayered<float>( obj, x+dx, y+dy, z+dz )

__device__
static
inline bool is_extremum( cudaTextureObject_t obj,
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
    inline __device__
    bool first_contrast_ok( const float val ) const;

    /* refine
     * returns -1 : break loop and fail
     *          0 : continue looping
     *          1 : break loop and succeed
     */
    inline __device__
    int refine( float3& d, int3& n, const int width, const int height, const int maxlevel, bool last_it );

    /*
     * returns true  : values after refine make sense
     *         false : they do not
     */
    inline __device__
    bool verify( const float xn, const float yn, const float sn, const int width, const int height, const int maxlevel ) const;
};

template<>
class ModeFunctions<Config::OpenCV>
{
public:
    inline __device__
    bool first_contrast_ok( const float val ) const
    {
        return ( fabsf( val ) >= floorf( d_threshold ) );
    }

    inline __device__
    int refine( float3& d, int3& n, const int width, const int height, const int maxlevel, bool last_it ) const
    {
        // OpenCV mode is a special case because d remain unmodified.
        // Either we return 1, and n has not been modified.
        // Or we quit the loop by exceeding the limit, and reject the point anyway.

        const float3 t = make_float3( fabsf(d.x), fabsf(d.y), fabsf(d.z) );

        if( t.x < 0.5f && t.y < 0.5f && t.z < 0.5f ) {
            // return false, quit the loop, success
            return 1;
        }

        // This test in OpenCV is totally useless in CUDA because the thread
        // would simple idle before failing 7 instructions below anyway.
        // if( t.x > (float)(INT_MAX/3) || t.y > (float)(INT_MAX/3) || t.z > (float)(INT_MAX/3) ) {
            // return false, quit the loop, fail
            // return -1;
        // }

        n.x += rintf( d.x );  // quicker roundf
        n.y += rintf( d.y );  // quicker roundf
        n.z += rintf( d.z );  // quicker roundf

        const int retval = ( n.x < 5 || n.x >= width-5 ||
                             n.y < 5 || n.y >= height-5 ||
                             n.z < 1 || n.z > maxlevel-2 ) ? -1 : 0;
            // if outside of all DoG images (minus border),
            // quit the loop, fail

        return retval;
    }

    inline __device__
    int verify( const float xn, const float yn, const float sn, const int width, const int height, const int maxlevel ) const
    {
        return true;
        // return ( xn < 5 || xn >= width-5 || yn < 5 || yn >= height-5 || sn < 1 || sn > maxlevel-2 ) ? false : true;
    }
};

template<>
class ModeFunctions<Config::VLFeat>
{
public:
    inline __device__
    bool first_contrast_ok( const float val ) const
    {
        return ( fabsf( val ) >= 0.8f * 2.0f * d_threshold );
    }

    inline __device__
    int refine( float3& d, int3& n, const int width, const int height, const int maxlevel, bool last_it ) const
    {
        if( last_it ) return 0;

        float2 t;

        t.x = ((d.x >= 0.6f && n.x < width-2) ?  1.0f : 0.0f )
            + ((d.x <= -0.6f && n.x > 1)? -1.0f : 0.0f );

        t.y = ((d.y >= 0.6f && n.y < height-2)  ?  1.0f : 0.0f )
            + ((d.y <= -0.6f && n.y > 1) ? -1.0f : 0.0f );

        // t.z = ((d.z >= 0.6f && n.z < maxlevel-1)  ?  1 : 0 )
            // + ((d.z <= -0.6f && n.z > 1) ? -1 : 0 );

        if( t.x == 0 && t.y == 0 ) {
            // no more changes
            return 1;
        }

        n.x += t.x;
        n.y += t.y;
        // n.z += t.z; - VLFeat is not changing levels !!!

        return 0;
    }

    inline __device__
    int verify( const float xn, const float yn, const float sn, const int width, const int height, const int maxlevel ) const
    {
        // reject if outside of image bounds or far outside DoG bounds
        return ( ( xn < 0.0f ||
                   xn > width - 1.0f ||
                   yn < 0.0f ||
                   yn > height - 1.0f ||
                   sn < 0.0f ||
                   sn > maxlevel ) ? false
                                   : true );
    }
};

template<>
class ModeFunctions<Config::PopSift>
{
public:
    inline __device__
    bool first_contrast_ok( const float val ) const
    {
        return ( fabsf( val ) >= 1.6f * d_threshold );
    }

    inline __device__
    int refine( float3& d, int3& n, const int width, const int height, const int maxlevel, bool last_it ) const
    {
        if( last_it ) return 0;

        float3 t;

        t.x = ((d.x >= 0.6f && n.x < width-2) ?  1.0f : 0.0f )
            + ((d.x <= -0.6f && n.x > 1)? -1.0f : 0.0f );

        t.y = ((d.y >= 0.6f && n.y < height-2)  ?  1.0f : 0.0f )
            + ((d.y <= -0.6f && n.y > 1) ? -1.0f : 0.0f );

        t.z = ((d.z >= 0.6f && n.z < maxlevel-1)  ?  1.0f : 0.0f )
            + ((d.z <= -0.6f && n.z > 1) ? -1.0f : 0.0f );

        if( t.x == 0 && t.y == 0 && t.z == 0 ) {
            // no more changes
            return 1;
        }

        n.x += t.x;
        n.y += t.y;
        n.z += t.z;

        return 0;
    }

    inline __device__
    int verify( const float xn, const float yn, const float sn, const int width, const int height, const int maxlevel ) const
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
};

template<int sift_mode>
__device__
bool find_extrema_in_dog_v6_sub( cudaTextureObject_t dog,
                                 int                 debug_octave,
                                 int                 level,
                                 int                 width,
                                 int                 height,
                                 const uint32_t      maxlevel,
                                 Extremum&           ec )
{
    ec.xpos    = 0;
    ec.ypos    = 0;
    ec.sigma   = 0;
    ec.num_ori = 0;

    /*
     * First consideration: extrema cannot be found on any outermost edge,
     * one pixel on the left, right, upper, lower edge will never qualify.
     * Also, the upper and lower DoG layer will never qualify. So there is
     * no reason for selecting any of those pixel for the center of a 3x3x3
     * region.
     * Instead, I use groups of 32xHEIGHT threads that read from a 34x34x3 area,
     * but implicitly, they fetch * 64xHEIGHT+2x3 floats (bad luck).
     * To find maxima, compare first on the left edge of the 3x3x3 cube, ie.
     * a 1x3x3 area. If the rightmost 2 threads of a warp (x==30 and 3==31)
     * are not extreme w.r.t. to the left slice, 8 fetch operations.
     */
    const int block_x = blockIdx.x * 32;
    const int block_y = blockIdx.y * blockDim.y;
    const int y       = block_y + threadIdx.y + 1;
    const int x       = block_x + threadIdx.x + 1;

    // int32_t x0 = x;
    // int32_t x1 = x+1;
    // int32_t x2 = x+2;
    // int32_t y0 = y;
    // int32_t y1 = y+1;
    // int32_t y2 = y+2;

    if( sift_mode == Config::OpenCV ) {
        if( x < 5 || y < 5 || x >= width-5 || y >= height-5 ) {
            return false;
        }
    }

    const float val = tex2DLayered<float>( dog, x, y, level );

    ModeFunctions<sift_mode> f;
    if( not f.first_contrast_ok( val ) ) return false;

#if 0
    if( debug_octave==0 && level==2 && x==14 && y==73 ) {
        printf("I am being tested\n");
        printf("thresh = %0.8f\n", floorf( d_threshold ) );
        printf("layer0 = %0.3f %0.3f %0.3f\n"
               "         %0.3f %0.3f %0.3f\n"
               "         %0.3f %0.3f %0.3f\n",
            tex2DLayered<float>( dog, x-1, y-1, level-1 ),
            tex2DLayered<float>( dog, x  , y-1, level-1 ),
            tex2DLayered<float>( dog, x+1, y-1, level-1 ),
            tex2DLayered<float>( dog, x-1, y  , level-1 ),
            tex2DLayered<float>( dog, x  , y  , level-1 ),
            tex2DLayered<float>( dog, x+1, y  , level-1 ),
            tex2DLayered<float>( dog, x-1, y+1, level-1 ),
            tex2DLayered<float>( dog, x  , y+1, level-1 ),
            tex2DLayered<float>( dog, x+1, y+1, level-1 ) );
        printf("layer1 = %0.3f %0.3f %0.3f\n"
               "         %0.3f %0.3f %0.3f\n"
               "         %0.3f %0.3f %0.3f\n",
            tex2DLayered<float>( dog, x-1, y-1, level ),
            tex2DLayered<float>( dog, x  , y-1, level ),
            tex2DLayered<float>( dog, x+1, y-1, level ),
            tex2DLayered<float>( dog, x-1, y  , level ),
            tex2DLayered<float>( dog, x  , y  , level ),
            tex2DLayered<float>( dog, x+1, y  , level ),
            tex2DLayered<float>( dog, x-1, y+1, level ),
            tex2DLayered<float>( dog, x  , y+1, level ),
            tex2DLayered<float>( dog, x+1, y+1, level ) );
        printf("layer2 = %0.3f %0.3f %0.3f\n"
               "         %0.3f %0.3f %0.3f\n"
               "         %0.3f %0.3f %0.3f\n",
            tex2DLayered<float>( dog, x-1, y-1, level+1 ),
            tex2DLayered<float>( dog, x  , y-1, level+1 ),
            tex2DLayered<float>( dog, x+1, y-1, level+1 ),
            tex2DLayered<float>( dog, x-1, y  , level+1 ),
            tex2DLayered<float>( dog, x  , y  , level+1 ),
            tex2DLayered<float>( dog, x+1, y  , level+1 ),
            tex2DLayered<float>( dog, x-1, y+1, level+1 ),
            tex2DLayered<float>( dog, x  , y+1, level+1 ),
            tex2DLayered<float>( dog, x+1, y+1, level+1 ) );
    }
#endif
    if( not is_extremum( dog, x-1, y-1, level-1 ) ) {
        // if( debug_octave==0 && level==2 && x==14 && y==73 ) printf("But I fail\n");
        return false;
    }

    // based on Bemap
    float3 D; // Dx Dy Ds
    float3 DD; // Dxx Dyy Dss
    float3 DX; // Dxy Dxs Dys
    float3 d; // dx dy ds

    float v = val;

    int3 n = make_int3( x, y, level ); // nj ni ns

    int32_t iter = 0;

#define MAX_ITERATIONS 5

    /* must be execute at least once */
    do {
        iter++;

        // const int z = level - 1;
        /* compute gradient */
        const float x2y1z1 = tex2DLayered<float>( dog, n.x+1, n.y  , n.z   );
        const float x0y1z1 = tex2DLayered<float>( dog, n.x-1, n.y  , n.z   );
        const float x1y2z1 = tex2DLayered<float>( dog, n.x  , n.y+1, n.z   );
        const float x1y0z1 = tex2DLayered<float>( dog, n.x  , n.y-1, n.z   );
        const float x1y1z2 = tex2DLayered<float>( dog, n.x  , n.y  , n.z+1 );
        const float x1y1z0 = tex2DLayered<float>( dog, n.x  , n.y  , n.z-1 );
        // D.x = 0.5f * ( x2y1z1 - x0y1z1 );
        // D.y = 0.5f * ( x1y2z1 - x1y0z1 );
        // D.z = 0.5f * ( x1y1z2 - x1y1z0 );
        D.x = scalbnf( x2y1z1 - x0y1z1, -1 );
        D.y = scalbnf( x1y2z1 - x1y0z1, -1 );
        D.z = scalbnf( x1y1z2 - x1y1z0, -1 );

        /* compute Hessian */
        const float x1y1z1 = tex2DLayered<float>( dog, n.x  , n.y  , n.z   );
        // DD.x = x2y1z1 + x0y1z1 - 2.0f * x1y1z1;
        // DD.y = x1y2z1 + x1y0z1 - 2.0f * x1y1z1;
        // DD.z = x1y1z2 + x1y1z0 - 2.0f * x1y1z1;
        DD.x = x2y1z1 + x0y1z1 - scalbnf( x1y1z1, 1 );
        DD.y = x1y2z1 + x1y0z1 - scalbnf( x1y1z1, 1 );
        DD.z = x1y1z2 + x1y1z0 - scalbnf( x1y1z1, 1 );

        const float x0y0z1 = tex2DLayered<float>( dog, n.x-1, n.y-1, n.z   );
        const float x0y1z0 = tex2DLayered<float>( dog, n.x-1, n.y  , n.z-1 );
        const float x0y1z2 = tex2DLayered<float>( dog, n.x-1, n.y  , n.z+1 );
        const float x0y2z1 = tex2DLayered<float>( dog, n.x-1, n.y+1, n.z   );
        const float x1y0z0 = tex2DLayered<float>( dog, n.x  , n.y-1, n.z-1 );
        const float x1y0z2 = tex2DLayered<float>( dog, n.x  , n.y-1, n.z+1 );
        const float x1y2z0 = tex2DLayered<float>( dog, n.x  , n.y+1, n.z-1 );
        const float x1y2z2 = tex2DLayered<float>( dog, n.x  , n.y+1, n.z+1 );
        const float x2y0z1 = tex2DLayered<float>( dog, n.x+1, n.y-1, n.z   );
        const float x2y1z0 = tex2DLayered<float>( dog, n.x+1, n.y  , n.z-1 );
        const float x2y1z2 = tex2DLayered<float>( dog, n.x+1, n.y  , n.z+1 );
        const float x2y2z1 = tex2DLayered<float>( dog, n.x+1, n.y+1, n.z   );
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

        if( solve( A, b ) == false ) {
            d.x = 0;
            d.y = 0;
            d.z = 0;
            break ;
        }
        // printf( "moving (%d,%d) by l: %f (%f,%f)\n", x+1, y+1, d.z, d.x, d.y );

        d = b;

        /* If the translation of the keypoint is big, move the keypoint
         * and re-iterate the computation. Otherwise we are all set.
         */
        const int retval = f.refine( d, n, width, height, maxlevel, iter==MAX_ITERATIONS );

        if( retval == -1 ) {
#ifdef PRINT_EXTREMA_DEBUG_INFO
            printf("Found an extremum at %d %d (o=%d,l=%d) - deviating to much in refinement l:%d (%d,%d)\n", x, y, debug_octave, level, n.z, n.x, n.y );
#endif // PRINT_EXTREMA_DEBUG_INFO
            return false;
        } else if( retval == 1 ) {
            break;
        }
    }
    while( iter < MAX_ITERATIONS ); /* go to next iter */

    if (iter >= MAX_ITERATIONS) {
        if( sift_mode == Config::OpenCV ) {
            /* ensure convergence of interpolation */
#ifdef PRINT_EXTREMA_DEBUG_INFO
            printf("Found an extremum at %d %d (o=%d,l=%d) - rejected in refinement, was moved to l:%d (%d,%d)\n", x, y, debug_octave, level, n.z, n.x, n.y );
#endif // PRINT_EXTREMA_DEBUG_INFO
            return false;
        }
    }

    if( sift_mode == Config::PopSift || sift_mode == Config::VLFeat ) {
        if( d.x >= 1.5f || d.y >= 1.5f || d.z >= 1.5f ) {
#ifdef PRINT_EXTREMA_DEBUG_INFO
            printf("Found an extremum at %d %d (o=%d,l=%d) - rejected because of large d l:%0.2f (%0.2f,%0.2f)\n", x, y, debug_octave, level, d.z, d.x, d.y );
#endif // PRINT_EXTREMA_DEBUG_INFO
            // excessive pixel movement in at least dimension, reject
            return false;
        }
    }

    const float xn      = n.x + d.x;
    const float yn      = n.y + d.y;
    const float sn      = n.z + d.z;

    if( not f.verify( xn, yn, sn, width, height, maxlevel ) ) {
#ifdef PRINT_EXTREMA_DEBUG_INFO
        printf("Found an extremum at %d %d (o=%d,l=%d) - rejected in verification, was moved to l:%0.2f (%0.2f,%0.2f)\n", x, y, debug_octave, level, sn, xn, yn ); // n.z, n.x, n.y );
#endif // PRINT_EXTREMA_DEBUG_INFO
        return false;
    }

    // float contr   = v + 0.5f * (D.x * d.x + D.y * d.y + D.z * d.z);
    const float contr   = v + scalbnf( D.x * d.x + D.y * d.y + D.z * d.z , -1 );
    const float tr      = DD.x + DD.y;
    const float det     = DD.x * DD.y - DX.x * DX.x;
    const float edgeval = tr * tr / det;

    if( sift_mode == Config::PopSift && iter >= MAX_ITERATIONS && ( sn<0 || sn>maxlevel) ) {
#ifdef PRINT_EXTREMA_DEBUG_INFO
        printf("Found an extremum at %d %d (o=%d,l=%d) - rejected in refinement, was moved to l:%d (%d,%d)\n", x, y, debug_octave, level, n.z, n.x, n.y );
#endif
        return false;
    }

    /* negative determinant => curvatures have different signs -> reject it */
    if (det <= 0.0f) {
#ifdef PRINT_EXTREMA_DEBUG_INFO
        printf("Found an extremum at %d %d (o=%d,l=%d) - negative determinant\n", x, y, debug_octave, level );
#endif // PRINT_EXTREMA_DEBUG_INFO
        return false;
    }

    /* accept-reject extremum */
    // if( fabsf(contr) < (d_threshold*2.0f) )
    if( fabsf(contr) < scalbnf( d_threshold, 1 ) )
    {
#ifdef PRINT_EXTREMA_DEBUG_INFO
        printf("Found an extremum at %d %d (o=%d,l=%d) - 2nd peak tresh failed\n", x, y, debug_octave, level );
#endif // PRINT_EXTREMA_DEBUG_INFO
        return false;
    }

    /* reject condition: tr(H)^2/det(H) < (r+1)^2/r */
    if( edgeval >= (d_edge_limit+1.0f)*(d_edge_limit+1.0f)/d_edge_limit ) {
#ifdef PRINT_EXTREMA_DEBUG_INFO
        printf("Found an extremum at %d %d (o=%d,l=%d) - edge tresh failed\n", x, y, debug_octave, level );
#endif // PRINT_EXTREMA_DEBUG_INFO
        return false;
    }

    ec.xpos    = xn;
    ec.ypos    = yn;
    ec.sigma   = d_sigma0 * pow(d_sigma_k, sn); // * 2;
        // const float sigma_k = powf(2.0f, 1.0f / levels );


// #ifdef DEBUG_MEGA_LOOP
//     float sigw = 3.0f * ec.sigma;
//     int   rad  = (int)rintf((3.0f * sigw));
// 
//     if( sn < -1 ) {
//         printf("o/l=%d/%d pos=(%0.1f, %0.1f, %0.1f) n.z=%d d.z=%0.2f det=%0.2f scale=%0.2f pixel radius=%d\n", debug_octave, level, ec.xpos, ec.ypos, sn, n.z, d.z, det, ec.sigma, rad );
//     } else if( sn > maxlevel ) {
//         printf("o/l=%d/%d pos=(%0.1f, %0.1f, %0.1f) n.z=%d d.z=%0.2f det=%0.2f scale=%0.2f pixel radius=%d\n", debug_octave, level, ec.xpos, ec.ypos, sn, n.z, d.z, det, ec.sigma, rad );
//     } else if( sn > maxlevel-1 ) {
//         printf("o/l=%d/%d pos=(%0.1f, %0.1f, %0.1f) n.z=%d d.z=%0.2f det=%0.2f scale=%0.2f pixel radius=%d\n", debug_octave, level, ec.xpos, ec.ypos, sn, n.z, d.z, det, ec.sigma, rad );
//     }
// #endif // DEBUG_MEGA_LOOP

#ifdef PRINT_EXTREMA_DEBUG_INFO
    printf("Found an extremum at %d %d (o=%d,l=%d)     -> x:%.1f y:%.1f z:%.1f sigma:%0.1f\n", x, y, debug_octave, level, xn, yn, sn, ec.sigma );
#endif // PRINT_EXTREMA_DEBUG_INFO

    return true;
}



template<int HEIGHT, int vlfeat_mode>
__global__
void find_extrema_in_dog_v6( cudaTextureObject_t dog,
                             int                 debug_octave,
                             int                 level,
                             int                 width,
                             int                 height,
                             const uint32_t      maxlevel,
                             int*                extrema_counter,
                             Extremum*           d_extrema,
                             int*                d_number_of_blocks,
                             int                 number_of_blocks )
{
    Extremum ec;

    bool indicator = find_extrema_in_dog_v6_sub<vlfeat_mode>( dog, debug_octave, level, width, height, maxlevel, ec );

    uint32_t write_index = extrema_count<HEIGHT>( indicator, extrema_counter );

    if( indicator && write_index < d_max_extrema ) {
        d_extrema[write_index] = ec;
    }

    // without syncthreads, (0,0) threads may precede some calls to extrema_count()
    // in non-(0,0) threads and increase barrier count too early
    __syncthreads();

    // __threadfence(); probably not needed

    if( threadIdx.x == 0 && threadIdx.y == 0 ) {
        int ct = atomicAdd( d_number_of_blocks, 1 );
        if( ct >= number_of_blocks-1 ) {
            int num_ext = atomicMin( extrema_counter, d_max_extrema );
            // printf("counted to %d, num extrema %d\n", ct, num_ext );
            // printf("Number of extrema at level %d: %d\n", level, num_ext );
        }
    }
}


/*************************************************************
 * V6: host side
 *************************************************************/
template<int HEIGHT>
__host__
void Pyramid::find_extrema_v6_sub( const Config& conf )
{
    for( int octave=0; octave<_num_octaves; octave++ ) {
        Octave&      oct_obj = _octaves[octave];

        cudaEvent_t  reset_done_ev  = oct_obj.getEventExtremaDone(0);

        int*  extrema_counters   = oct_obj.getExtremaMgmtD( );
        int*  extrema_num_blocks = oct_obj.getNumberOfBlocks( );

        for( int level=1; level<_levels-2; level++ ) {
            int cols = oct_obj.getWidth();
            int rows = oct_obj.getHeight();

            dim3 block( 32, HEIGHT );
            dim3 grid;
            grid.x  = grid_divide( cols, block.x );
            grid.y  = grid_divide( rows, block.y );

            cudaStream_t oct_str = oct_obj.getStream(level+2);

            cudaEvent_t  upp_ev  = oct_obj.getEventDogDone(level+0);
            cudaEvent_t  mid_ev  = oct_obj.getEventDogDone(level+1);
            // cudaEvent_t  low_ev  = oct_obj.getEventDogDone(level+2); - we are in the same stream

            int*  extrema_counter = &extrema_counters[level];
            int*  num_blocks      = &extrema_num_blocks[level];

            cudaStreamWaitEvent( oct_str, reset_done_ev, 0 );
            cudaStreamWaitEvent( oct_str, upp_ev, 0 );
            cudaStreamWaitEvent( oct_str, mid_ev, 0 );
            // cudaStreamWaitEvent( oct_str, low_ev, 0 ); - we are in the same stream

            switch( conf.getSiftMode() )
            {
            case Config::VLFeat :
                find_extrema_in_dog_v6<HEIGHT,Config::VLFeat>
                    <<<grid,block,0,oct_str>>>
                    ( oct_obj.getDogTexture( ),
                      octave,
                      level,
                      cols,
                      rows,
                      _levels-1,
                      extrema_counter,
                      oct_obj.getExtrema( level ),
                      num_blocks,
                      grid.x * grid.y );
                break;
            case Config::OpenCV :
                find_extrema_in_dog_v6<HEIGHT,Config::OpenCV>
                    <<<grid,block,0,oct_str>>>
                    ( oct_obj.getDogTexture( ),
                      octave,
                      level,
                      cols,
                      rows,
                      _levels-1,
                      extrema_counter,
                      oct_obj.getExtrema( level ),
                      num_blocks,
                      grid.x * grid.y );
                break;
            default :
                find_extrema_in_dog_v6<HEIGHT,Config::PopSift>
                    <<<grid,block,0,oct_str>>>
                    ( oct_obj.getDogTexture( ),
                      octave,
                      level,
                      cols,
                      rows,
                      _levels-1,
                      extrema_counter,
                      oct_obj.getExtrema( level ),
                      num_blocks,
                      grid.x * grid.y );
                break;
            }

            cudaEvent_t  extrema_done_ev  = oct_obj.getEventExtremaDone(level+2);
            cudaEventRecord( extrema_done_ev, oct_str );
        }
    }
}

__host__
void Pyramid::find_extrema_v6( const Config& conf )
{
#define MANYLY(H) \
    find_extrema_v6_sub<H> ( conf );

    // MANYLY(1)
    // MANYLY(2)
    // MANYLY(3)
    MANYLY(4)
    // MANYLY(5)
    // MANYLY(6)
    // MANYLY(7)
    // MANYLY(8)
    // MANYLY(16)
    // fails // MANYLY(32)
}

} // namespace popart

