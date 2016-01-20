#include "s_pyramid.h"
#include "s_solve.h"
#include "debug_macros.h"
#include "assist.h"
#include "clamp.h"
#include <cuda_runtime.h>

#include <cub/block/block_scan.cuh>

namespace popart {
/*************************************************************
 * V5: device side
 *************************************************************/

#if 1
template<int HEIGHT>
__device__
static
inline int extrema_count( int indicator, ExtremaMgmt* mgmt )
{
    typedef cub::BlockScan<int, 32, cub::BLOCK_SCAN_RAKING, HEIGHT> BlockScan;

    // Allocate shared memory for BlockScan
    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ int base_offset;

    int offset;
    int total;

    // Collectively compute the block-wide exclusive prefix sum
    BlockScan(temp_storage).ExclusiveSum( indicator, offset, total );

    if( threadIdx.x == 0 && threadIdx.y == 0 ) {
        // atomicAdd returns the old value, we consider this the based
        // index for this thread's write operation
        base_offset = atomicAdd( &mgmt->counter, total );
    }
    int write_index = base_offset + offset;

    return write_index;
}
#else
template<int HEIGHT>
__device__
static
inline uint32_t extrema_count( int indicator, ExtremaMgmt* mgmt )
{
    uint32_t mask = __ballot( indicator ); // bitfield of warps with results

    uint32_t ct = __popc( mask );          // horizontal reduce

    uint32_t write_index;
    if( threadIdx.x == 0 ) {
        // atomicAdd returns the old value, we consider this the based
        // index for this thread's write operation
        write_index = atomicAdd( &mgmt->counter, ct );
    }
    // broadcast from thread 0 to all threads in warp
    write_index = __shfl( write_index, 0 );

    // this thread's offset: count only bits below the bit of the own
    // thread index; this provides the 0 result and every result up to ct
    write_index += __popc( mask & ((1 << threadIdx.x) - 1) );

    return write_index;
}
#endif

__device__
static
inline void extremum_cmp( float val, float f, uint32_t& gt, uint32_t& lt, uint32_t mask )
{
    gt |= ( ( val > f ) ? mask : 0 );
    lt |= ( ( val < f ) ? mask : 0 );
}

#ifdef USE_DOG_ARRAY
#define TX(dx,dy,dz) tex2DLayered<float>( obj, x+dx, y+dy, z+dz )
__device__
static
inline bool is_extremum( cudaTextureObject_t obj,
                            int x, int y, int z )
{
    uint32_t gt = 0;
    uint32_t lt = 0;

    float val0 = TX( 0, 1, 1 );
    float val2 = TX( 2, 1, 1 );
    float val  = TX( 1, 1, 1 );


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
#else // not USE_DOG_ARRAY
__device__
inline bool is_extremum( Plane2D_float& dog0, Plane2D_float& dog1, Plane2D_float& dog2,
                            uint32_t y0, uint32_t y1, uint32_t y2,
                            uint32_t x0, uint32_t x1, uint32_t x2 )
{
    // somewhat annoying: to read center val, x1==31 requires a second 128-byte read
    // so: read left value first (one 128-byte read)
    //     read right value after (30 floats from cache, 2 from next 128-byte read)
    //     finally, read center value (from cache)
    uint32_t gt = 0;
    uint32_t lt = 0;

    float* dog0y0 = dog0.ptr(y0);
    float* dog0y1 = dog0.ptr(y1);
    float* dog0y2 = dog0.ptr(y2);
    float* dog1y0 = dog1.ptr(y0);
    float* dog1y1 = dog1.ptr(y1);
    float* dog1y2 = dog1.ptr(y2);
    float* dog2y0 = dog2.ptr(y0);
    float* dog2y1 = dog2.ptr(y1);
    float* dog2y2 = dog2.ptr(y2);

    float val0 = dog1y1[x0];
    float val2 = dog1y1[x2];
    float val  = dog1y1[x1];

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
    extremum_cmp( val, dog1y0[x0], gt, lt, 0x00800000 ); // ( 0x01<<7 ) << 16
    extremum_cmp( val, dog1y2[x0], gt, lt, 0x00200000 ); // ( 0x01<<5 ) << 16
    extremum_cmp( val, dog0y0[x0], gt, lt, 0x80000000 ); // ( 0x01<<6 ) << 24
    extremum_cmp( val, dog0y2[x0], gt, lt, 0x40000000 ); // ( 0x01<<6 ) << 24
    extremum_cmp( val, dog0y1[x0], gt, lt, 0x20000000 ); // ( 0x01<<6 ) << 24
    extremum_cmp( val, dog2y0[x0], gt, lt, 0x00008000 ); // ( 0x01<<6 ) <<  8
    extremum_cmp( val, dog2y1[x0], gt, lt, 0x00004000 ); // ( 0x01<<6 ) <<  8
    extremum_cmp( val, dog2y2[x0], gt, lt, 0x00002000 ); // ( 0x01<<6 ) <<  8

    if( ( gt != 0xe0e4e000 ) && ( lt != 0xe0e4e000 ) ) return false;

    // 3rd group: remaining 2 cache misses in own layer
    extremum_cmp( val, dog1y0[x1], gt, lt, 0x00010000 ); // ( 0x01<<0 ) << 16
    extremum_cmp( val, dog1y0[x2], gt, lt, 0x00020000 ); // ( 0x01<<1 ) << 16
    extremum_cmp( val, dog1y2[x1], gt, lt, 0x00100000 ); // ( 0x01<<4 ) << 16
    extremum_cmp( val, dog1y2[x2], gt, lt, 0x00080000 ); // ( 0x01<<3 ) << 16

    if( ( gt != 0xe0ffe000 ) && ( lt != 0xe0ffe000 ) ) return false;

    // 4th group: 3 cache misses higher layer
    extremum_cmp( val, dog0y0[x1], gt, lt, 0x01000000 ); // ( 0x01<<0 ) << 24
    extremum_cmp( val, dog0y0[x2], gt, lt, 0x02000000 ); // ( 0x01<<1 ) << 24
    extremum_cmp( val, dog0y1[x1], gt, lt, 0x00000004 ); // ( 0x01<<2 )
    extremum_cmp( val, dog0y1[x2], gt, lt, 0x04000000 ); // ( 0x01<<2 ) << 24
    extremum_cmp( val, dog0y2[x1], gt, lt, 0x10000000 ); // ( 0x01<<4 ) << 24
    extremum_cmp( val, dog0y2[x2], gt, lt, 0x08000000 ); // ( 0x01<<3 ) << 24

    if( ( gt != 0xffffe004 ) && ( lt != 0xffffe004 ) ) return false;

    // 5th group: 3 cache misss lower layer
    extremum_cmp( val, dog2y0[x1], gt, lt, 0x00000100 ); // ( 0x01<<0 ) <<  8
    extremum_cmp( val, dog2y0[x2], gt, lt, 0x00000200 ); // ( 0x01<<1 ) <<  8
    extremum_cmp( val, dog2y1[x1], gt, lt, 0x00000001 ); // ( 0x01<<0 )
    extremum_cmp( val, dog2y1[x2], gt, lt, 0x00000400 ); // ( 0x01<<2 ) <<  8
    extremum_cmp( val, dog2y2[x1], gt, lt, 0x00001000 ); // ( 0x01<<4 ) <<  8
    extremum_cmp( val, dog2y2[x2], gt, lt, 0x00000800 ); // ( 0x01<<3 ) <<  8

    if( ( gt != 0xffffff05 ) && ( lt != 0xffffff05 ) ) return false;
    
    return true;
}
#endif // not USE_DOG_ARRAY

#ifdef USE_DOG_ARRAY
__device__
bool find_extrema_in_dog_v5_sub( cudaTextureObject_t dog,
                                   int                 level,
                                   int                 width,
                                   int                 height,
                                   float               edge_limit,
                                   float               threshold,
                                   const uint32_t      maxlevel,
                                   ExtremumCandidate&  ec )
{
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
    int32_t block_x = blockIdx.x * 32;
    int32_t block_y = blockIdx.y * blockDim.y;
    int32_t y       = block_y + threadIdx.y;
    int32_t x       = block_x + threadIdx.x;

    // int32_t x0 = x;
    // int32_t x1 = x+1;
    // int32_t x2 = x+2;
    // int32_t y0 = y;
    // int32_t y1 = y+1;
    // int32_t y2 = y+2;

    float val = tex2DLayered<float>( dog, x+1, y+1, level );

    if( fabs( val ) < threshold ) {
        // atomicAdd( &debug_r.under_threshold, 1 );
        return false;
    }

    if( not is_extremum( dog, x, y, level-1 ) ) {
        // atomicAdd( &debug_r.not_extremum, 1 );
        return false;
    }

    // based on Bemap
    float Dx  = 0.0f;
    float Dy  = 0.0f;
    float Ds  = 0.0f;
    float Dxx = 0.0f;
    float Dyy = 0.0f;
    float Dss = 0.0f;
    float Dxy = 0.0f;
    float Dxs = 0.0f;
    float Dys = 0.0f;
    float dx  = 0.0f;
    float dy  = 0.0f;
    float ds  = 0.0f;

    float v = val;

    int32_t ni = y+1; // y1w;
    int32_t nj = x+1;
    int32_t ns = level;

    int32_t tx = 0;
    int32_t ty = 0;
    int32_t ts = 0;

    int32_t iter;

    /* must be execute at least once */
    for ( iter = 0; iter < 5; iter++) {
        const int z = level - 1;
        /* compute gradient */
        const float x2y1z1 = tex2DLayered<float>( dog, x+2, y+1, z+1 );
        const float x0y1z1 = tex2DLayered<float>( dog, x+0, y+1, z+1 );
        const float x1y2z1 = tex2DLayered<float>( dog, x+1, y+2, z+1 );
        const float x1y0z1 = tex2DLayered<float>( dog, x+1, y+0, z+1 );
        const float x1y1z2 = tex2DLayered<float>( dog, x+1, y+1, z+2 );
        const float x1y1z0 = tex2DLayered<float>( dog, x+1, y+1, z+0 );
        Dx = 0.5 * ( x2y1z1 - x0y1z1 );
        Dy = 0.5 * ( x1y2z1 - x1y0z1 );
        Ds = 0.5 * ( x1y1z2 - x1y1z0 );

        /* compute Hessian */
        const float x1y1z1 = tex2DLayered<float>( dog, x+1, y+1, z+1 );
        Dxx = x2y1z1 + x0y1z1 - 2.0 * x1y1z1;
        Dyy = x1y2z1 + x1y0z1 - 2.0 * x1y1z1;
        Dss = x1y1z2 + x1y1z0 - 2.0 * x1y1z1;

        const float x0y0z1 = tex2DLayered<float>( dog, x+0, y+0, z+1 );
        const float x0y1z0 = tex2DLayered<float>( dog, x+0, y+1, z+0 );
        const float x0y1z2 = tex2DLayered<float>( dog, x+0, y+1, z+2 );
        const float x0y2z1 = tex2DLayered<float>( dog, x+0, y+2, z+1 );
        const float x1y0z0 = tex2DLayered<float>( dog, x+1, y+0, z+0 );
        const float x1y0z2 = tex2DLayered<float>( dog, x+1, y+0, z+2 );
        const float x1y2z0 = tex2DLayered<float>( dog, x+1, y+2, z+0 );
        const float x1y2z2 = tex2DLayered<float>( dog, x+1, y+2, z+2 );
        const float x2y0z1 = tex2DLayered<float>( dog, x+2, y+0, z+1 );
        const float x2y1z0 = tex2DLayered<float>( dog, x+2, y+1, z+0 );
        const float x2y1z2 = tex2DLayered<float>( dog, x+2, y+1, z+2 );
        const float x2y2z1 = tex2DLayered<float>( dog, x+2, y+2, z+1 );
        Dxy = 0.25f * ( x2y2z1 + x0y0z1 - x0y2z1 - x2y0z1 );
        Dxs = 0.25f * ( x2y1z2 + x0y1z0 - x0y1z2 - x2y1z0 );
        Dys = 0.25f * ( x1y2z2 + x1y0z0 - x1y2z0 - x1y0z2 );

        float b[3];
        float A[3][3];

        /* Solve linear system. */
        A[0][0] = Dxx;
        A[1][1] = Dyy;
        A[2][2] = Dss;
        A[1][0] = A[0][1] = Dxy;
        A[2][0] = A[0][2] = Dxs;
        A[2][1] = A[1][2] = Dys;

        b[0] = -Dx;
        b[1] = -Dy;
        b[2] = -Ds;

        if( solve( A, b ) == false ) {
            dx = 0;
            dy = 0;
            ds = 0;
            break ;
        }

        dx = b[0];
        dy = b[1];
        ds = b[2];

        /* If the translation of the keypoint is big, move the keypoint
         * and re-iterate the computation. Otherwise we are all set.
         */
        if( fabs(ds) < 0.5f && fabs(dy) < 0.5f && fabs(dx) < 0.5f) break;

        tx = ((dx >= 0.5f && nj < width-2) ?  1 : 0 )
           + ((dx <= -0.5f && nj > 1)? -1 : 0 );

        ty = ((dy >= 0.5f && ni < height-2)  ?  1 : 0 )
           + ((dy <= -0.5f && ni > 1) ? -1 : 0 );

        ts = ((ds >= 0.5f && ns < maxlevel-1)  ?  1 : 0 )
           + ((ds <= -0.5f && ns > 1) ? -1 : 0 );

        ni += ty;
        nj += tx;
        ns += ts;
    } /* go to next iter */

    /* ensure convergence of interpolation */
    if (iter >= 5) {
        // atomicAdd( &debug_r.convergence_failure, 1 );
        return false;
    }

    float contr   = v + 0.5f * (Dx * dx + Dy * dy + Ds * ds);
    float tr      = Dxx + Dyy;
    float det     = Dxx * Dyy - Dxy * Dxy;
    float edgeval = tr * tr / det;
    float xn      = nj + dx;
    float yn      = ni + dy;
    float sn      = ns + ds;

    /* negative determinant => curvatures have different signs -> reject it */
    if (det <= 0.0) {
        // atomicAdd( &debug_r.determinant_zero, 1 );
        return false;
    }

    /* accept-reject extremum */
    if( fabs(contr) < (threshold*2.0f) ) {
        // atomicAdd( &debug_r.thresh_exceeded, 1 );
        return false;
    }

    /* reject condition: tr(H)^2/det(H) < (r+1)^2/r */
    if( edgeval > (edge_limit+1.0f)*(edge_limit+1.0f)/edge_limit ) {
        // atomicAdd( &debug_r.edge_exceeded, 1 );
        return false;
    }

    ec.xpos    = xn;
    ec.ypos    = yn;
    ec.sigma   = d_sigma0 * pow(d_sigma_k, sn);
            // key_candidate->sigma = sigma0 * pow(sigma_k, sn);
        // ec.value   = 0;
        // ec.edge    = 0;
    ec.angle_from_bemap = 0;
    ec.not_a_keypoint   = 0;

    return true;
}
#else // not USE_DOG_ARRAY
__device__
static
bool find_extrema_in_dog_v5_sub( Plane2D_float&     dog0,
                                   Plane2D_float&     dog1,
                                   Plane2D_float&     dog2,
                                   float              edge_limit,
                                   float              threshold,
                                   const uint32_t     level,
                                   const uint32_t     maxlevel,
                                   ExtremumCandidate& ec )
{
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
    int32_t block_x = blockIdx.x * 32;
    int32_t block_y = blockIdx.y * blockDim.y;
    int32_t y       = block_y + threadIdx.y;
    int32_t x       = block_x + threadIdx.x;
    // int32_t z       = 0;

    int width  = dog0.getWidth();
    int height = dog0.getHeight();

    if ( x+2 >= width ) {
        // atomicAdd( &debug_r.too_wide, 1 );
        return false;
    }
    if ( y+2 >= height ) {
        // atomicAdd( &debug_r.too_high, 1 );
        return false;
    }

    int32_t x0 = clamp( x,   width );
    int32_t x1 = clamp( x+1, width );
    int32_t x2 = clamp( x+2, width );
    int32_t y0 = clamp( y,   height );
    int32_t y1 = clamp( y+1, height );
    int32_t y2 = clamp( y+2, height );
    // int32_t y0w     = y * pitch;
    // int32_t y1w     = (y+1) * pitch;
    // int32_t y2w     = (y+2) * pitch;

    float val = dog1.ptr(y1)[x1];

    if( fabs( val ) < threshold ) {
        // atomicAdd( &debug_r.under_threshold, 1 );
        return false;
    }

    if( not is_extremum( dog0, dog1, dog2, y0, y1, y2, x0, x1, x2 ) ) {
        // atomicAdd( &debug_r.not_extremum, 1 );
        return false;
    }

    // based on Bemap
    float Dx  = 0.0f;
    float Dy  = 0.0f;
    float Ds  = 0.0f;
    float Dxx = 0.0f;
    float Dyy = 0.0f;
    float Dss = 0.0f;
    float Dxy = 0.0f;
    float Dxs = 0.0f;
    float Dys = 0.0f;
    float dx  = 0.0f;
    float dy  = 0.0f;
    float ds  = 0.0f;

    float v = val;

    int32_t ni = y+1; // y1w;
    int32_t nj = x1;
    int32_t ns = level;

    int32_t tx = 0;
    int32_t ty = 0;
    int32_t ts = 0;

    int32_t iter;

    /* must be execute at least once */
    for ( iter = 0; iter < 5; iter++) {
        /* compute gradient */
        Dx = 0.5 * ( dog1.ptr(y1)[x2] - dog1.ptr(y1)[x0] );
        Dy = 0.5 * ( dog1.ptr(y2)[x1] - dog1.ptr(y0)[x1] );
        Ds = 0.5 * ( dog2.ptr(y1)[x1] - dog0.ptr(y1)[x1] );

        /* compute Hessian */
        Dxx = dog1.ptr(y1)[x2] + dog1.ptr(y1)[x0] - 2.0 * dog1.ptr(y1)[x1];
        Dyy = dog1.ptr(y2)[x1] + dog1.ptr(y0)[x1] - 2.0 * dog1.ptr(y1)[x1];
        Dss = dog2.ptr(y1)[x1] + dog0.ptr(y1)[x1] - 2.0 * dog1.ptr(y1)[x1];

        Dxy = 0.25f * ( dog1.ptr(y2)[x2] + dog1.ptr(y0)[x0]
                      - dog1.ptr(y2)[x0] - dog1.ptr(y0)[x2] );
        Dxs = 0.25f * ( dog2.ptr(y1)[x2] + dog0.ptr(y1)[x0]
                      - dog2.ptr(y1)[x0] - dog0.ptr(y1)[x2] );
        Dys = 0.25f * ( dog2.ptr(y2)[x1] + dog0.ptr(y0)[x1]
                      - dog0.ptr(y2)[x1] - dog2.ptr(y0)[x1] );

        float b[3];
        float A[3][3];

        /* Solve linear system. */
        A[0][0] = Dxx;
        A[1][1] = Dyy;
        A[2][2] = Dss;
        A[1][0] = A[0][1] = Dxy;
        A[2][0] = A[0][2] = Dxs;
        A[2][1] = A[1][2] = Dys;

        b[0] = -Dx;
        b[1] = -Dy;
        b[2] = -Ds;

        if( solve( A, b ) == false ) {
            dx = 0;
            dy = 0;
            ds = 0;
            break ;
        }

        dx = b[0];
        dy = b[1];
        ds = b[2];

        /* If the translation of the keypoint is big, move the keypoint
         * and re-iterate the computation. Otherwise we are all set.
         */
        if( fabs(ds) < 0.5f && fabs(dy) < 0.5f && fabs(dx) < 0.5f) break;

        tx = ((dx >= 0.5f && nj < width-2) ?  1 : 0 )
           + ((dx <= -0.5f && nj > 1)? -1 : 0 );

        ty = ((dy >= 0.5f && ni < height-2)  ?  1 : 0 )
           + ((dy <= -0.5f && ni > 1) ? -1 : 0 );

        ts = ((ds >= 0.5f && ns < maxlevel-1)  ?  1 : 0 )
           + ((ds <= -0.5f && ns > 1) ? -1 : 0 );

        ni += ty;
        nj += tx;
        ns += ts;
    } /* go to next iter */

    /* ensure convergence of interpolation */
    if (iter >= 5) {
        // atomicAdd( &debug_r.convergence_failure, 1 );
        return false;
    }

    float contr   = v + 0.5f * (Dx * dx + Dy * dy + Ds * ds);
    float tr      = Dxx + Dyy;
    float det     = Dxx * Dyy - Dxy * Dxy;
    float edgeval = tr * tr / det;
    float xn      = nj + dx;
    float yn      = ni + dy;
    float sn      = ns + ds;

    /* negative determinant => curvatures have different signs -> reject it */
    if (det <= 0.0) {
        // atomicAdd( &debug_r.determinant_zero, 1 );
        return false;
    }

    /* accept-reject extremum */
    if( fabs(contr) < (threshold*2.0f) ) {
        // atomicAdd( &debug_r.thresh_exceeded, 1 );
        return false;
    }

    /* reject condition: tr(H)^2/det(H) < (r+1)^2/r */
    if( edgeval > (edge_limit+1.0f)*(edge_limit+1.0f)/edge_limit ) {
        // atomicAdd( &debug_r.edge_exceeded, 1 );
        return false;
    }

    ec.xpos    = xn;
    ec.ypos    = yn;
    ec.sigma   = d_sigma0 * pow(d_sigma_k, sn);
            // key_candidate->sigma = sigma0 * pow(sigma_k, sn);
        // ec.value   = 0;
        // ec.edge    = 0;
    ec.angle_from_bemap = 0;
    ec.not_a_keypoint   = 0;

    return true;
}
#endif // not USE_DOG_ARRAY

#ifdef USE_DOG_ARRAY
template<int HEIGHT>
__global__
void find_extrema_in_dog_v5( cudaTextureObject_t dog,
                             int                 level,
                             int                 width,
                             int                 height,
                             float               edge_limit,
                             float               threshold,
                             const uint32_t      maxlevel,
                             ExtremaMgmt*        mgmt_array,
                             ExtremumCandidate*  d_extrema )
{
    ExtremaMgmt* mgmt = &mgmt_array[level];
    ExtremumCandidate ec;

    bool indicator = find_extrema_in_dog_v5_sub( dog, level, width, height, edge_limit, threshold, maxlevel, ec );

    uint32_t write_index = extrema_count<HEIGHT>( indicator, mgmt );

    if( indicator && write_index < mgmt->max1 ) {
        // atomicAdd( &debug_r.continuing, 1 );
        // __syncthreads();

        d_extrema[write_index] = ec;
    } else {
        // atomicAdd( &debug_r.max_exceeded, 1 );
    }
}
#else // not USE_DOG_ARRAY
template<int HEIGHT>
__global__
void find_extrema_in_dog_v5( Plane2D_float      dog_upper,
                             Plane2D_float      dog_here,
                             Plane2D_float      dog_lower,
                             float              edge_limit,
                             float              threshold,
                             const uint32_t     level,
                             const uint32_t     maxlevel,
                             ExtremaMgmt*       mgmt_array,
                             ExtremumCandidate* d_extrema )
{
    ExtremaMgmt* mgmt = &mgmt_array[level];
    ExtremumCandidate ec;

    bool indicator = find_extrema_in_dog_v5_sub( dog_upper, dog_here, dog_lower, edge_limit, threshold, level, maxlevel, ec );

    uint32_t write_index = extrema_count<HEIGHT>( indicator, mgmt );

    if( indicator && write_index < mgmt->max1 ) {
        // atomicAdd( &debug_r.continuing, 1 );
        // __syncthreads();

        d_extrema[write_index] = ec;
    } else {
        // atomicAdd( &debug_r.max_exceeded, 1 );
    }
}
#endif // not USE_DOG_ARRAY

__global__
void fix_extrema_count_v5( ExtremaMgmt* mgmt_array, uint32_t mgmt_level )
{
    ExtremaMgmt* mgmt = &mgmt_array[mgmt_level];

    mgmt->counter = min( mgmt->counter, mgmt->max1 );

    // printf("%s>%d - %d\n", __FILE__, __LINE__, mgmt->counter );
}

#if 0
__global__
void start_orientation_v5( ExtremumCandidate* extrema,
                           ExtremaMgmt*       mgmt,
                           const float*       layer,
                           int                layer_pitch,
                           int                layer_height )
{
    mgmt->counter = min( mgmt->counter, mgmt->max1 );

    compute_keypoint_orientations_v2
        <<<mgmt->counter,16>>>
        ( extrema,
          mgmt,
          layer,
          layer_pitch,
          layer_height );
}
#endif

/*************************************************************
 * V5: host side
 *************************************************************/
template<int HEIGHT>
__host__
void Pyramid::find_extrema_v5_sub( float edgeLimit, float threshold )
{
    // cerr << "Entering " << __FUNCTION__ << " - bitfield, 32x" << height << " kernels" << endl;

#if 0
    cudaDeviceSynchronize();
    ReturnReasons a;
    a.too_wide = 0;
    a.too_high = 0;
    a.under_threshold = 0;
    a.not_extremum = 0;
    a.convergence_failure = 0;
    a.determinant_zero = 0;
    a.thresh_exceeded = 0;
    a.edge_exceeded = 0;
    a.max_exceeded = 0;
    a.continuing = 0;
    cudaMemcpyToSymbol( debug_r, &a, sizeof(ReturnReasons), 0, cudaMemcpyHostToDevice );
#endif

    _keep_time_extrema_v5.start();

    for( int octave=0; octave<_num_octaves; octave++ ) {
        for( int level=1; level<_levels-2; level++ ) {
            int cols = _octaves[octave].getData(level).getCols();
            int rows = _octaves[octave].getData(level).getRows();
            dim3 block( 32, HEIGHT );
            dim3 grid;
            grid.x  = grid_divide( cols, block.x );
            grid.y  = grid_divide( rows, block.y );

#if 0
        cerr << "In " << __FUNCTION__ << endl
             << "    Configuration for octave " << octave << " and level " << level << endl
             << "      Horiz: layer size: "
             << _octaves[octave].getData(level).getWidth() << "x" << _octaves[octave].getData(level).getHeight() << endl
             << "      Vert: layer size: "
             << _octaves[octave].getData2(level).getWidth() << "x" << _octaves[octave].getData2(level).getHeight() << endl
             << "      grid: "
             << "(" << grid.x << "," << grid.y << "," << grid.z << ")"
             << " block: "
             << "(" << block.x << "," << block.y << "," << block.z << ")" << endl;
#endif

#ifdef USE_DOG_ARRAY
            find_extrema_in_dog_v5<HEIGHT>
                <<<grid,block>>>
                ( _octaves[octave].getDogTexture( ),
                  level,
                  cols,
                  rows,
                  edgeLimit,
                  threshold,
                  _levels,
                  _octaves[octave].getExtremaMgmtD( ),
                  _octaves[octave].getExtrema( level ) );
#else // not USE_DOG_ARRAY
            Plane2D_float& d0( _octaves[octave].getDogData( level-1 ) );
            Plane2D_float& d1( _octaves[octave].getDogData( level   ) );
            Plane2D_float& d2( _octaves[octave].getDogData( level+1 ) );

            find_extrema_in_dog_v5<THE_HEIGHT>
                <<<grid,block>>>
                ( d0, d1, d2,
                  edgeLimit,
                  threshold,
                  level,
                  _levels,
                  _octaves[octave].getExtremaMgmtD( ),
                  _octaves[octave].getExtrema( level ) );
#endif // not USE_DOG_ARRAY

#if 1
            fix_extrema_count_v5
                <<<1,1>>>
                ( _octaves[octave].getExtremaMgmtD( ),
                  level );
#else
    // this does not work yet: I have no idea how to link with CUDA
    // and still achieve dynamic parallelism
            start_orientation_v5
                <<<1,1>>>
                ( _octaves[octave].getExtrema( level ),
                  _octaves[octave].getExtremaMgmtD( level ),
                  d1,
                  _octaves[octave].getPitch( ),
                  _octaves[octave].getHeight( ) );
#endif
        }
    }
    cudaDeviceSynchronize( );
    cudaError_t err = cudaGetLastError();
    POP_CUDA_FATAL_TEST( err, "find_extrema_in_dog_v5 failed: " );

    _keep_time_extrema_v5.stop();

#if 0
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol( &a, debug_r, sizeof(ReturnReasons), 0, cudaMemcpyDeviceToHost );
    cerr << __FILE__ << ":" << __LINE__ << endl
         << "reasons for returning:" << endl
         << "  too wide: " << a.too_wide << endl
         << "  too high: " << a.too_high << endl
         << "  under threshold: " << a.under_threshold << endl
         << "  not extremum: " << a.not_extremum << endl
         << "  convergence failure: " << a.convergence_failure << endl
         << "  determinant zero: " << a.determinant_zero << endl
         << "  threshold exceeded: " << a.thresh_exceeded << endl
         << "  edge limit exceeded: " << a.edge_exceeded << endl
         << "  max exceeded: " << a.max_exceeded << endl
         << "  everything OK: " << a.continuing << endl
         << endl;
#endif
}

__host__
void Pyramid::find_extrema_v5( float edgeLimit, float threshold )
{
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaError_t err;
    err = cudaEventCreate( &start );
    POP_CUDA_FATAL_TEST( err, "event create failed: " );
    err = cudaEventCreate( &stop );
    POP_CUDA_FATAL_TEST( err, "event create failed: " );

    float diff;

#define MANYLY(H) \
    err = cudaEventRecord( start, 0 ); \
    POP_CUDA_FATAL_TEST( err, "event record failed: " ); \
    \
    find_extrema_v5_sub<H> ( edgeLimit, threshold ); \
    \
    err = cudaEventRecord( stop, 0 ); \
    POP_CUDA_FATAL_TEST( err, "event record failed: " ); \
    err = cudaStreamSynchronize( 0 ); \
    POP_CUDA_FATAL_TEST( err, "stream sync failed: " ); \
    err = cudaEventElapsedTime( &diff, start, stop ); \
    POP_CUDA_FATAL_TEST( err, "elapsed time failed: " ); \
    cerr << "Time for find_extrema_v5_sub<" #H ">: " << diff << " ms" << endl;

    MANYLY(1)
    // MANYLY(2)
    // MANYLY(3)
    // MANYLY(4)
    // MANYLY(5)
    // MANYLY(6)
    // MANYLY(7)
    // MANYLY(8)
    // MANYLY(16)
    // MANYLY(32)

    err = cudaEventDestroy( start );
    POP_CUDA_FATAL_TEST( err, "event destroy failed: " );
    err = cudaEventDestroy( stop );
    POP_CUDA_FATAL_TEST( err, "event destroy failed: " );
}

} // namespace popart

