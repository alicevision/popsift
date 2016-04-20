#include "sift_pyramid.h"
#include "s_sigma.h"
#include "s_solve.h"
#include "debug_macros.h"
#include "assist.h"
#include "clamp.h"
#include <cuda_runtime.h>
#include <stdio.h>

namespace popart{

/*************************************************************
 * V6 (with dog array): device side
 *************************************************************/

template<int HEIGHT>
__device__
static
inline uint32_t extrema_count( int indicator, ExtremaMgmt* mgmt )
{
    uint32_t mask = __ballot( indicator ); // bitfield of warps with results

    uint32_t ct = __popc( mask );          // horizontal reduce

    int write_index;
    if( threadIdx.x == 0 ) {
        // atomicAdd returns the old value, we consider this the based
        // index for this thread's write operation
        write_index = mgmt->atomicAddCounter( ct );
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

__device__
bool find_extrema_in_dog_v6_sub( cudaTextureObject_t dog,
                                 int                 level,
                                 int                 width,
                                 int                 height,
                                 const uint32_t      maxlevel,
                                 Extremum&           ec )
{
    ec.xpos    = 0;
    ec.ypos    = 0;
    ec.sigma   = 0;
    ec.orientation = 0;

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

    if( fabs( val ) < d_threshold ) {
        return false;
    }

    if( not is_extremum( dog, x, y, level-1 ) ) {
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
        return false;
    }

    /* accept-reject extremum */
    if( fabs(contr) < (d_threshold*2.0f) ) {
        return false;
    }

    /* reject condition: tr(H)^2/det(H) < (r+1)^2/r */
    if( edgeval >= (d_edge_limit+1.0f)*(d_edge_limit+1.0f)/d_edge_limit ) {
        return false;
    }

    ec.xpos    = xn;
    ec.ypos    = yn;
    ec.sigma   = d_sigma0 * pow(d_sigma_k, sn);
    // key_candidate->sigma = sigma0 * pow(sigma_k, sn);
    // ec.value   = 0;
    // ec.edge    = 0;
    ec.orientation = 0;

    return true;
}



template<int HEIGHT>
__global__
void find_extrema_in_dog_v6( cudaTextureObject_t dog,
                             int                 level,
                             int                 width,
                             int                 height,
                             const uint32_t      maxlevel,
                             ExtremaMgmt*        mgmt_array,
                             Extremum*           d_extrema )
{
    ExtremaMgmt* mgmt = &mgmt_array[level];
    Extremum ec;

    bool indicator = find_extrema_in_dog_v6_sub( dog, level, width, height, maxlevel, ec );

    uint32_t write_index = extrema_count<HEIGHT>( indicator, mgmt );

    if( indicator && write_index < d_max_extrema ) {
        d_extrema[write_index] = ec;
    }
}


__global__
void reset_extrema_count_v6( ExtremaMgmt* mgmt_array, uint32_t mgmt_level )
{
    ExtremaMgmt* mgmt = &mgmt_array[mgmt_level];

    mgmt->resetCounter();
}

__global__
void fix_extrema_count_v6( ExtremaMgmt* mgmt_array, uint32_t mgmt_level )
{
    ExtremaMgmt* mgmt = &mgmt_array[mgmt_level];

    mgmt->clampCounter1();

    printf("Number of extrema: %d\n", mgmt->getCounter() );
}

/*************************************************************
 * V6: host side
 *************************************************************/
template<int HEIGHT>
__host__
void Pyramid::find_extrema_v6_sub( )
{
    for( int octave=0; octave<_num_octaves; octave++ ) {
        for( int level=1; level<_levels-2; level++ ) {
            Octave&      oct_obj = _octaves[octave];
            cudaStream_t oct_str = oct_obj.getStream(level);

            reset_extrema_count_v6
                <<<1,1,0,oct_str>>>
                ( _octaves[octave].getExtremaMgmtD( ), level );
        }
    }

    cudaDeviceSynchronize();

    for( int octave=0; octave<_num_octaves; octave++ ) {
        for( int level=1; level<_levels-2; level++ ) {
            int cols = _octaves[octave].getData(level).getCols();
            int rows = _octaves[octave].getData(level).getRows();
            dim3 block( 32, HEIGHT );
            dim3 grid;
            grid.x  = grid_divide( cols, block.x );
            grid.y  = grid_divide( rows, block.y );

            Octave&      oct_obj = _octaves[octave];
            cudaStream_t oct_str = oct_obj.getStream(level);
            cudaEvent_t  oct_ev  = oct_obj.getEventGaussDone(level+1);

            cudaStreamWaitEvent( oct_str, oct_ev, 0 );

            find_extrema_in_dog_v6<HEIGHT>
                <<<grid,block,0,oct_str>>>
                ( _octaves[octave].getDogTexture( ),
                  level,
                  cols,
                  rows,
                  _levels,
                  _octaves[octave].getExtremaMgmtD( ),
                  _octaves[octave].getExtrema( level ) );
        }
    }

    cudaDeviceSynchronize();

    for( int octave=0; octave<_num_octaves; octave++ ) {
        for( int level=1; level<_levels-2; level++ ) {
            Octave&      oct_obj = _octaves[octave];
            cudaStream_t oct_str = oct_obj.getStream(level);

            fix_extrema_count_v6
                <<<1,1,0,oct_str>>>
                ( _octaves[octave].getExtremaMgmtD( ), level );
        }
    }
}

__host__
void Pyramid::find_extrema_v6( )
{
#define MANYLY(H) \
    find_extrema_v6_sub<H> ( );

    MANYLY(1)
    // MANYLY(2)
    // MANYLY(3)
    // MANYLY(4)
    // MANYLY(5)
    // MANYLY(6)
    // MANYLY(7)
    // MANYLY(8)
    // MANYLY(16)
    // fails // MANYLY(32)
}

} // namespace popart

