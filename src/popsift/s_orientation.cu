/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <math.h>
#include <stdio.h>
#include <inttypes.h>

#include "assist.h"
#include "sift_pyramid.h"
#include "sift_constants.h"
#include "s_gradiant.h"
#include "common/excl_blk_prefix_sum.h"
#include "common/warp_bitonic_sort.h"
#include "common/debug_macros.h"

using namespace popsift;
using namespace std;

/* Smoothing like VLFeat is the default mode.
 * If you choose to undefine it, you get the smoothing approach taken by OpenCV
 */
#define WITH_VLFEAT_SMOOTHING

__device__
inline float compute_angle( int bin, float hc, float hn, float hp )
{
    /* interpolate */
    float di = bin + 0.5f * (hn - hp) / (hc+hc-hn-hp);

    /* clamp */
    di = (di < 0) ? 
            (di + ORI_NBINS) : 
            ((di >= ORI_NBINS) ? (di - ORI_NBINS) : (di));

    float th = __fdividef( M_PI2 * di, ORI_NBINS ) - M_PI;
    // float th = ((M_PI2 * di) / ORI_NBINS);
    return th;
}

/*
 * Compute the keypoint orientations for each extremum
 * using 16 threads for each of them.
 * direct curve fitting approach
 */
template<int HEIGHT>
__global__
void ori_par( Extremum*     extremum,
              const int*    extrema_counter,
              Plane2D_float layer )
{
    uint32_t w   = layer.getWidth();
    uint32_t h   = layer.getHeight();

    const int extremum_index = blockIdx.x * blockDim.y + threadIdx.y;

    if( extremum_index >= *extrema_counter ) return; // a few trailing warps

    Extremum* ext = &extremum[extremum_index];

    __shared__ float hist   [HEIGHT][ORI_NBINS];
    __shared__ float sm_hist[HEIGHT][ORI_NBINS];

    for( int i = threadIdx.x; i < ORI_NBINS; i += blockDim.x )  hist[threadIdx.y][i] = 0.0f;

    /* keypoint fractional geometry */
    const float x    = ext->xpos;
    const float y    = ext->ypos;
    const float sig  = ext->sigma;

    /* orientation histogram radius */
    float  sigw = ORI_WINFACTOR * sig;
    int32_t rad  = (int)rintf((3.0f * sigw));

    float factor = __fdividef( -0.5f, (sigw * sigw) );
    int sq_thres  = rad * rad;

    int32_t xmin = max(1,     (int32_t)floor(x - rad));
    int32_t xmax = min(w - 2, (int32_t)floor(x + rad));
    int32_t ymin = max(1,     (int32_t)floor(y - rad));
    int32_t ymax = min(h - 2, (int32_t)floor(y + rad));

    int wx = xmax - xmin + 1;
    int hy = ymax - ymin + 1;
    int loops = wx * hy;

    for( int i = threadIdx.x; ::__any(i < loops); i += blockDim.x )
    {
        if( i < loops ) {
            int yy = i / wx + ymin;
            int xx = i % wx + xmin;

            float grad;
            float theta;
            get_gradiant( grad,
                        theta,
                        xx,
                        yy,
                        layer );

            float dx = xx - x;
            float dy = yy - y;

            int sq_dist  = dx * dx + dy * dy;
            if (sq_dist <= sq_thres) {
                float weight = grad * expf(sq_dist * factor);

                int bidx = (int)rintf( __fdividef( ORI_NBINS * (theta + M_PI), M_PI2 ) );
                // int bidx = (int)roundf( __fdividef( ORI_NBINS * (theta + M_PI), M_PI2 ) );

                if( bidx > ORI_NBINS ) {
                    printf("Crashing: bin %d theta %f :-)\n", bidx, theta);
                }

                bidx = (bidx == ORI_NBINS) ? 0 : bidx;

                atomicAdd( &hist[threadIdx.y][bidx], weight );
            }
        }
        __syncthreads();
    }

#ifdef WITH_VLFEAT_SMOOTHING
    for( int i=0; i<3; i++ ) {
        for( int bin = threadIdx.x; bin < ORI_NBINS; bin += blockDim.x ) {
            int prev = bin == 0 ? ORI_NBINS-1 : bin-1;
            int next = bin == ORI_NBINS-1 ? 0 : bin+1;
            sm_hist[threadIdx.y][bin] = ( hist[threadIdx.y][prev] + hist[threadIdx.y][bin] + hist[threadIdx.y][next] ) / 3.0f;
        }
        __syncthreads();
        for( int bin = threadIdx.x; bin < ORI_NBINS; bin += blockDim.x ) {
            int prev = bin == 0 ? ORI_NBINS-1 : bin-1;
            int next = bin == ORI_NBINS-1 ? 0 : bin+1;
            hist[threadIdx.y][bin] = ( sm_hist[threadIdx.y][prev] + sm_hist[threadIdx.y][bin] + sm_hist[threadIdx.y][next] ) / 3.0f;
        }
        __syncthreads();
    }
    for( int bin = threadIdx.x; bin < ORI_NBINS; bin += blockDim.x ) {
        sm_hist[threadIdx.y][bin] = hist[threadIdx.y][bin];
    }
    __syncthreads();
#else // not WITH_VLFEAT_SMOOTHING
    for( int bin = threadIdx.x; bin < ORI_NBINS; bin += blockDim.x ) {
        int prev2 = bin - 2;
        int prev1 = bin - 1;
        int next1 = bin + 1;
        int next2 = bin + 2;
        if( prev2 < 0 )          prev2 += ORI_NBINS;
        if( prev1 < 0 )          prev1 += ORI_NBINS;
        if( next1 >= ORI_NBINS ) next1 -= ORI_NBINS;
        if( next2 >= ORI_NBINS ) next2 -= ORI_NBINS;
        sm_hist[threadIdx.y][bin] = (   hist[threadIdx.y][prev2] + hist[threadIdx.y][next2]
                         + ( hist[threadIdx.y][prev1] + hist[threadIdx.y][next1] ) * 4.0f
                         +   hist[threadIdx.y][bin] * 6.0f ) / 16.0f;
    }
    __syncthreads();
#endif // not WITH_VLFEAT_SMOOTHING

    // sub-cell refinement of the histogram cell index, yielding the angle
    __shared__ float refined_angle[HEIGHT][64];
    __shared__ float yval         [HEIGHT][64];

    for( int bin = threadIdx.x; ::__any( bin < ORI_NBINS ); bin += blockDim.x ) {
        const int prev = bin == 0 ? ORI_NBINS-1 : bin-1;
        const int next = bin == ORI_NBINS-1 ? 0 : bin+1;

        bool predicate = ( bin < ORI_NBINS ) && ( sm_hist[threadIdx.y][bin] > max( sm_hist[threadIdx.y][prev], sm_hist[threadIdx.y][next] ) );

        const float num  = predicate ? 3.0f *   sm_hist[threadIdx.y][prev] - 4.0f * sm_hist[threadIdx.y][bin] + sm_hist[threadIdx.y][next]   : 0.0f;
        const float denB = predicate ? 2.0f * ( sm_hist[threadIdx.y][prev] - 2.0f * sm_hist[threadIdx.y][bin] + sm_hist[threadIdx.y][next] ) : 1.0f;

        const float newbin = __fdividef( num, denB );

        predicate   = ( predicate && newbin >= 0.0f && newbin <= 2.0f );

        refined_angle[threadIdx.y][bin] = predicate ? prev + newbin                               : -1;
        yval[threadIdx.y][bin]          = predicate ?  -(num*num) / (4.0f * denB) + sm_hist[threadIdx.y][prev] : -INFINITY;
    }

    int2 best_index = make_int2( threadIdx.x, threadIdx.x + 32 );

    BitonicSort::Warp32<float> sorter( yval[threadIdx.y] );
    sorter.sort64( best_index );
    __syncthreads();

    // All threads retrieve the yval of thread 0, the largest
    // of all yvals.
    const float best_val = yval[threadIdx.y][best_index.x];
    const float yval_ref = 0.8f * __shfl( best_val, 0 );
    const bool  valid    = ( best_val >= yval_ref );
    bool        written  = false;

    if( threadIdx.x < ORIENTATION_MAX_COUNT ) {
        if( valid ) {
            float chosen_bin = refined_angle[threadIdx.y][best_index.x];
            if( chosen_bin >= ORI_NBINS ) chosen_bin -= ORI_NBINS;
            // float th = __fdividef(M_PI2 * chosen_bin , ORI_NBINS) - M_PI;
            float th = ::fmaf( M_PI2 * chosen_bin, 1.0f/ORI_NBINS, - M_PI );
            ext->orientation[threadIdx.x] = th;
            written = true;
        }
    }

    int angles = __popc( __ballot( written ) );
    if( threadIdx.x == 0 ) {
        ext->num_ori = angles;
    }
}

class ExtremaRead
{
    const Extremum* const _oris;
public:
    inline __device__
    ExtremaRead( const Extremum* const d_oris ) : _oris( d_oris ) { }

    inline __device__
    int get( int n ) const { return _oris[n].num_ori; }
};

class ExtremaWrt
{
    Extremum* _oris;
public:
    inline __device__
    ExtremaWrt( Extremum* d_oris ) : _oris( d_oris ) { }

    inline __device__
    void set( int n, int value ) { _oris[n].idx_ori = value; }
};

class ExtremaTot
{
    int* _extrema_counter;
public:
    inline __device__
    ExtremaTot( int* extrema_counter ) : _extrema_counter( extrema_counter ) { }

    inline __device__
    void set( int value ) { *_extrema_counter = value; }
};

class ExtremaWrtMap
{
    int* _featvec_to_extrema_mapper;
    int  _max_feat;
public:
    inline __device__
    ExtremaWrtMap( int* featvec_to_extrema_mapper, int max_feat )
        : _featvec_to_extrema_mapper( featvec_to_extrema_mapper )
        , _max_feat( max_feat )
    { }

    inline __device__
    void set( int base, int num, int value )
    {
        int* baseptr = &_featvec_to_extrema_mapper[base];
        do {
            num--;
            if( base + num < _max_feat ) {
                baseptr[num] = value;
            }
        } while( num > 0 );
    }
};

__global__
void ori_prefix_sum( int*      extrema_counter,
                     int*      featvec_counter,
                     Extremum* extremum,
                     int*      d_feat_to_ext_map )
{
    ExtremaRead r( extremum );
    ExtremaWrt  w( extremum );
    ExtremaTot  t( featvec_counter );
    ExtremaWrtMap wrtm( d_feat_to_ext_map, d_consts.orientations );
    ExclusivePrefixSum::Block<ExtremaRead,ExtremaWrt,ExtremaTot,ExtremaWrtMap>( *extrema_counter, r, w, t, wrtm );

    __syncthreads();

    if( threadIdx.x == 0 && threadIdx.y == 0 ) {
        *featvec_counter = min( *featvec_counter, d_consts.orientations );
    }
}

#if __CUDA_ARCH__ > 350
__global__
void orientation_starter( Extremum*     extremum,
                          int*          extrema_counter,
                          int*          featvec_counter,
                          int*          d_feat_to_ext_map,
                          Plane2D_float layer )
{
    int num = *extrema_counter;

    if( num > 32 ) {
        dim3 block( 32, 2 );
        dim3 grid( grid_divide( num, 2 ) );

        ori_par<2>
            <<<grid,block>>>
            ( extremum,
              extrema_counter,
              layer );
    } else if( num > 0 ) {
        dim3 block( 32, 1 );
        dim3 grid( num );

        ori_par<1>
            <<<grid,block>>>
            ( extremum,
              extrema_counter,
              layer );
    }

    if( num > 0 ) {
        dim3 block( 32, 32 );
        dim3 grid( 1 );

        ori_prefix_sum
            <<<grid,block>>>
            ( extrema_counter,
              featvec_counter,
              extremum,
              d_feat_to_ext_map );
    }
}
#else // __CUDA_ARCH__ > 350
__global__
void orientation_starter( Extremum*     extremum,
                          int*          extrema_counter,
                          int*          featvec_counter,
                          int*          d_feat_to_ext_map,
                          Plane2D_float layer )
{
    printf( "Dynamic Parallelism requires a card with Compute Capability 3.5 or higher\n" );
}
#endif // __CUDA_ARCH__ > 350

__host__
void Pyramid::orientation( const Config& conf )
{
    if( conf.useDPOrientation() ) {
        // cerr << "Calling ori with dynamic parallelism" << endl;

        for( int octave=0; octave<_num_octaves; octave++ ) {
            Octave&      oct_obj = _octaves[octave];

            for( int level=1; level<_levels-2; level++ ) {
                cudaStream_t oct_str = oct_obj.getStream(level+2);

                orientation_starter
                    <<<1,1,0,oct_str>>>
                    ( oct_obj.getExtrema( level ),
                      oct_obj.getExtremaCtPtrD( level ),
                      oct_obj.getFeatVecCtPtrD( level ),
                      oct_obj.getFeatToExtMapD( level ),
                      oct_obj.getData( level ) );
            }
        }
    } else {
        // cerr << "Calling ori with -no- dynamic parallelism" << endl;

        for( int octave=0; octave<_num_octaves; octave++ ) {
            Octave&      oct_obj = _octaves[octave];

            for( int level=3; level<_levels; level++ ) {
                cudaStreamSynchronize( oct_obj.getStream(level) );
            }

            oct_obj.readExtremaCount( );
            cudaDeviceSynchronize( );

            for( int level=1; level<_levels-2; level++ ) {
                cudaStream_t oct_str = oct_obj.getStream(level+2);

                int num = oct_obj.getExtremaCountH(level);

                if( num > 0 ) {
                    dim3 block;
                    dim3 grid;

                    if( num > 32 ) {
                        block.x = 32;
                        block.y = 2;
                        grid.x  = grid_divide( num, 2 );

                        ori_par<2>
                            <<<grid,block,0,oct_str>>>
                            ( oct_obj.getExtrema( level ),
                              oct_obj.getExtremaCtPtrD( level ),
                              oct_obj.getData( level ) );
                    } else {
                        block.x = 32;
                        block.y = 1;
                        grid.x  = num;

                        ori_par<1>
                            <<<grid,block,0,oct_str>>>
                            ( oct_obj.getExtrema( level ),
                              oct_obj.getExtremaCtPtrD( level ),
                              oct_obj.getData( level ) );
                    }

                    block.x = 32;
                    block.y = 32;
                    grid.x  = 1;
                    ori_prefix_sum
                        <<<grid,block,0,oct_str>>>
                        ( oct_obj.getExtremaCtPtrD( level ),
                          oct_obj.getFeatVecCtPtrD( level ),
                          oct_obj.getExtrema( level ),
                          oct_obj.getFeatToExtMapD(level) );
                }
            }
        }
    }
}

