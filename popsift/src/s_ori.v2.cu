#include "s_ori.v2.h"
#include "s_gradiant.h"
#include "debug_macros.h"
#include "clamp.h"

#define ORI_V2_NUM_THREADS 16
#define NBINS_V2           36
#define WINFACTOR_V2       1.5f

/*
 * Compute the keypoint orientations for each extremum
 * using 16 threads for each of them.
 */
__global__
void compute_keypoint_orientations_v2( ExtremumCandidate* extremum,
                                       ExtremaMgmt*       mgmt_array,
                                       uint32_t           mgmt_level,
                                       Plane2D_float      layer )
{
    // uint32_t lid = threadIdx.x; // <- this is only the ID of the threads that are cooperating
    uint32_t w   = layer.getWidth();
    uint32_t h   = layer.getHeight();

    ExtremaMgmt* mgmt = &mgmt_array[mgmt_level];

    // if( blockIdx.x >= mgmt->counter ) return; // can never happen

    const int e_index = blockIdx.x;

    ExtremumCandidate* ext = &extremum[e_index];

    /* An orientation histogram is formed from the gradient
     * orientations of sample points within a region around
     * the keypoint. The orientation histogram has 36 bins
     * covering the 360 degree range of orientations. */
    float hist[NBINS_V2];
    for( int i = 0; i < NBINS_V2; i++ ) hist[i] = 0.0f;

    /* keypoint fractional geometry */
    float x    = ext->xpos;  /* keypoint's feature */
    float y    = ext->ypos;  /* keypoint's feature */
    float sig  = ext->sigma; /* keypoint's feature */

    /* orientation histogram radius */
    /* Each sample added to the histogram is weighted by its
     * gradient magnitude and by a Gaussian-weighted circular
     * window with a σ that is 1.5 times that of the scale of
     * the keypoint. */
    float sigw = WINFACTOR_V2 * sig;

    int   rad  = (int)rintf((3.0f * sigw)); // rintf is recommended for rounding

#ifndef __APPLE__
    assert( sigw != 0 );
#endif
    float factor  = -0.5f / (sigw * sigw);
    int sq_thres  = rad * rad;
    int32_t xmin = max( 1,     int32_t(x - rad));
    int32_t ymin = max( 1,     int32_t(y - rad));
    int32_t xmax = min( w - 2, int32_t(x + rad));
    int32_t ymax = min( h - 2, int32_t(y + rad));

    int wx = xmax - xmin + 1;
    int hy = ymax - ymin + 1;
    int loops = wx * hy;

    for( int i = threadIdx.x; i < loops; i+=ORI_V2_NUM_THREADS )
    {
        float grad;
        float theta;
#ifndef __APPLE__
        assert( wx + ymin != 0 );
#endif
        int yy = i / wx + ymin;
        int xx = i % wx + xmin;

        get_gradiant( grad,
                      theta,
                      xx,
                      yy,
                      layer );

        float dx = xx - x;
        float dy = yy - y;

        int sq_dist  = dx * dx + dy * dy;
        if (sq_dist <= sq_thres) {
            float   weight = grad * __expf( sq_dist * factor );
            int32_t bidx   = (int32_t)rintf(NBINS_V2 * (theta + M_PI) / M_PI2);
            bidx = (bidx < NBINS_V2) ? bidx : 0;

            hist[bidx] += weight;
        }
    }
    __syncthreads();

    /* reduction here */
    for (int i = 0; i < NBINS_V2; i++) {
        hist[i] += __shfl_down( hist[i], 8 );
        hist[i] += __shfl_down( hist[i], 4 );
        hist[i] += __shfl_down( hist[i], 2 );
        hist[i] += __shfl_down( hist[i], 1 );
        hist[i]  = __shfl( hist[i], 0 );
    }

    __syncthreads();

    /* smooth histogram */
    for( int bin=threadIdx.x; bin < NBINS_V2; bin+=ORI_V2_NUM_THREADS ) {
        int32_t bin_prev = (bin-1+NBINS_V2) % NBINS_V2;
        int32_t bin_next = (bin+1) % NBINS_V2;
        hist[bin] = 0.25f * hist[bin_prev] + 0.5f * hist[bin] + 0.25f * hist[bin_next];
    }
    __syncthreads();

    // sync is lost at the end of this loop, but __shfl auto-syncs
    for( int bin=0; bin < NBINS_V2; bin++ ) {
        // CAREFUL: THIS DOES PROBABLY NOT WORK !!!
        hist[bin] = __shfl( hist[bin], bin % ORI_V2_NUM_THREADS );
    }

    // all warps have the complete 1-smoothed history, not smoothe again
    __syncthreads();

    for( int bin=threadIdx.x; bin < NBINS_V2; bin+=ORI_V2_NUM_THREADS ) {
        int32_t bin_prev = (bin-1+NBINS_V2) % NBINS_V2;
        int32_t bin_next = (bin+1) % NBINS_V2;
        hist[bin] = 0.25f * hist[bin_prev] + 0.5f * hist[bin] + 0.25f * hist[bin_next];
    }

    // sync is lost at the end of this loop, and we should sync before fmaxf
    __syncthreads();

    float maxh = fmaxf( hist[threadIdx.x], hist[threadIdx.x+ORI_V2_NUM_THREADS] );
    if( threadIdx.x < NBINS_V2 % ORI_V2_NUM_THREADS ) {
        // this case is only for 32 (thread 0) and 33 (thread 1)
        maxh = fmaxf( maxh, hist[threadIdx.x+2*ORI_V2_NUM_THREADS] );
    }
    __syncthreads();
    // at this point, the 16 threads contain the maxima of 2 or 3 values, now reduce

    maxh = fmaxf( maxh, __shfl_down( maxh, 8 ) );
    maxh = fmaxf( maxh, __shfl_down( maxh, 4 ) );
    maxh = fmaxf( maxh, __shfl_down( maxh, 2 ) );
    maxh = fmaxf( maxh, __shfl_down( maxh, 1 ) );
    // at this point, thread 0 holds the maximum, share with all the others
    maxh = __shfl( maxh, 0 );

    /* new oris */
    float ang[3] = {NINF, NINF, NINF};
    int   angles = 0;

    bool  found_angle = false;

    /* find other peaks, boundary of 80% of max */
    for( int bin=threadIdx.x; ::__any(bin < NBINS_V2); bin+=ORI_V2_NUM_THREADS ) {
        if( bin < NBINS_V2 ) {
            float hc = hist[bin];
            float hn = hist[(bin+1) % NBINS_V2];
            float hp = hist[(bin-1+NBINS_V2) % NBINS_V2];

            /* Find if a peak.
             * Note that the condition may imply that we find no peak at all.
             */
            if (hc >= (0.8f * maxh) && hc > hn && hc > hp) {

                /* found another angle ! */
                found_angle = true;
    
                /* interpolate */
                float di = bin + 0.5f * (hn - hp) / (hc+hc-hn-hp);
            
                /* clamp */
                di = (di < 0) ? (di + NBINS_V2)
                            : ((di >= NBINS_V2) ? (di - NBINS_V2)
                                            : (di));
            
                ang[angles] = ((M_PI2 * di) / NBINS_V2) - M_PI;

                angles += 1;
            }
        }
        __syncthreads();
    }

    // perform a prefix sum, provides a relative index to all threads
    uint32_t incl_prefix_sum = angles;                  // o = 1 0 2 0 1 0 0 1
    incl_prefix_sum += __shfl_up( incl_prefix_sum, 1 ); // 1 = 1 1 2 2 1 1 0 1
    incl_prefix_sum += __shfl_up( incl_prefix_sum, 2 ); // 2 = 1 1 3 3 3 3 1 2
    incl_prefix_sum += __shfl_up( incl_prefix_sum, 4 ); // 4 = 1 1 3 3 4 4 4 5
    incl_prefix_sum += __shfl_up( incl_prefix_sum, 8 );
    const uint32_t total_sum = __shfl( incl_prefix_sum, 15 );

    if( total_sum == 0 ) {
    } else if( total_sum == 1 ) {
        if( found_angle ) {
            ext->xpos             = x;
            ext->ypos             = y;
            ext->sigma            = sig;
            ext->angle_from_bemap = ang[0];
        }
    } else {
        uint32_t write_index;
        if( threadIdx.x == 0 ) {
            // adding sum-1 because the first slot is already known
            write_index = atomicAdd( &mgmt->counter, total_sum-1 );
        }
        // tell everybody about the base
        write_index = __shfl( write_index, 0 );
        // all 16 threads are now in sync

        if( found_angle ) {
            // Only threads that have found one or more angles are going to write
            const uint32_t excl_prefix_sum = incl_prefix_sum - 1;
            int            i = 0;
            int            off = 0;

            if( excl_prefix_sum == 0 ) {
                ext->angle_from_bemap = ang[i++];
            }

            while( i<angles ) {
                ExtremumCandidate* ext = &extremum[ write_index + excl_prefix_sum + off ];
                ext->xpos             = x;
                ext->ypos             = y;
                ext->sigma            = sig;
                ext->angle_from_bemap = ang[i];
                i++;
                off++;
            }
        }
    }
}

/*************************************************************
 * V4: host side
 *************************************************************/
__host__
void Pyramid::orientation_v2( )
{
    // cerr << "Enter " << __FUNCTION__ << endl;

    _keep_time_orient_v2.start();
    for( int octave=0; octave<_num_octaves; octave++ ) {
        cerr << __FILE__ << ":" << __LINE__ << " read extrema count" << endl;
        _octaves[octave].readExtremaCount( );
        cudaDeviceSynchronize( );
        for( int level=1; level<_levels-1; level++ ) {
            dim3 block;
            dim3 grid;
            // grid.x  = _octaves[octave].getExtremaMgmtH(level)->max1;
            grid.x  = _octaves[octave].getExtremaMgmtH(level)->counter;
            block.x = ORI_V2_NUM_THREADS;
            if( grid.x != 0 ) {
#if 1
                cout << "computing keypoint orientation in octave "
                     << octave << " level " << level
                     << " for " << grid.x << " blocks a " << block.x << " threads" << endl;
#endif

                cudaDeviceSynchronize();
                POP_CHK;

                compute_keypoint_orientations_v2
                    <<<grid,block>>>
                    ( _octaves[octave].getExtrema( level ),
                      _octaves[octave].getExtremaMgmtD( ),
                      level,
                      _octaves[octave].getData( level ) );

                cudaDeviceSynchronize();
                POP_CHK;
            }
        }
    }
    _keep_time_orient_v2.stop();

    // cerr << "Leave " << __FUNCTION__ << endl;
}

