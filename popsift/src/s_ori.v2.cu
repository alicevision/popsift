#include "s_ori.v2.h"
#include "s_gradiant.h"
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

    int idy = clamp( threadIdx.y, mgmt->counter );

    ExtremumCandidate* ext = &extremum[idy];

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
        if (sq_dist > sq_thres) continue;

        float   weight = grad * __expf( sq_dist * factor );
        int32_t bidx   = (int32_t)rintf(NBINS_V2 * (theta + M_PI) / M_PI2);
        bidx = (bidx < NBINS_V2) ? bidx : 0;

        hist[bidx] += weight;
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

    /* smooth histogram */

    for( int bin=threadIdx.x; bin < NBINS_V2; bin+=ORI_V2_NUM_THREADS ) {
        int32_t bin_prev = (bin-1+NBINS_V2) % NBINS_V2;
        int32_t bin_next = (bin+1) % NBINS_V2;
        hist[bin] = 0.25f * hist[bin_prev] + 0.5f * hist[bin] + 0.25f * hist[bin_next];
    }
    __syncthreads();

    // sync is lost at the end of this loop, but __shfl auto-syncs
    for( int bin=0; bin < NBINS_V2; bin++ ) {
        hist[bin] = __shfl( hist[bin], bin % ORI_V2_NUM_THREADS );
    }
    // all warps have the complete 1-smoothed history, not smoothe again

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

    if( idy != threadIdx.y ) {
        found_angle = false;
    }

    uint32_t ct = __popc( __ballot( found_angle ) );
    if( ct == 0 ) {
        if( threadIdx.x == 0 ) {
            ext->not_a_keypoint = 1;
        }
    } else if( ct == 1 ) {
        if( found_angle ) {
            ext->xpos             = x;
            ext->ypos             = y;
            ext->sigma            = sig;
            ext->angle_from_bemap = ang[0];
        }
    } else {
        uint32_t sum = ct;

        // perform a prefix sum, provides a relative index to all threads
        sum += __shfl_up( sum, 1 );
        sum += __shfl_up( sum, 2 );
        sum += __shfl_up( sum, 4 );
        sum += __shfl_up( sum, 8 );

        uint32_t write_index;
        if( threadIdx.x == 31 ) {
            // adding sum-1 because the first slot is already known
            write_index = atomicAdd( &mgmt->counter, sum-1 );
        }
        // tell everybody about the base
        write_index = __shfl( write_index, 31 );
        int i = 0;
        sum -= ct; // this threads's base index
        if( ct > 0 && sum == 0 ) {
            /* this thread should use the existing extremum */
            // ext->x                = x;
            // ext->y                = y;
            // ext->sigma            = sig;
            ext->angle_from_bemap = ang[i++];
            ct--;
        }
        if( write_index + sum -1 < mgmt->max2 ) {
            for( ; i<ct; i++ ) {
                // reduce 1 because the lowest index is written to existing keypoint
                ExtremumCandidate* ext = &extremum[ write_index + sum - 1 ];
                ext->xpos             = x;
                ext->ypos             = y;
                ext->sigma            = sig;
                ext->angle_from_bemap = ang[i++];
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
#if 0
    _keep_time_orient_v2.start();
    for( int octave=0; octave<_num_octaves; octave++ ) {
        _octaves[octave].readExtremaCount( );
        cudaDeviceSynchronize( );
        for( int level=1; level<_levels-1; level++ ) {
            dim3 block;
            dim3 grid;
            // grid.x  = _octaves[octave].getExtremaMgmtH(level)->max1;
            grid.x  = _octaves[octave].getExtremaMgmtH(level)->counter;
            block.x = ORI_V2_NUM_THREADS;
            if( grid.x != 0 ) {
#if 0
                cout << "computing keypoint orientation in octave "
                     << octave << " level " << level
                     << " for " << grid.x << " blocks a " << block.x << " threads" << endl;
#endif

                compute_keypoint_orientations_v2
                    <<<grid,block>>>
                    ( _octaves[octave].getExtrema( level ),
                      _octaves[octave].getExtremaMgmtD( ),
                      level,
                      _octaves[octave].getData( level ) );
            }
        }
    }
    _keep_time_orient_v2.stop();
#endif
}

