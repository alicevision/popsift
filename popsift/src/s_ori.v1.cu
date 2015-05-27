#include "s_ori.v1.h"
#include "s_gradiant.h"

#define ORI_V1_NUM_THREADS 16
#define NBINS_V1           36
#define WINFACTOR_V1       1.5F

/*************************************************************
 * V1: device side
 *************************************************************/

/*
 * Compute the keypoint orientations for each extremum
 * using 16 threads for each of them.
 */
__global__
void compute_keypoint_orientations_v1( ExtremumCandidate* extremum,
                                       ExtremaMgmt*       mgmt_array,
                                       uint32_t           mgmt_level,
                                       const float*       layer,
                                       int                layer_pitch,
                                       int                layer_height )
{
    uint32_t w   = layer_pitch;
    uint32_t h   = layer_height;

    ExtremaMgmt* mgmt = &mgmt_array[mgmt_level];

    if( threadIdx.y >= mgmt->counter ) return;

    ExtremumCandidate* ext = &extremum[threadIdx.y];

    float hist[NBINS_V1];
    for (int i = 0; i < NBINS_V1; i++) hist[i] = 0.0f;

    /* keypoint fractional geometry */
    float x    = ext->xpos;
    float y    = ext->ypos;
    float sig  = ext->sigma;

    /* orientation histogram radius */
    float  sigw = WINFACTOR_V1 * sig;
    int32_t rad  = (int)rintf((3.0f * sigw));

    float factor = -0.5f / (sigw * sigw);
    int sq_thres  = rad * rad;
    int32_t xmin = max(1,     (int32_t)floor(x - rad));
    int32_t xmax = min(w - 2, (int32_t)floor(x + rad));
    int32_t ymin = max(1,     (int32_t)floor(y - rad));
    int32_t ymax = min(h - 2, (int32_t)floor(y + rad));

    int wx = xmax - xmin + 1;
    int hy = ymax - ymin + 1;
    int loops = wx * hy;

    for(int i = threadIdx.x; i < loops; i+=ORI_V1_NUM_THREADS)
    {
        int yy = i / wx + ymin;
        int xx = i % wx + xmin;

        float grad;
        float theta;
        get_gradiant( grad,
                      theta,
                      xx,
                      yy,
                      layer,
                      layer_pitch,
                      layer_height );

        float dx = xx - x;
        float dy = yy - y;

        int sq_dist  = dx * dx + dy * dy;
        if (sq_dist <= sq_thres) {
            float weight = grad * exp(sq_dist * factor);
            int bidx = (int)rintf(NBINS_V1 * (theta + M_PI) / M_PI2);
            bidx = (bidx < NBINS_V1) ? bidx : 0;
            hist[bidx] += weight;
        }
    }

    /* reduction here */
    for (int i = 0; i < NBINS_V1; i++) {
        hist[i] += __shfl_down( hist[i], 8 );
        hist[i] += __shfl_down( hist[i], 4 );
        hist[i] += __shfl_down( hist[i], 2 );
        hist[i] += __shfl_down( hist[i], 1 );
        hist[i]  = __shfl( hist[i], 0 );
    }

    if( threadIdx.x != 0 ) return;

    /* new oris */
    float ang[2] = {NINF, NINF};

    /* smooth histogram */
    for( int iter = 0; iter < 2; iter++ ) {
        float first = hist[0];
        float prev  = hist[(NBINS_V1-1)];

        int bin;
        for (bin = 0; bin < NBINS_V1 - 1; bin++) {
            float temp = hist[bin];
            hist[bin] = 0.25f * prev + 0.5f * hist[bin] + 0.25f * hist[bin+1];
            prev = temp;
        }
        hist[bin] = 0.25f * prev + 0.5f * hist[bin] + 0.25f * first;
    }

    /* find histogram maximum */
    float maxh = NINF;
    for (int bin = 0; bin < NBINS_V1; bin++) {
        maxh = fmaxf(maxh, hist[bin]);
    }

    /* find other peaks, boundary of 80% of max */
    int nangles = 0;
    for (int bin = 0; bin < NBINS_V1; bin++) {
        float hc = hist[bin];
        float hn = hist[((bin+1+NBINS_V1)%NBINS_V1)];
        float hp = hist[((bin-1+NBINS_V1)%NBINS_V1)];

        /* find if a peak */
        if (hc >= (0.8f * maxh) && hc > hn && hc > hp) {
    
            /* interpolate */
            float di = bin + 0.5f * (hn - hp) / (hc+hc-hn-hp);
            
            /* clamp */
            di = (di < 0) ? 
                (di + NBINS_V1) : 
                ((di >= NBINS_V1) ? (di - NBINS_V1) : (di));
            
            double th = ((M_PI2 * di) / NBINS_V1) - M_PI;
            ang[nangles++] = th;
            if(nangles >= 2) break;
        }
    }

    if( nangles < 1 ) return;

    // ext->xpos             = x;
    // ext->ypos             = y;
    // ext->sigma            = sig;
    ext->angle_from_bemap = ang[0];

    for( int bin=1; bin<nangles; bin++ ) {
        int idx = atomicAdd( &mgmt->counter, 1 );
        if( idx < mgmt->max2 ) {
            ext = &extremum[idx];
            ext->xpos             = x;
            ext->ypos             = y;
            ext->sigma            = sig;
            ext->angle_from_bemap = ang[bin];
        }
    }
}

/*************************************************************
 * V4: host side
 *************************************************************/
__host__
void Pyramid::orientation_v1( )
{
    _keep_time_orient_v1.start();
    for( int octave=0; octave<_octaves; octave++ ) {
        _layers[octave].readExtremaCount( _stream );
        cudaStreamSynchronize( _stream );
        for( int level=1; level<_levels-1; level++ ) {
            dim3 block;
            dim3 grid;
            // grid.x  = _layers[octave].getExtremaMgmtH(level)->max1;
            grid.x  = _layers[octave].getExtremaMgmtH(level)->counter;
            block.x = ORI_V1_NUM_THREADS;
            if( grid.x != 0 ) {
                compute_keypoint_orientations_v1
                    <<<grid,block,0,_stream>>>
                    ( _layers[octave].getExtrema( level ),
                      _layers[octave].getExtremaMgmtD( ),
                      level,
                      _layers[octave].getData( level ),
                      _layers[octave].getPitch(),
                      _layers[octave].getHeight() );
            }
        }
    }
    _keep_time_orient_v1.stop();
}

