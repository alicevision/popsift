#include <stdio.h>

#include "s_ori.v1.h"
#include "debug_macros.h"
#include "s_gradiant.h"
#include <stdio.h>
#define ORI_V1_NUM_THREADS 16
#define NBINS_V1           36
#define WINFACTOR_V1       1.5F
#if __CUDA_ARCH__ >= 350
#define USE_DYNAMIC_PARALLELISM
#else
#undef USE_DYNAMIC_PARALLELISM
#endif

/*************************************************************
 * V1: device side
 *************************************************************/

__device__
inline float compute_angle( int bin, float hc, float hn, float hp )
{
    /* interpolate */
    float di = bin + 0.5f * (hn - hp) / (hc+hc-hn-hp);

    /* clamp */
    di = (di < 0) ? 
            (di + NBINS_V1) : 
            ((di >= NBINS_V1) ? (di - NBINS_V1) : (di));

    float th = ((M_PI2 * di) / NBINS_V1) - M_PI;
    // float th = ((M_PI2 * di) / NBINS_V1);
    return th;
}

/*
 * Compute the keypoint orientations for each extremum
 * using 16 threads for each of them.
 */
__global__
void compute_keypoint_orientations_v1( ExtremumCandidate* extremum,
                                       ExtremaMgmt*       mgmt_array,
                                       uint32_t           mgmt_level,
                                       Plane2D_float      layer )
{
    uint32_t w   = layer.getWidth();
    uint32_t h   = layer.getHeight();

    ExtremaMgmt* mgmt = &mgmt_array[mgmt_level];

    // if( threadIdx.y >= mgmt->counter ) return;

    ExtremumCandidate* ext = &extremum[blockIdx.x];

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
                      layer );

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


#if 0
    /* new oris */
    // float ang[2] = {NINF, NINF};

    /* smooth histogram */
    //0->1
    for (int iter = 0; iter < 2; iter++) {
        float first = hist[0];//needed for last bin as it's last bins 'next'

        float prev1 = hist[(threadIdx.x     - 1 + ORI_V1_NUM_THREADS) % ORI_V1_NUM_THREADS];
        float prev2 = hist[(threadIdx.x * 2 - 1 + ORI_V1_NUM_THREADS) % ORI_V1_NUM_THREADS];

        //we need to handle 36 values with 16 threads, rip.
        float tmp1_curr = hist[threadIdx.x];    //0->15
        float tmp1_next = hist[threadIdx.x+1];  //1->16

        float tmp2_curr = hist[threadIdx.x*2];  //16->31
        float tmp2_next = hist[threadIdx.x*2+1];//17->32

        hist[threadIdx.x  ] = 0.25f * prev1 + 0.5f * tmp1_curr + 0.25f * tmp1_next;
        hist[threadIdx.x*2] = 0.25f * prev2 + 0.5f * tmp2_curr + 0.25f * tmp2_next;

        //printf("(%d): bin %d: %f\n", threadIdx.x, hist[threadIdx.x]);
        //printf("(%d): bin %d: %f\n", threadIdx.x*2, hist[threadIdx.x*2]);

        //0->31 done
        //missing 32->35 (4 bins)
        if(threadIdx.x < 3){
	//0+32
	//1+32
	//2+32 34
            //bin 32->34
            float prev = hist[(threadIdx.x+32 - 1 + ORI_V1_NUM_THREADS) % ORI_V1_NUM_THREADS];
            float curr = hist[threadIdx.x]+32; //32->34
            float next = hist[threadIdx.x+33]; //33->35
            hist[threadIdx.x +32] = 0.25f * prev + 0.5f * curr + 0.25f * next;
            //printf("(%d): bin %d: %f\n", threadIdx.x+32, hist[threadIdx.x+32]);
        }else if(threadIdx.x == 3){
            //bin 35
            hist[threadIdx.x +32] = 0.25f * hist[threadIdx.x +32 -1]
                                   + 0.5f * hist[threadIdx.x +32] + 0.25f * first;
            //printf("(%d): bin %d: %f\n", threadIdx.x+32, hist[threadIdx.x+32]);
        }
    }
#else
    
        if(threadIdx.x != 0) return;
        for (int iter = 0; iter < 2; iter++) {
            float first = hist[0];
            float prev = hist[(NBINS_V1 - 1)];

            int bin;
            //0,35
            for (bin = 0; bin < NBINS_V1 - 1; bin++) {
                float temp = hist[bin];
                hist[bin] = 0.25f * prev + 0.5f * hist[bin] + 0.25f * hist[bin + 1];
                prev = temp;
            }

            hist[bin] = 0.25f * prev + 0.5f * hist[bin] + 0.25f * first;
            //z vprintf("val: %f, indx: %d\n", hist[bin], bin);
        }
	
        /* find histogram maximum */
        float maxh = NINF;
        int binh = 0;
        for (int bin = 0; bin < NBINS_V1; bin++) {
            // maxh = fmaxf(maxh, hist[bin]);
            if (hist[bin] > maxh) {
                maxh = hist[bin];
                binh = bin;
            }
        }

        {
            float hc = hist[binh];
            float hn = hist[((binh + 1 + NBINS_V1) % NBINS_V1)];
            float hp = hist[((binh - 1 + NBINS_V1) % NBINS_V1)];
            float th = compute_angle(binh, hc, hn, hp);

            ext->angle_from_bemap = th;
        }


        /* find other peaks, boundary of 80% of max */
        int nangles = 1;

        for (int numloops = 1; numloops < NBINS_V1; numloops++) {
            int bin = (binh + numloops) % NBINS_V1;

            float hc = hist[bin];
            float hn = hist[((bin + 1 + NBINS_V1) % NBINS_V1)];
            float hp = hist[((bin - 1 + NBINS_V1) % NBINS_V1)];

            /* find if a peak */
            if (hc >= (0.8f * maxh) && hc > hn && hc > hp) {
                int idx = atomicAdd(&mgmt->counter, 1);
                if (idx >= mgmt->max2) break;

                float th = compute_angle(bin, hc, hn, hp);

                ext = &extremum[idx];
                ext->xpos = x;
                ext->ypos = y;
                ext->sigma = sig;
                ext->angle_from_bemap = th;

                nangles++;
                if (nangles > 2) break;
            }
        }
#endif

}

/*************************************************************
 * V4: host side
 *************************************************************/
#ifdef USE_DYNAMIC_PARALLELISM

__global__
void orientation_starter_v1( ExtremumCandidate* extremum,
                             ExtremaMgmt*       mgmt_array,
                             uint32_t           mgmt_level,
                             Plane2D_float      layer )
{
    ExtremaMgmt* mgmt = &mgmt_array[mgmt_level];

    dim3 block;
    dim3 grid;
    grid.x  = mgmt->counter;
    block.x = ORI_V1_NUM_THREADS;

    if( grid.x != 0 ) {
        compute_keypoint_orientations_v1
            <<<grid,block>>>
            ( extremum,
              mgmt_array,
              mgmt_level,
              layer );
    }
}

__host__
void Pyramid::orientation_v1( )
{
    cerr << "Enter " << __FUNCTION__ << endl;

    _keep_time_orient_v1.start();
    for( int octave=0; octave<_num_octaves; octave++ ) {
        for( int level=1; level<_levels-1; level++ ) {
            orientation_starter_v1
                <<<1,1>>>
                ( _octaves[octave].getExtrema( level ),
                  _octaves[octave].getExtremaMgmtD( ),
                  level,
                  _octaves[octave].getData( level ) );
        }
    }
    _keep_time_orient_v1.stop();

    cerr << "Leave " << __FUNCTION__ << endl;
}

#else // not USE_DYNAMIC_PARALLELISM

__global__
void orientation_starter_v1( ExtremumCandidate*,
                             ExtremaMgmt*,
                             uint32_t,
                             Plane2D_float )
{
    /* dummy to make the linker happy */
}

__host__
void Pyramid::orientation_v1( )
{
    cerr << "Enter " << __FUNCTION__ << endl;

    _keep_time_orient_v1.start();
    for( int octave=0; octave<_num_octaves; octave++ ) {
        _octaves[octave].readExtremaCount( );
        cudaDeviceSynchronize( );
        for( int level=1; level<_levels-1; level++ ) {
            dim3 block;
            dim3 grid;
            // grid.x  = _octaves[octave].getExtremaMgmtH(level)->max1;
            grid.x  = _octaves[octave].getExtremaMgmtH(level)->counter;
            block.x = ORI_V1_NUM_THREADS;
            if( grid.x != 0 ) {
                compute_keypoint_orientations_v1
                    <<<grid,block>>>
                    ( _octaves[octave].getExtrema( level ),
                      _octaves[octave].getExtremaMgmtD( ),
                      level,
                      _octaves[octave].getData( level ) );
            }
        }
    }
    _keep_time_orient_v1.stop();

    cerr << "Leave " << __FUNCTION__ << endl;
}
#endif // not USE_DYNAMIC_PARALLELISM

