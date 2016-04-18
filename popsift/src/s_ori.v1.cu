#include "sift_pyramid.h"
#include "s_gradiant.h"
#include "debug_macros.h"

#include <stdio.h>
#include <inttypes.h>

#define ORI_V1_NUM_THREADS 16
#define NBINS_V1           36
#define WINFACTOR_V1       1.5F

using namespace popart;

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
void compute_keypoint_orientations_v1( Extremum* extremum,
                                       ExtremaMgmt*       mgmt_array,
                                       uint32_t           mgmt_level,
                                       Plane2D_float      layer )
{
    uint32_t w   = layer.getWidth();
    uint32_t h   = layer.getHeight();

    ExtremaMgmt* mgmt = &mgmt_array[mgmt_level];

    // if( threadIdx.y >= mgmt->getCounter() ) return;

    Extremum* ext = &extremum[blockIdx.x];

    float hist[NBINS_V1];
    for (int i = 0; i < NBINS_V1; i++) hist[i] = 0.0f;

    /* keypoint fractional geometry */
    const float x    = ext->xpos;
    const float y    = ext->ypos;
    const float sig  = ext->sigma;

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

            // int bidx = (int)rintf(NBINS_V1 * (theta + M_PI) / M_PI2);
            int bidx = (int)roundf(NBINS_V1 * (theta + M_PI) / M_PI2);
            // int bidx = (int)roundf(NBINS_V1 * (theta + M_PI) / M_PI2 - 0.5f);

            if( bidx > NBINS_V1 ) {
                printf("Crashing: bin %d theta %f :-)\n", bidx, theta);
            }

            bidx = (bidx == NBINS_V1) ? 0 : bidx;

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


#define OLD_ORIENTATION

        if(threadIdx.x != 0) return;

    // for (int bin = 0; bin < NBINS_V1; bin++) {
        // printf( "%f %f %d %f\n", x, y, bin, hist[bin] );
    // }

#ifdef OLD_ORIENTATION
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

            if( isnan(th) ) {
                printf("NAN value in compute_angle\n");
            }

            ext->orientation = th;
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
                int idx = mgmt->atomicAddCounter( 1 );
                if( idx >= mgmt->getOrientationMax() ) break;

                float th = compute_angle(bin, hc, hn, hp);

                ext = &extremum[idx];
                ext->xpos = x;
                ext->ypos = y;
                ext->sigma = sig;
                ext->orientation = th;

                nangles++;
                if (nangles > 2) break;
            }
        }
#else // not OLD_ORIENTATION
        float xcoord[NBINS_V1];
        float yval[NBINS_V1];

        int   maxbin = 0;
        float y_max = 0;
        for(int bin = 0; bin < NBINS_V1; bin++) {
            int prev = bin - 1;
            if( prev < 0 ) prev = NBINS_V1 - 1;
            int next = bin + 1;
            if( next == NBINS_V1 ) next = 0;

            if( hist[bin] > max( hist[prev], hist[next] ) ) {
                const float num = 3.0f * hist[prev] - 4.0f * hist[bin] + hist[next];
                const float denB = 2.0f * ( hist[prev] - 2.0f * hist[bin] + hist[next] );
                float newbin = num / denB; // * M_PI/18.0f; // * 10.0f;
                if( newbin >= 0 && newbin <= 2 ) {
                    xcoord[bin] = prev + newbin;
                    yval[bin]   = -(num*num) / (4.0f * denB) + hist[prev];

                    if( yval[bin] > y_max ) {
                        y_max = yval[bin];
                        maxbin = bin;
                    }
                }
            }
        }
        float th = ((M_PI2 * xcoord[maxbin]) / NBINS_V1) - M_PI;

        ext->orientation = th;
#endif // not OLD_ORIENTATION
}

/*************************************************************
 * V4: host side
 *************************************************************/
#ifdef USE_DYNAMIC_PARALLELISM // defined in_s_pyramid.h

__global__
void orientation_starter_v1( Extremum* extremum,
                             ExtremaMgmt*       mgmt_array,
                             uint32_t           mgmt_level,
                             Plane2D_float      layer )
{
    ExtremaMgmt* mgmt = &mgmt_array[mgmt_level];

    dim3 block;
    dim3 grid;
    grid.x  = mgmt->getCounter();
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
    for( int octave=0; octave<_num_octaves; octave++ ) {
        Octave&      oct_obj = _octaves[octave];

        for( int level=1; level<_levels-1; level++ ) {
            cudaStream_t oct_str = oct_obj.getStream(level);

            orientation_starter_v1
                <<<1,1,0,oct_str>>>
                ( oct_obj.getExtrema( level ),
                  oct_obj.getExtremaMgmtD( ),
                  level,
                  oct_obj.getData( level ) );
        }
    }
}

#else // not USE_DYNAMIC_PARALLELISM

__global__
void orientation_starter_v1( Extremum*,
                             ExtremaMgmt*,
                             uint32_t,
                             Plane2D_float )
{
    /* dummy to make the linker happy */
}

__host__
void Pyramid::orientation_v1( )
{
    for( int octave=0; octave<_num_octaves; octave++ ) {
        Octave&      oct_obj = _octaves[octave];

        for( int level=1; level<_levels-1; level++ ) {
            cudaStreamSynchronize( oct_obj.getStream(level) );
        }

        oct_obj.readExtremaCount( );
        cudaDeviceSynchronize( );

        for( int level=1; level<_levels-1; level++ ) {
            cudaStream_t oct_str = oct_obj.getStream(level);

            dim3 block;
            dim3 grid;
            // grid.x  = _octaves[octave].getExtremaMgmtH(level)->max1;
            grid.x  = oct_obj.getExtremaMgmtH(level)->getCounter();
            block.x = ORI_V1_NUM_THREADS;
            if( grid.x != 0 ) {
                compute_keypoint_orientations_v1
                    <<<grid,block,0,oct_str>>>
                    ( oct_obj.getExtrema( level ),
                      oct_obj.getExtremaMgmtD( ),
                      level,
                      oct_obj.getData( level ) );
            }
        }
    }
}
#endif // not USE_DYNAMIC_PARALLELISM

