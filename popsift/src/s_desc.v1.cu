#include "s_pyramid.h"
#include "s_gradiant.h"

#define DESCR_BINS_V1        8
#define MAGNIFY_V1           3.0f
#define DESCR_V1_NUM_THREADS 32
// #define DESCR_V1_NUM_THREADS 16
// #define DESCR_V1_NUM_THREADS 8
#define RPI_V1               (4.0f / M_PI)
#define EPS_V1               1E-15

#define USE_ROOT_SIFT

/*************************************************************
 * V1: device side
 *************************************************************/

using namespace popart;

__global__
void keypoint_descriptors( ExtremumCandidate* cand,
                           Descriptor*        descs,
                           Plane2D_float      layer )
{
    const uint32_t width  = layer.getWidth();
    const uint32_t height = layer.getHeight();

    // int bidx = blockIdx.x & 0xf; // lower 4 bits of block ID
    const int ix   = threadIdx.y; // bidx & 0x3;       // lower 2 bits of block ID
    const int iy   = threadIdx.z; // bidx >> 2;        // next lowest 2 bits of block ID

    ExtremumCandidate* ext = &cand[blockIdx.x];

    const float x    = ext->xpos;
    const float y    = ext->ypos;
    const float sig  = ext->sigma;
    const float ang  = ext->orientation;
    const float SBP  = fabs(MAGNIFY_V1 * sig);

    const float cos_t = cosf(ang);
    const float sin_t = sinf(ang);

    const float csbp  = cos_t * SBP;
    const float ssbp  = sin_t * SBP;
    const float crsbp = cos_t / SBP;
    const float srsbp = sin_t / SBP;

    const float offsetptx = ix - 1.5f;
    const float offsetpty = iy - 1.5f;
    const float ptx = csbp * offsetptx - ssbp * offsetpty + x;
    const float pty = csbp * offsetpty + ssbp * offsetptx + y;

    const float bsz = fabs(csbp) + fabs(ssbp);

    const int32_t xmin = max(1,          (int32_t)floor(ptx - bsz));
    const int32_t ymin = max(1,          (int32_t)floor(pty - bsz));
    const int32_t xmax = min(width - 2,  (int32_t)floor(ptx + bsz));
    const int32_t ymax = min(height - 2, (int32_t)floor(pty + bsz));

    const int32_t wx = xmax - xmin + 1;
    const int32_t hy = ymax - ymin + 1;
    const int32_t loops = wx * hy;

    float dpt[9];
    for (int i = 0; i < 9; i++) dpt[i] = 0.0f;

    for(int i = threadIdx.x; i < loops; i+=DESCR_V1_NUM_THREADS)
    {
        const int ii = i / wx + ymin;
        const int jj = i % wx + xmin;     

        const float dx = jj - ptx;
        const float dy = ii - pty;
        const float nx = crsbp * dx + srsbp * dy;
        const float ny = crsbp * dy - srsbp * dx;
        const float nxn = fabs(nx);
        const float nyn = fabs(ny);
        if (nxn < 1.0f && nyn < 1.0f) {
            float mod;
            float th;
            get_gradiant( mod, th,
                          jj, ii,
                          layer );

            const float dnx = nx + offsetptx;
            const float dny = ny + offsetpty;
            const float ww  = __expf(-0.125f * (dnx*dnx + dny*dny));
            const float wx  = 1.0f - nxn;
            const float wy  = 1.0f - nyn;
            const float wgt = ww * wx * wy * mod;

            th -= ang;
            while (th < 0.0f) th += M_PI2;
            while (th >= M_PI2) th -= M_PI2;

            const float   tth  = th * RPI_V1;
            const int32_t fo0  = (int32_t)floor(tth);
            const float   do0  = tth - fo0;             
            const float   wgt1 = 1.0f - do0;
            const float   wgt2 = do0;

            int fo  = fo0 % DESCR_BINS_V1;
            if(fo < 8) {
                dpt[fo]   += (wgt1*wgt);
                dpt[fo+1] += (wgt2*wgt);
            }
        }
    }
    __syncthreads();

    dpt[0] += dpt[8];

    /* reduction here */
    for (int i = 0; i < 8; i++) {
#if DESCR_V1_NUM_THREADS==32
        dpt[i] += __shfl_down( dpt[i], 16 );
#endif
#if DESCR_V1_NUM_THREADS==32 || DESCR_V1_NUM_THREADS==16
        dpt[i] += __shfl_down( dpt[i], 8 );
#else
#endif
        dpt[i] += __shfl_down( dpt[i], 4 );
        dpt[i] += __shfl_down( dpt[i], 2 );
        dpt[i] += __shfl_down( dpt[i], 1 );
        dpt[i]  = __shfl     ( dpt[i], 0 );
    }

    // int hid    = blockIdx.x % 16;
    // int offset = hid*8;
    uint32_t offset = ( ( threadIdx.z << 2 ) + threadIdx.y ) * 8;

    Descriptor* desc = &descs[blockIdx.x];

    if( threadIdx.x == 0 ) {
        for (int i = 0; i < 8; i++) {
            desc->features[offset+i] = dpt[i];
        }
    }
}

__global__
void normalize_histogram( Descriptor* descs )
{
    Descriptor* desc = &descs[blockIdx.x];

    float*  ptr1 = desc->features;
    float4* ptr4 = (float4*)ptr1;

    float4 descr;
    descr = ptr4[threadIdx.x];

#ifdef USE_ROOT_SIFT
    // L1 norm
    float norm = descr.x + descr.y + descr.z + descr.w;

    norm += __shfl_down( norm, 16 );
    norm += __shfl_down( norm,  8 );
    norm += __shfl_down( norm,  4 );
    norm += __shfl_down( norm,  2 );
    norm += __shfl_down( norm,  1 );
    norm += __shfl     ( norm,  0 );

    norm = 512.0f / norm; /* multiplying with 512 is some scaling by convention */
    descr.x *= norm;
    descr.y *= norm;
    descr.z *= norm;
    descr.w *= norm;
#else // not USE_ROOT_SIFT
    // L2 norm
    descr.x *= descr.x;
    descr.y *= descr.y;
    descr.z *= descr.z;
    descr.w *= descr.w;

    float norm = descr.x + descr.y + descr.z + descr.w;

    norm += __shfl_down( norm, 16 );
    norm += __shfl_down( norm,  8 );
    norm += __shfl_down( norm,  4 );
    norm += __shfl_down( norm,  2 );
    norm += __shfl_down( norm,  1 );

    norm = sqrt(norm) + EPS_V1;

    norm += __shfl     ( norm,  0 );

    descr.x /= norm;
    descr.y /= norm;
    descr.z /= norm;
    descr.w /= norm;
#endif // not USE_ROOT_SIFT

    ptr4[threadIdx.x] = descr;
}


/*************************************************************
 * V4: host side
 *************************************************************/
__host__
void Pyramid::descriptors_v1( )
{
    for( int octave=0; octave<_num_octaves; octave++ ) {
        Octave&      oct_obj = _octaves[octave];

        for( int level=1; level<_levels-1; level++ ) {
            cudaStreamSynchronize( oct_obj.getStream(level) );
        }

        // async copy of extrema from device to host
        oct_obj.readExtremaCount( );
    }

    // wait until that is finished, so we can alloc space for descriptor
    cudaDeviceSynchronize( );

    for( int octave=0; octave<_num_octaves; octave++ ) {
        // allocate the descriptor array for this octave, all levels
        _octaves[octave].allocDescriptors( );
    }

    for( int octave=0; octave<_num_octaves; octave++ ) {
        Octave&      oct_obj = _octaves[octave];

        for( int level=1; level<_levels-1; level++ ) {
            dim3 block;
            dim3 grid;
            grid.x  = oct_obj.getExtremaMgmtH(level)->counter;

            if( grid.x != 0 ) {
                block.x = DESCR_V1_NUM_THREADS;
                block.y = 4;
                block.z = 4;

                keypoint_descriptors
                    <<<grid,block,0,oct_obj.getStream(level)>>>
                    ( oct_obj.getExtrema( level ),
                      oct_obj.getDescriptors( level ),
                      oct_obj.getData( level ) );

                block.x = 32;
                block.y = 1;
                block.z = 1;

                normalize_histogram
                    <<<grid,block,0,oct_obj.getStream(level)>>>
                    ( oct_obj.getDescriptors( level ) );
            }
        }
    }

    cudaDeviceSynchronize( );
}

