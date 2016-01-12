#include "s_pyramid.h"
#include "s_gradiant.h"

#define DESCR_BINS_V1        8
#define MAGNIFY_V1           3.0f
#define DESCR_V1_NUM_THREADS 32
#define RPI_V1               (4.0f / M_PI)
#define EPS_V1               1E-15

/*************************************************************
 * V1: device side
 *************************************************************/

using namespace popart;

__global__
void keypoint_descriptors( ExtremumCandidate* cand,
                           Descriptor*        descs,
                           Plane2D_float      layer )
{
    uint32_t width  = layer.getWidth();
    uint32_t height = layer.getHeight();

    // int bidx = blockIdx.x & 0xf; // lower 4 bits of block ID
    int ix   = threadIdx.y; // bidx & 0x3;       // lower 2 bits of block ID
    int iy   = threadIdx.z; // bidx >> 2;        // next lowest 2 bits of block ID

    ExtremumCandidate* ext = &cand[blockIdx.x];

    if( ext->not_a_keypoint ) return;

    float x    = ext->xpos;
    float y    = ext->ypos;
    float sig  = ext->sigma;
    float ang  = ext->angle_from_bemap;
    float SBP  = fabs(MAGNIFY_V1 * sig);

    float cos_t = cosf(ang);
    float sin_t = sinf(ang);

    float csbp  = cos_t * SBP;
    float ssbp  = sin_t * SBP;
    float crsbp = cos_t / SBP;
    float srsbp = sin_t / SBP;

    float offsetptx = ix - 1.5f;
    float offsetpty = iy - 1.5f;
    float ptx = csbp * offsetptx - ssbp * offsetpty + x;
    float pty = csbp * offsetpty + ssbp * offsetptx + y;

    float bsz = fabs(csbp) + fabs(ssbp);

    int32_t xmin = max(1,          (int32_t)floor(ptx - bsz));
    int32_t ymin = max(1,          (int32_t)floor(pty - bsz));
    int32_t xmax = min(width - 2,  (int32_t)floor(ptx + bsz));
    int32_t ymax = min(height - 2, (int32_t)floor(pty + bsz));

    int32_t wx = xmax - xmin + 1;
    int32_t hy = ymax - ymin + 1;
    int32_t loops = wx * hy;

    float dpt[9];
    for (int i = 0; i < 9; i++) dpt[i] = 0.0f;

#if 0
    if( threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 ) {
        printf("loop %d\n", loops );
    }
#endif

    for(int i = threadIdx.x; i < loops; i+=DESCR_V1_NUM_THREADS)
    {
        int ii = i / wx + ymin;
        int jj = i % wx + xmin;     

        float dx = jj - ptx;
        float dy = ii - pty;
        float nx = crsbp * dx + srsbp * dy;
        float ny = crsbp * dy - srsbp * dx;
        float nxn = fabs(nx);
        float nyn = fabs(ny);
        if (nxn < 1.0f && nyn < 1.0f) {
#if 1
            float mod;
            float th;
            get_gradiant( mod, th,
                          ii, jj,
                          layer );
#else
            float mod = at(grad,  ii, jj);
            float th  = at(theta, ii, jj);
#endif
            float dnx = nx + offsetptx;
            float dny = ny + offsetpty;
            float ww  = __expf(-0.125f * (dnx*dnx + dny*dny));
            float wx  = 1.0f - nxn;
            float wy  = 1.0f - nyn;
            float wgt = ww * wx * wy * mod;

            th -= ang;
            while (th < 0.0f) th += M_PI2;
            while (th >= M_PI2) th -= M_PI2;

            float   tth  = th * RPI_V1;
            int32_t fo0  = (int32_t)floor(tth);
            float   do0  = tth - fo0;             
            float   wgt1 = 1.0f - do0;
            float   wgt2 = do0;

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
        dpt[i] += __shfl_down( dpt[i], 16 );
        dpt[i] += __shfl_down( dpt[i], 8 );
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

    ptr4[threadIdx.x] = descr;
}


/*************************************************************
 * V4: host side
 *************************************************************/
__host__
void Pyramid::descriptors_v1( )
{
    _keep_time_descr_v1.start();
    for( int octave=0; octave<_num_octaves; octave++ ) {
        // async copy of extrema from device to host
        _octaves[octave].readExtremaCount( );
    }

    // wait until that is finished, so we can alloc space for descriptor
    cudaDeviceSynchronize( );

    for( int octave=0; octave<_num_octaves; octave++ ) {
        // allocate the descriptor array for this octave, all levels
        _octaves[octave].allocDescriptors( );
    }

    for( int octave=0; octave<_num_octaves; octave++ ) {
        for( int level=1; level<_levels-1; level++ ) {
            dim3 block;
            dim3 grid;
            grid.x  = _octaves[octave].getExtremaMgmtH(level)->counter;

            if( grid.x != 0 ) {
                block.x = DESCR_V1_NUM_THREADS;
                block.y = 4;
                block.z = 4;

                keypoint_descriptors
                    <<<grid,block>>>
                    ( _octaves[octave].getExtrema( level ),
                      _octaves[octave].getDescriptors( level ),
                      _octaves[octave].getData( level ) );

                block.x = DESCR_V1_NUM_THREADS;
                block.y = 1;
                block.z = 1;

                normalize_histogram
                    <<<grid,block>>>
                    ( _octaves[octave].getDescriptors( level ) );
            }
        }
    }
    _keep_time_descr_v1.stop();
}

#undef DESCR_BINS_V1
#undef MAGNIFY_V1
#undef DESCR_V1_NUM_THREADS

