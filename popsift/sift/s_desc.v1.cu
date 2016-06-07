#include <iostream>
#include <stdio.h>

#include "sift_pyramid.h"
#include "sift_constants.h"
#include "s_gradiant.h"
#include "assist.h"

#undef DESCRIPTORS_FROM_UNBLURRED_IMAGE

/*************************************************************
 * V1: device side
 *************************************************************/

using namespace popart;
using namespace std;

#ifdef DEBUG_SEARCH_FOR_NANS
__global__
void pre_desc_validate_data_layer( int octave, int level, Plane2D_float layer )
{
    const int width  = layer.getWidth();
    const int height = layer.getHeight();
    for( int y=0; y<height; y++ ) {
        for( int x=0; x<width; x++ ) {
            if( isnan( layer.ptr(y)[x] ) ) {
                printf( "PRE DESC: Found a NAN value in blur layer octave %d level %d at (%d,%d)\n", octave, level, x,y );
                return;
            }
            if( isinf( layer.ptr(y)[x] ) ) {
                printf( "PRE DESC: Found an INF value in blur layer octave %d level %d at (%d,%d)\n", octave, level, x,y );
                return;
            }
        }
    }
}
#endif // DEBUG_SEARCH_FOR_NANS

__global__
void keypoint_descriptors( Extremum*     cand,
                           Descriptor*   descs,
                           Plane2D_float layer )
{
    const uint32_t width  = layer.getWidth();
    const uint32_t height = layer.getHeight();

    // int bidx = blockIdx.x & 0xf; // lower 4 bits of block ID
    const int ix   = threadIdx.y; // bidx & 0x3;       // lower 2 bits of block ID
    const int iy   = threadIdx.z; // bidx >> 2;        // next lowest 2 bits of block ID

    Extremum* ext = &cand[blockIdx.x];

#ifdef DEBUG_SEARCH_FOR_NANS
    if( ext->invalid ) return;
#endif // DEBUG_SEARCH_FOR_NANS

    const float x    = ext->xpos;
    const float y    = ext->ypos;
    const float sig  = ext->sigma;
    const float ang  = ext->orientation;
    const float SBP  = fabsf(DESC_MAGNIFY * sig);

    if( SBP == 0 ) {
#ifdef DEBUG_SEARCH_FOR_NANS
        if( threadIdx.x == 0 ) {
            ext->invalid |= SIGMA_NULL;
        }
#endif // DEBUG_SEARCH_FOR_NANS
        return;
    }

    // const float cos_t = cosf(ang);
    // const float sin_t = sinf(ang);
    float cos_t;
    float sin_t;
    sincosf( ang, &sin_t, &cos_t );

    const float csbp  = cos_t * SBP;
    const float ssbp  = sin_t * SBP;
    const float crsbp = cos_t / SBP;
    const float srsbp = sin_t / SBP;

    const float offsetptx = ix - 1.5f;
    const float offsetpty = iy - 1.5f;
    const float ptx = csbp * offsetptx - ssbp * offsetpty + x;
    const float pty = csbp * offsetpty + ssbp * offsetptx + y;

    const float bsz = fabsf(csbp) + fabsf(ssbp);

    const int32_t xmin = max(1,          (int32_t)floorf(ptx - bsz));
    const int32_t ymin = max(1,          (int32_t)floorf(pty - bsz));
    const int32_t xmax = min(width - 2,  (int32_t)floorf(ptx + bsz));
    const int32_t ymax = min(height - 2, (int32_t)floorf(pty + bsz));

    const int32_t wx = xmax - xmin + 1;
    const int32_t hy = ymax - ymin + 1;
    const int32_t loops = wx * hy;

    float dpt[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    // for (int i = 0; i < 9; i++) dpt[i] = 0.0f;

    for(int i = threadIdx.x; i < loops; i+=32)
    {
#ifdef DEBUG_SEARCH_FOR_NANS
        if( wx + ymin == 0 ) {
            ext->invalid |= DESC_WINDOW_EMPTY;
        }
#endif // DEBUG_SEARCH_FOR_NANS

        const int ii = i / wx + ymin;
        const int jj = i % wx + xmin;     

        const float dx = jj - ptx;
        const float dy = ii - pty;
        const float nx = crsbp * dx + srsbp * dy;
        const float ny = crsbp * dy - srsbp * dx;
        const float nxn = fabs(nx);
        const float nyn = fabs(ny);
        if (nxn < 1.0f && nyn < 1.0f) {
            const float2 mod_th = get_gradiant( jj, ii, layer );
#ifdef DEBUG_SEARCH_FOR_NANS
            if( mod_th.x == HUGE_VALF || isnan(mod_th.x) ) {
                ext->invalid |= HYPOT_OUT_OF_RANGE;
            }
            if( isnan(mod_th.y) ) {
                ext->invalid |= ATAN_OUT_OF_RANGE;
            }
#endif // DEBUG_SEARCH_FOR_NANS
            const float& mod    = mod_th.x;
            float        th     = mod_th.y;

            const float dnx = nx + offsetptx;
            const float dny = ny + offsetpty;
            const float ww  = __expf(-0.125f * (dnx*dnx + dny*dny));
            const float wx  = 1.0f - nxn;
            const float wy  = 1.0f - nyn;
            const float wgt = ww * wx * wy * mod;

            th -= ang;
            while (th < 0.0f) th += M_PI2;
            while (th >= M_PI2) th -= M_PI2;

            const float   tth  = th * M_4RPI;
            const int32_t fo0  = (int32_t)floorf(tth);
            const float   do0  = tth - fo0;             
            const float   wgt1 = 1.0f - do0;
            const float   wgt2 = do0;

            int fo  = fo0 % DESC_BINS;
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

    if( threadIdx.x < 8 ) {
        desc->features[offset+threadIdx.x] = dpt[threadIdx.x];
#ifdef DEBUG_SEARCH_FOR_NANS
        if( isnan( dpt[threadIdx.x] ) || isinf( dpt[threadIdx.x] ) ) {
            if( not ext->invalid )
                ext->invalid |= NAN_SOURCE_UNKNOWN;
        }
#endif // DEBUG_SEARCH_FOR_NANS
    }
}

__global__
void normalize_histogram( Descriptor* descs, int num_orientations )
{
    int offset = ( blockIdx.x < num_orientations ) ? blockIdx.x
                                                   : num_orientations-1;
    Descriptor* desc = &descs[offset];

    float*  ptr1 = desc->features;
    float4* ptr4 = (float4*)ptr1;

    float4 descr;
    descr = ptr4[threadIdx.x];

#ifdef DESC_USE_ROOT_SIFT
    // L1 norm
    float sum = descr.x + descr.y + descr.z + descr.w;

    sum += __shfl_down( sum, 16 );
    sum += __shfl_down( sum,  8 );
    sum += __shfl_down( sum,  4 );
    sum += __shfl_down( sum,  2 );
    sum += __shfl_down( sum,  1 );

    sum = __shfl( sum,  0 );

    /* multiplying with 512 is some scaling by convention */
    // sum = 512.0f / sum;
    // sum = __frcp_rn( scalbnf( sum, -9 ) );
    float val;
    val = 512.0f * __fsqrt_rn( __fdividef( descr.x, sum ) );
    descr.x = val;
    val = 512.0f * __fsqrt_rn( __fdividef( descr.y, sum ) );
    descr.y = val;
    val = 512.0f * __fsqrt_rn( __fdividef( descr.z, sum ) );
    descr.z = val;
    val = 512.0f * __fsqrt_rn( __fdividef( descr.w, sum ) );
    descr.w = val;
#else // not DESC_USE_ROOT_SIFT
    // L2 norm
    __shared__ float normalized_y[32];

    float norm;
    if( threadIdx.y == 0 ) {
        // rnormf() is powerful: reciprocal L2 norm of N 
        // consecutive floats, here N=128
        norm = rnormf( 128, ptr1 );
        normalized_y[threadIdx.x] = norm;
    }
    __syncthreads();
    norm = normalized_y[threadIdx.y];

    descr.x *= norm;
    descr.y *= norm;
    descr.z *= norm;
    descr.w *= norm;
#endif // not DESC_USE_ROOT_SIFT

    ptr4[threadIdx.x] = descr;
}

__global__ void descriptor_starter( int*          extrema_counter,
                                    Extremum*     extrema,
                                    Descriptor*   descs,
                                    Plane2D_float layer )
{
#ifdef USE_DYNAMIC_PARALLELISM
    dim3 block;
    dim3 grid;
    grid.x  = *extrema_counter;

    if( grid.x == 0 ) return;

    // printf("Number of extrema after ori: %d\n", grid.x );

    block.x = 32;
    block.y = 4;
    block.z = 4;

    keypoint_descriptors
        <<<grid,block>>>
        ( extrema,
          descs,
          layer );

    // it may be good to start more threads, but this kernel
    // is too fast to be noticable in profiling
    grid.x  = grid_divide( *extrema_counter, 32 );
    block.x = 32;
    block.y = 32;
    block.z = 1;

    normalize_histogram
        <<<grid,block>>>
        ( descs, *extrema_counter );
#endif // not USE_DYNAMIC_PARALLELISM
}

/*************************************************************
 * V4: host side
 *************************************************************/
__host__
void Pyramid::descriptors_v1( )
{
#ifdef USE_DYNAMIC_PARALLELISM
    cerr << "Calling descriptors with dynamic parallelism" << endl;
    for( int octave=0; octave<_num_octaves; octave++ ) {
        Octave&      oct_obj = _octaves[octave];

        for( int level=1; level<_levels-2; level++ ) {
            cudaStream_t oct_str = oct_obj.getStream(level+2);

#ifdef DESCRIPTORS_FROM_UNBLURRED_IMAGE
            Plane2D_float& data = oct_obj.getData( 0 );
#ifdef DEBUG_SEARCH_FOR_NANS
            pre_desc_validate_data_layer
                <<<1,1,0,oct_str>>>
                ( octave, 0, data );
#endif
#else // not DESCRIPTORS_FROM_UNBLURRED_IMAGE
            Plane2D_float& data = oct_obj.getData( level );
#ifdef DEBUG_SEARCH_FOR_NANS
            pre_desc_validate_data_layer
                <<<1,1,0,oct_str>>>
                ( octave, level, data );
#endif
#endif // not DESCRIPTORS_FROM_UNBLURRED_IMAGE

            int* extrema_counters = oct_obj.getExtremaMgmtD();
            int* extrema_counter  = &extrema_counters[level];
            descriptor_starter
                <<<1,1,0,oct_str>>>
                ( extrema_counter,
                  oct_obj.getExtrema( level ),
                  oct_obj.getDescriptors( level ),
                  data );
        }
    }

    cudaDeviceSynchronize();

    for( int octave=0; octave<_num_octaves; octave++ ) {
        Octave& oct_obj = _octaves[octave];
        oct_obj.readExtremaCount( );
    }
#else // not USE_DYNAMIC_PARALLELISM
    cerr << "Calling descriptors -no- dynamic parallelism" << endl;
    for( int octave=0; octave<_num_octaves; octave++ ) {
        Octave&      oct_obj = _octaves[octave];

        for( int level=3; level<_levels; level++ ) {
            cudaStreamSynchronize( oct_obj.getStream(level) );
        }

        // async copy of extrema from device to host
        oct_obj.readExtremaCount( );
    }

    for( int octave=0; octave<_num_octaves; octave++ ) {
        Octave&      oct_obj = _octaves[octave];

        int* num_orientations = oct_obj.getExtremaMgmtH();

        for( int level=1; level<_levels-2; level++ ) {
            dim3 block;
            dim3 grid;
            grid.x  = num_orientations[level];

            if( grid.x != 0 ) {
                block.x = 32;
                block.y = 4;
                block.z = 4;

#ifdef DESCRIPTORS_FROM_UNBLURRED_IMAGE
                Plane2D_float& data = oct_obj.getData( 0 );
#else // not DESCRIPTORS_FROM_UNBLURRED_IMAGE
                Plane2D_float& data = oct_obj.getData( level );
#endif // not DESCRIPTORS_FROM_UNBLURRED_IMAGE

                keypoint_descriptors
                    <<<grid,block,0,oct_obj.getStream(level+2)>>>
                    ( oct_obj.getExtrema( level ),
                      oct_obj.getDescriptors( level ),
                      data );

                grid.x  = grid_divide( num_orientations[level], 32 );
                block.x = 32;
                block.y = 32;
                block.z = 1;

                normalize_histogram
                    <<<grid,block,0,oct_obj.getStream(level+2)>>>
                    ( oct_obj.getDescriptors( level ),
                      num_orientations[level] );
            }
        }
    }

    cudaDeviceSynchronize( );
#endif // not USE_DYNAMIC_PARALLELISM
}

