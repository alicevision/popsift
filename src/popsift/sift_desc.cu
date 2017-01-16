/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <iostream>
#include <stdio.h>
#include <iso646.h>

#include "sift_pyramid.h"
#include "sift_constants.h"
#include "s_gradiant.h"
#include "assist.h"

/*************************************************************
 * V1: device side
 *************************************************************/

using namespace popsift;
using namespace std;

inline __device__
void keypoint_descriptors_sub( const float     ang,
                               const Extremum* ext,
                               Descriptor*     desc,
                               Plane2D_float   layer )
{
    const int width  = layer.getWidth();
    const int height = layer.getHeight();

    // int bidx = blockIdx.x & 0xf; // lower 4 bits of block ID
    const int ix   = threadIdx.y; // bidx & 0x3;       // lower 2 bits of block ID
    const int iy   = threadIdx.z; // bidx >> 2;        // next lowest 2 bits of block ID

    const float x    = ext->xpos;
    const float y    = ext->ypos;
    const float sig  = ext->sigma;
    // const float ang  = ext->orientation;
    const float SBP  = fabsf(DESC_MAGNIFY * sig);

    if( SBP == 0 ) {
        return;
    }

    // const float cos_t = cosf(ang);
    // const float sin_t = sinf(ang);
    float cos_t;
    float sin_t;
    __sincosf( ang, &sin_t, &cos_t );

    const float csbp  = cos_t * SBP;
    const float ssbp  = sin_t * SBP;
    const float crsbp = cos_t / SBP;
    const float srsbp = sin_t / SBP;

    const float offsetptx = ix - 1.5f;
    const float offsetpty = iy - 1.5f;

    // The following 2 lines were the primary bottleneck of this kernel
    // const float ptx = csbp * offsetptx - ssbp * offsetpty + x;
    // const float pty = csbp * offsetpty + ssbp * offsetptx + y;
    const float ptx = ::fmaf( csbp, offsetptx, ::fmaf( -ssbp, offsetpty, x ) );
    const float pty = ::fmaf( csbp, offsetpty, ::fmaf(  ssbp, offsetptx, y ) );

    const float bsz = fabsf(csbp) + fabsf(ssbp);

    const int xmin = max(1,          (int)floorf(ptx - bsz));
    const int ymin = max(1,          (int)floorf(pty - bsz));
    const int xmax = min(width - 2,  (int)floorf(ptx + bsz));
    const int ymax = min(height - 2, (int)floorf(pty + bsz));

    const int wx = xmax - xmin + 1;
    const int hy = ymax - ymin + 1;
    const int loops = wx * hy;

    float dpt[9] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for( int i = threadIdx.x; i < loops; i+=blockDim.x )
    {
        const int ii = i / wx + ymin;
        const int jj = i % wx + xmin;     

        const float dx = jj - ptx;
        const float dy = ii - pty;
        const float nx = ::fmaf( crsbp, dx,  srsbp * dy ); // crsbp * dx + srsbp * dy;
        const float ny = ::fmaf( crsbp, dy, -srsbp * dx ); // crsbp * dy - srsbp * dx;
        const float nxn = fabsf(nx);
        const float nyn = fabsf(ny);
        if (nxn < 1.0f && nyn < 1.0f) {
            const float2 mod_th = get_gradiant( jj, ii, layer );
            const float& mod    = mod_th.x;
            float        th     = mod_th.y;

            const float dnx = nx + offsetptx;
            const float dny = ny + offsetpty;
            const float ww  = __expf( -scalbnf(dnx*dnx + dny*dny, -3)); // speedup !
            // const float ww  = __expf(-0.125f * (dnx*dnx + dny*dny)); // speedup !
            const float wx  = 1.0f - nxn;
            const float wy  = 1.0f - nyn;
            const float wgt = ww * wx * wy * mod;

            th -= ang;
            th += ( th <  0.0f  ? M_PI2 : 0.0f ); //  if (th <  0.0f ) th += M_PI2;
            th -= ( th >= M_PI2 ? M_PI2 : 0.0f ); //  if (th >= M_PI2) th -= M_PI2;

            const float tth  = __fmul_ru( th, M_4RPI ); // th * M_4RPI;
            const int   fo0  = (int)floorf(tth);
            const float do0  = tth - fo0;             
            const float wgt1 = 1.0f - do0;
            const float wgt2 = do0;

            int fo  = fo0 % DESC_BINS;
            // if(fo < 8) {
                // maf: multiply-add
                // _ru - round to positive infinity equiv to froundf since always >=0
            dpt[fo]   = __fmaf_ru( wgt1, wgt, dpt[fo] );   // dpt[fo]   += (wgt1*wgt);
            dpt[fo+1] = __fmaf_ru( wgt2, wgt, dpt[fo+1] ); // dpt[fo+1] += (wgt2*wgt);
            // }
        }
        __syncthreads();
    }

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
    int offset = ( ( ( threadIdx.z << 2 ) + threadIdx.y ) << 3 ); // ( ( threadIdx.z * 4 ) + threadIdx.y ) * 8;

    if( threadIdx.x < 8 ) {
        desc->features[offset+threadIdx.x] = dpt[threadIdx.x];
    }
}

__global__
void keypoint_descriptors( Extremum*     extrema,
                           Descriptor*   descs,
                           int*          feat_to_ext_map,
                           Plane2D_float layer )
{
    const int   offset   = blockIdx.x;
    Descriptor* desc     = &descs[offset];
    const int   ext_idx  = feat_to_ext_map[offset];
    Extremum*   ext      = &extrema[ext_idx];
    const int   ext_base = ext->idx_ori;
    const int   ext_num  = offset - ext_base;
    const float ang      = ext->orientation[ext_num];

    keypoint_descriptors_sub( ang,
                              ext,
                              desc,
                              layer );
}

__global__
void normalize_histogram_root_sift( Descriptor* descs, int num_orientations )
{
    // root sift normalization

    int offset = blockIdx.x * 32 + threadIdx.y;

    // all of these threads are useless
    if( blockIdx.x * 32 >= num_orientations ) return;

    bool ignoreme = ( offset >= num_orientations );

    offset = ( offset < num_orientations ) ? offset
                                           : num_orientations-1;
    Descriptor* desc = &descs[offset];

    float*  ptr1 = desc->features;
    float4* ptr4 = (float4*)ptr1;

    float4 descr;
    descr = ptr4[threadIdx.x];

    float sum = descr.x + descr.y + descr.z + descr.w;

    sum += __shfl_down( sum, 16 );
    sum += __shfl_down( sum,  8 );
    sum += __shfl_down( sum,  4 );
    sum += __shfl_down( sum,  2 );
    sum += __shfl_down( sum,  1 );

    sum = __shfl( sum,  0 );

#if 1
    float val;
    val = scalbnf( __fsqrt_rn( __fdividef( descr.x, sum ) ),
                   d_consts.norm_multi );
    descr.x = val;
    val = scalbnf( __fsqrt_rn( __fdividef( descr.y, sum ) ),
                   d_consts.norm_multi );
    descr.y = val;
    val = scalbnf( __fsqrt_rn( __fdividef( descr.z, sum ) ),
                   d_consts.norm_multi );
    descr.z = val;
    val = scalbnf( __fsqrt_rn( __fdividef( descr.w, sum ) ),
                   d_consts.norm_multi );
    descr.w = val;
#else
    float val;
    val = 512.0f * __fsqrt_rn( __fdividef( descr.x, sum ) );
    descr.x = val;
    val = 512.0f * __fsqrt_rn( __fdividef( descr.y, sum ) );
    descr.y = val;
    val = 512.0f * __fsqrt_rn( __fdividef( descr.z, sum ) );
    descr.z = val;
    val = 512.0f * __fsqrt_rn( __fdividef( descr.w, sum ) );
    descr.w = val;
#endif

    if( not ignoreme ) {
        ptr4[threadIdx.x] = descr;
    }
}

__global__
void normalize_histogram( Descriptor* descs, int num_orientations )
{
    // OpenCV normalization

    int offset = blockIdx.x * 32 + threadIdx.y;

    // all of these threads are useless
    if( blockIdx.x * 32 >= num_orientations ) return;

    bool ignoreme = ( offset >= num_orientations );

    offset = ( offset < num_orientations ) ? offset
                                           : num_orientations-1;
    Descriptor* desc = &descs[offset];

    float*  ptr1 = desc->features;
    float4* ptr4 = (float4*)ptr1;

    float4 descr;
    descr = ptr4[threadIdx.x];

#undef HAVE_NORMF
    // normf() is an elegant function: sqrt(sum_0^127{v^2})
    // It exists from CUDA 7.5 but the trouble with CUB on the GTX 980 Ti forces
    // us to with CUDA 7.0 right now

#ifdef HAVE_NORMF
    float norm;

    if( threadIdx.x == 0 ) {
        norm = normf( 128, ptr1 );
    }

    norm = __shfl( norm,  0 );

    descr.x = min( descr.x, 0.2f*norm );
    descr.y = min( descr.y, 0.2f*norm );
    descr.z = min( descr.z, 0.2f*norm );
    descr.w = min( descr.w, 0.2f*norm );

    if( threadIdx.x == 0 ) {
        norm = scalbnf( rnormf( 128, ptr1 ), d_consts.norm_multi );
    }
#else
    float norm;

    norm = descr.x * descr.x
         + descr.y * descr.y
         + descr.z * descr.z
         + descr.w * descr.w;
    norm += __shfl_down( norm, 16 );
    norm += __shfl_down( norm,  8 );
    norm += __shfl_down( norm,  4 );
    norm += __shfl_down( norm,  2 );
    norm += __shfl_down( norm,  1 );
    if( threadIdx.x == 0 ) {
        norm = __fsqrt_rn( norm );
    }
    norm = __shfl( norm,  0 );

    descr.x = min( descr.x, 0.2f*norm );
    descr.y = min( descr.y, 0.2f*norm );
    descr.z = min( descr.z, 0.2f*norm );
    descr.w = min( descr.w, 0.2f*norm );

    norm = descr.x * descr.x
         + descr.y * descr.y
         + descr.z * descr.z
         + descr.w * descr.w;
    norm += __shfl_down( norm, 16 );
    norm += __shfl_down( norm,  8 );
    norm += __shfl_down( norm,  4 );
    norm += __shfl_down( norm,  2 );
    norm += __shfl_down( norm,  1 );
    if( threadIdx.x == 0 ) {
        // norm = __fsqrt_rn( norm );
        // norm = __fdividef( 512.0f, norm );
        norm = __frsqrt_rn( norm ); // inverse square root
        norm = scalbnf( norm, d_consts.norm_multi );
    }
#endif
    norm = __shfl( norm,  0 );

    descr.x = descr.x * norm;
    descr.y = descr.y * norm;
    descr.z = descr.z * norm;
    descr.w = descr.w * norm;

    if( not ignoreme ) {
        ptr4[threadIdx.x] = descr;
    }
}

#if __CUDA_ARCH__ > 350
__global__ void descriptor_starter( int*          extrema_counter,
                                    int*          featvec_counter,
                                    Extremum*     extrema,
                                    Descriptor*   descs,
                                    int*          feat_to_ext_map,
                                    Plane2D_float layer,
                                    bool          use_root_sift )
{
    dim3 block;
    dim3 grid;
    grid.x  = *featvec_counter;

    if( grid.x == 0 ) return;

    block.x = 32;
    block.y = 4;
    block.z = 4;

    keypoint_descriptors
        <<<grid,block>>>
        ( extrema,
          descs,
          feat_to_ext_map,
          layer );

    // it may be good to start more threads, but this kernel
    // is too fast to be noticable in profiling

    grid.x  = grid_divide( *featvec_counter, 32 );
    block.x = 32;
    block.y = 32;
    block.z = 1;

    if( use_root_sift ) {
        normalize_histogram_root_sift
            <<<grid,block>>>
            ( descs, *featvec_counter );
    } else {
        normalize_histogram
            <<<grid,block>>>
            ( descs, *featvec_counter );
    }
}
#else // __CUDA_ARCH__ > 350
__global__ void descriptor_starter( int*          extrema_counter,
                                    int*          featvec_counter,
                                    Extremum*     extrema,
                                    Descriptor*   descs,
                                    int*          feat_to_ext_map,
                                    Plane2D_float layer,
                                    bool          use_root_sift )
{
    printf( "Dynamic Parallelism requires a card with Compute Capability 3.5 or higher\n" );
}
#endif // __CUDA_ARCH__ > 350

/*************************************************************
 * V4: host side
 *************************************************************/
__host__
void Pyramid::descriptors( const Config& conf )
{
    if( conf.useDPDescriptors() ) {
        // cerr << "Calling descriptors with dynamic parallelism" << endl;

        for( int octave=0; octave<_num_octaves; octave++ ) {
            Octave&      oct_obj = _octaves[octave];

            for( int level=1; level<_levels-2; level++ ) {
                cudaStream_t oct_str = oct_obj.getStream(level+2);

                descriptor_starter
                    <<<1,1,0,oct_str>>>
                    ( oct_obj.getExtremaCtPtrD( level ),
                      oct_obj.getFeatVecCtPtrD( level ),
                      oct_obj.getExtrema( level ),
                      oct_obj.getDescriptors( level ),
                      oct_obj.getFeatToExtMapD( level ),
                      oct_obj.getData( level ),
                      conf.getUseRootSift() );
            }
        }

        cudaDeviceSynchronize();

        for( int octave=0; octave<_num_octaves; octave++ ) {
            Octave& oct_obj = _octaves[octave];
            oct_obj.readExtremaCount( );
        }
    } else {
        // cerr << "Calling descriptors -no- dynamic parallelism" << endl;

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


            for( int level=1; level<_levels-2; level++ ) {
                dim3 block;
                dim3 grid;
                grid.x = oct_obj.getFeatVecCountH( level );

                if( grid.x != 0 ) {
                    block.x = 32;
                    block.y = 4;
                    block.z = 4;

                    keypoint_descriptors
                        <<<grid,block,0,oct_obj.getStream(level+2)>>>
                        ( oct_obj.getExtrema( level ),
                          oct_obj.getDescriptors( level ),
                          oct_obj.getFeatToExtMapD( level ),
                          oct_obj.getData( level ) );

                    grid.x  = grid_divide( oct_obj.getFeatVecCountH( level ), 32 );
                    block.x = 32;
                    block.y = 32;
                    block.z = 1;

                    if( conf.getUseRootSift() ) {
                        normalize_histogram_root_sift
                            <<<grid,block,0,oct_obj.getStream(level+2)>>>
                            ( oct_obj.getDescriptors( level ),
                              oct_obj.getFeatVecCountH( level ) );
                    } else {
                        normalize_histogram
                            <<<grid,block,0,oct_obj.getStream(level+2)>>>
                            ( oct_obj.getDescriptors( level ),
                              oct_obj.getFeatVecCountH( level ) );
                    }
                }
            }
        }
    }

    cudaDeviceSynchronize( );
}

