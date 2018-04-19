/*
 * Copyright 2018, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <iostream>

#include <cuda_runtime.h>
#include <thrust/for_each.h>
#include <thrust/system/cuda/execution_policy.h>

#include "registration.h"
#include "common/debug_macros.h"
#include "common/assist.h"
#include "common/write_plane_2d.h"

namespace popsift {

Registration::Registration( )
    : _keypt_a( 0 )
    , _keypt_b( 0 )
    , _plane_a( 0 )
    , _plane_b( 0 )
{ }

void Registration::setKeypointsA( const FeaturesDev* p )
{
    _keypt_a = p;
}

void Registration::setKeypointsB( const FeaturesDev* p )
{
    _keypt_b = p;
}

void Registration::setPlaneA( const Plane2D<float>* p )
{
    _plane_a = p;
}

void Registration::setPlaneB( const Plane2D<float>* p )
{
    _plane_b = p;
}

void Registration::private_makeTexture( cudaTextureObject_t& tex, const Plane2D<float>* plane )
{
    cudaTextureDesc     texDesc;
    cudaResourceDesc    resDesc;

    memset( &texDesc, 0, sizeof(cudaTextureDesc) );
    memset( &resDesc, 0, sizeof(cudaResourceDesc) );

    texDesc.normalizedCoords = 0;   // address using width and height
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.addressMode[2]   = cudaAddressModeClamp;
    texDesc.readMode         = cudaReadModeElementType; // read as float
    texDesc.filterMode       = cudaFilterModeLinear; // bilinear interpolation
                            // cudaFilterModePoint; // nearest neighbour mode

    resDesc.resType                  = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr       = plane->getData();
    resDesc.res.pitch2D.desc.f       = cudaChannelFormatKindFloat;
    resDesc.res.pitch2D.desc.x       = 32; // sizeof(float)*8
    resDesc.res.pitch2D.desc.y       = 0;
    resDesc.res.pitch2D.desc.z       = 0;
    resDesc.res.pitch2D.desc.w       = 0;
    resDesc.res.pitch2D.pitchInBytes = plane->getPitch();
    resDesc.res.pitch2D.width        = plane->getWidth();
    resDesc.res.pitch2D.height       = plane->getHeight();

    cudaError_t err;
    err = cudaCreateTextureObject( &tex, &resDesc, &texDesc, 0 );
    POP_CUDA_FATAL_TEST( err, "Could not create texture object for registration: " );
}

void Registration::private_destroyTexture( cudaTextureObject_t& tex )
{
    cudaError_t err;
    err = cudaDestroyTextureObject( tex );
    POP_CUDA_FATAL_TEST( err, "Could not destroy texture object at end of registration: " );
}

void Registration::private_makeMatches( int3*& match_a_to_b, int& good_match_len, int*& good_match_a )
{
    match_a_to_b = _keypt_a->match_AllocMatchTable( );

    _keypt_a->match_computeMatchTable( match_a_to_b, _keypt_b );

    int a_len = _keypt_a->getDescriptorCount( );
    int b_len = _keypt_b->getDescriptorCount( );

    good_match_a = _keypt_a->match_getAcceptedIndex( match_a_to_b, good_match_len );
}

void Registration::private_destroyMatches( int3* match_a_to_b, int* good_match_a )
{
    _keypt_a->match_freeAcceptedIndex( good_match_a );
    _keypt_a->match_freeMatchTable( match_a_to_b );
}

struct Transformation
{
};

class AffineTrans : public Transformation
{
    float v[6];

public:
    __device__ inline float&       A(int y, int x)       { return v[(y<<1)+x]; }
    __device__ inline float&       t(int x)              { return v[4+x]; }
    __device__ inline const float& A(int y, int x) const { return v[(y<<1)+x]; }
    __device__ inline const float& t(int x) const        { return v[4+x]; }

    __device__ inline void initA( float a11, float a12 )
    {
        A(0,0) =  a11;
        A(0,1) = -a12;
        A(1,0) =  a12;
        A(1,1) =  a11;
    }
    __device__ inline void initt( float ax, float ay, float bx, float by )
    {
        t(0) = bx - ( ax * A(0,0) + ay * A(0,1) );
        t(1) = by - ( ax * A(1,0) + ay * A(1,1) );
    }
};

/** A simple Weighting class with equal weight for all pixel in the sampling grid.
 *  The grid size is T x T.
 */
template <int T>
class ConstantWeight
{
public:
    __device__ inline float operator()( int y, int x ) const {
        return ( 1.0f / float(T*T) );
    }
};

__global__
void print_pu( const Feature*      aptr,
               const Feature*      bptr,
               const int2*         matches,
               cudaTextureObject_t texA,
               cudaTextureObject_t texB,
               Plane2D<float>      debugPlane )
{
    int2  keypt = matches[blockIdx.x];

    const Feature& a = aptr[keypt.x];
    const Feature& b = bptr[keypt.y];

    // const float ascale      = a.scale * DESC_MAGNIFY; - never used
    const float bscale      = b.scale * DESC_MAGNIFY * 100;
    const float scale_ratio = b.scale / a.scale;

    const float& ax = a.xpos;
    const float& ay = a.ypos;
    const float& bx = b.xpos;
    const float& by = b.ypos;

    // 1. take the rotation of both points
    // const float cos_a = cosf( a.orientation[0] );
    // const float sin_a = sinf( a.orientation[0] );
    // const float cos_b = cosf( b.orientation[0] );
    // const float sin_b = sinf( b.orientation[0] );
    //
    // 2. we have no use for the rotation of the target
    // image, only the difference of orientations matters.
    // const float cos_a = 1.0f;
    // const float sin_a = 0.0f;
    // const float cos_b = cosf( a.orientation[0] - b.orientation[0] );
    // const float sin_b = sinf( a.orientation[0] - b.orientation[0] );
    //
    // 3. use __sincosf for a little bit of speed-up, but slightly more
    // inaccurate than option 2
    // const float cos_a = 1.0f; - never used
    // const float sin_a = 0.0f; - never used
    float       cos_b;
    float       sin_b;
    __sincosf( a.orientation[0] - b.orientation[0], &sin_b, &cos_b );

    // 1. original using variable orientation of a
    // const float a11 = scale_ratio * (  cos_a * cos_b + sin_a * sin_b );
    // const float a12 = scale_ratio * (  cos_a * sin_b - cos_b * sin_a );
    //
    // 2. use only relative orientation, cos_a->1 sin_a->0
    const float a11 = scale_ratio * cos_b;
    const float a12 = scale_ratio * sin_b;

    AffineTrans trans;
    trans.initA( a11, a12 );
    trans.initt( ax, ay, bx, by );

    ConstantWeight<32> weight;

    float diff = 0.0f;

    const float step  = ( 1.0f / ( float(32)-1.0f ) ) * bscale;

    int debug_plane_idx = blockIdx.x - 100;

    float       col   = bx - 0.5f * bscale;
    const float row   = by - 0.5f * bscale + threadIdx.x * step;
    for( int i=0; i<32; col += step, i++ )
    {
        const float rx = trans.A(0,0) * (bx+col) + trans.A(0,1) * (by+row) + trans.t(0);
        const float ry = trans.A(1,0) * (bx+col) + trans.A(1,1) * (by+row) + trans.t(1);
        const float src = readTex( texB, bx+col, by+row );
        const float tgt = readTex( texA, rx, ry );
        diff = diff
             + weight(i,threadIdx.x) * ( src - tgt );
        if( debug_plane_idx >= 0 && debug_plane_idx < 10 )
        {
            debugPlane.ptr( debug_plane_idx * 32 + threadIdx.x )[   i] = tgt;
            debugPlane.ptr( debug_plane_idx * 32 + threadIdx.x )[32+i] = src;
        }
    }

    diff += __shfl_down( diff, 16 );
    diff += __shfl_down( diff,  8 );
    diff += __shfl_down( diff,  4 );
    diff += __shfl_down( diff,  2 );
    diff += __shfl_down( diff,  1 );
    diff  = __shfl     ( diff,  0 );

    if( threadIdx.x == 0 )
    {
#if 0
        if( diff != 0 ) {
            printf( "A=[ [ %8.6f %8.6f ] [ %8.6f %8.6f ] ] t=[ %8.6f %8.6f ]\n",
                    A_ab[0][0], A_ab[0][1], A_ab[1][0], A_ab[1][1],
                    t_ab[0], t_ab[1] );
        }
#else
        if( debug_plane_idx >= 0 && debug_plane_idx < 10 )
        {
        printf("%4d %4.2f,%4.2f,%4.2f to %4.2f,%4.2f,%4.2f dist %4.2f\n",
                blockIdx.x,
                a.xpos,
                a.ypos,
                a.scale,
                b.xpos,
                b.ypos,
                b.scale,
                diff );
        }
#endif
    }
}

void Registration::compute( )
{
    Plane2D<float> devDebugPlane;
    Plane2D<float> hstDebugPlane;

    devDebugPlane.allocDev ( 64, 10*32 );
    hstDebugPlane.allocHost( 64, 10*32, CudaAllocated );

    cudaTextureObject_t texA;
    cudaTextureObject_t texB;

    private_makeTexture( texA, _plane_a );
    private_makeTexture( texB, _plane_b );

    thrust::device_vector<int2> matching_features;
    _keypt_a->match_getAcceptedFeatureMatches( matching_features, _keypt_b );

    int size = matching_features.end() - matching_features.begin();
    print_pu
        <<<size,32>>>
        ( _keypt_a->getFeatures(),
          _keypt_b->getFeatures(),
          thrust::raw_pointer_cast( matching_features.data() ),
          texA,
          texB,
          devDebugPlane );

    hstDebugPlane.memcpyFromDevice( devDebugPlane );
    write_plane2D( "test.pgm", hstDebugPlane );

    std::cout << "Doing nothing" << std::endl;

    private_destroyTexture( texA );
    private_destroyTexture( texB );
}

} // namespace popsift
