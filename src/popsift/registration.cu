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

__global__
void print_pu( const Feature*      aptr,
               const Feature*      bptr,
               const int2*         matches,
               cudaTextureObject_t texA,
               cudaTextureObject_t texB )
{
    int2  keypt = matches[blockIdx.x];

    float diff = 0.0f;

    const Feature& a = aptr[keypt.x];
    const Feature& b = bptr[keypt.y];

    const float ascale = a.scale * DESC_MAGNIFY;
    const float bscale = b.scale * DESC_MAGNIFY;
    const float scale_ratio = bscale / ascale;

    const float ax = a.xpos;
    const float ay = a.ypos;
    const float bx = b.xpos;
    const float by = b.ypos;

    float cos_a;
    float sin_a;
    float cos_b;
    float sin_b;
    __sincosf( a.orientation[0], &sin_a, &cos_a );
    __sincosf( b.orientation[0], &sin_b, &cos_b );

    const float a11 = scale_ratio * (  cos_a * cos_b + sin_a * sin_b );
    const float a12 = scale_ratio * (  cos_a * sin_b - cos_b * sin_a );
    float4 A_ab = make_float4( a11, -a12, a12, a11 );
    float2 t_ab = make_float2( bx - ( ax * A_ab.x + ay * A_ab.y ),
                               by - ( ax * A_ab.z + ay * A_ab.w ) );

    const float step  = ( 1.0f / 31.0f ) * bscale;

    const float row   = by - 0.5f * bscale + threadIdx.x * step;
    const float bstop = bx + 0.5f*bscale + step/2.0f; /* last term is for float inaccuracy */
    for( float col=bx-0.5f*bscale; col <= bstop; col+=step )
    {
        diff += readTex( texB, bx+col, by+row );

        float rx = A_ab.x * (bx+col) + A_ab.y * (by+row) + t_ab.x;
        float ry = A_ab.z * (bx+col) + A_ab.w * (by+row) + t_ab.y;
        diff -= readTex( texA, rx, ry );
    }

    diff += __shfl_down( diff, 16 );
    diff += __shfl_down( diff,  8 );
    diff += __shfl_down( diff,  4 );
    diff += __shfl_down( diff,  2 );
    diff += __shfl_down( diff,  1 );
    diff  = __shfl     ( diff,  0 );

    if( threadIdx.x == 0 )
    {
        printf("%4.2f,%4.2f,%4.2f,%4.2f to %4.2f,%4.2f,%4.2f,%4.2f dist %4.2f\n",
                a.xpos,
                a.ypos,
                a.scale,
                a.orientation[0],
                a.xpos - b.xpos,
                a.ypos - b.ypos,
                a.scale - b.scale,
                a.orientation[0] - b.orientation[0],
                diff );
    }
}

void Registration::compute( )
{
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
          texB );

    std::cout << "Doing nothing" << std::endl;

    private_destroyTexture( texA );
    private_destroyTexture( texB );
}

} // namespace popsift
