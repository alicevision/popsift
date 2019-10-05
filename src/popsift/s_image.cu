/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "s_image.h"
#include <iostream>
#include <fstream>
#include "common/debug_macros.h"
#include "common/assist.h"
#include <stdio.h>
#include <assert.h>

#ifdef USE_NVTX
#include <nvToolsExtCuda.h>
#else
#define nvtxRangePushA(a)
#define nvtxRangePop()
#endif

using namespace std;

namespace popsift {

/*************************************************************
 * ImageBase
 *************************************************************/

ImageBase::ImageBase( )
    : _w(0), _h(0)
    , _max_w(0), _max_h(0)
{
}

ImageBase::ImageBase( int w, int h )
    : _w(w), _h(h)
    , _max_w(w), _max_h(h)
{
}

ImageBase::~ImageBase( )
{
}

    // Taken from here: https://se.mathworks.com/matlabcentral/answers/287847-what-is-wallis-filter-i-have-an-essay-on-it-and-i-cannot-understand-of-find-info-on-it

    // function WallisFilter(obj, Md, Dd, Amax, p, W)
    // Md and Dd are mean and contrast to match,
    // Amax and p constrain the change in individual pixels,
void ImageBase::wallisFilter( int filterWidth )
{
    const NppiSize COMPLETE = { .height = _h, .width = _w };
    const NppiPoint NOOFFSET = { .w = 0, .h = 0 };

    if( filterWidth %2 == 0 ) filterWidth++;
    const NppiSize FILTERSIZE = { .height = filterWidth, .width = filterWidht };

    int w = filterWidth >> 1; // floor(W/2)


    Plane<float> M( _w, _h );
    nppiFilterBox_32f_C1R( _input_image_d.getBuffer(), // src ptr
                           _input_image_d.getPitch(),  // src step
                           M.getBuffer(),              // dst ptr
                           M.getPitch(),               // dst step
                           COMPLETE,                   // region
                           FILTERSIZE,                 // filtersize
                           NOOFFSET );                 // shift
    // Plane<float> ipsum( _w, _h );
    // compute the inclusive prefix sum on all horizontals
    // after that compute the inclusive prefix sum on all verticals
    // that creates the basis for a box filter
    // ipsum = initBoxFilter( _input_image_d );
    // compute box filter ( pix(x+filterWidth/2,y+filterWidth/2) - pix(x-filterWidth/2,y-filterWidth/2) ) / filerWidth^2
    // M = runBoxFilter( ipsum, w );

    Plane<float> FminusM( _w, _h );
    // FminusM = _input_image_d - M; // element-wise substract 
    nppiSub_32s_C1R( _input_image_d.getBuffer(),
                     _input_image_d.getPitch(),
                     M.getBuffer(),
                     M.getPitch(),
                     FminusM.getBuffer(),
                     FminusM.getPitch(),
                     COMPLETE );

    Plane<float> Dprep( _w, _h );
    Plane<float> D( _w, _h );
    // compute element-wise: ( _input_image_d[pos] - M[pos] )^2
    // D = FminusM;
    // D.square(); // element-wise square
    nppiSqr_32f_C1R( FminusM.getBuffer(),
                     FminusM.getPitch(),
                     Dprep.getBuffer(),
                     Dprep.getPitch(),
                     COMPLETE );

    // ipsum = initBoxFilter( D );
    // D = runBoxFilter( ipsum, w );
    // D.divide( filterWidth^2 );
    nppiFilterBox_32f_C1R( Dprep.getBuffer(),
                           Dprep.getPitch(),
                           D.getBuffer(),
                           D.getPitch(),
                           COMPLETE,
                           FILTERSIZE,
                           NOOFFSET );

    // D.sqrt();
    nppiSqrt_32f_C1IR( D.getBuffer(),
                       D.getPitch(),
                       COMPLETE );
    // D.multiply( Amax );
    nppiMulC_32f_C1IR( Amax,
                       D.getBuffer(),
                       D.getPitch(),
                       COMPLETE );
    // D.add( Dd );
    nppiAddC_32f_C1IR( Dd,
                       D.getBuffer(),
                       D.getPitch(),
                       COMPLETE );

    Plane<float> G( _w, _h );
    // G = FminusM;
    // G.multiply( Amax * Dd );
    nppiMulC_32f_C1R( D.getBuffer(),
                      D.getPitch(),
                      Amax * Dd,
                      G.getBuffer(),
                      G.getPitch(),
                      COMPLETE );

    // D = G / D; // element-wise division
    nppiDiv_32f_C1IR( G.getBuffer(),
                      G.getPitch(),
                      D.getBuffer(),
                      D.getPitch(),
                      COMPLETE );

    // D.add( p * Md );
    nppiAddC_32f_C1IR( p * Md,
                       D.getBuffer(),
                       D.getPitch(),
                       COMPLETE );
    // M.multiply( 1-p );
    nppiMulC_32f_C1IR( 1.0f-p,
                       M.getBuffer(),
                       M.getPitch(),
                       COMPLETE );
    // D = D + M; // element-wise addition
    nppiAdd_32f_C1IR( M.getBuffer(),
                      M.getPitch(),
                      D.getBuffer(),
                      D.getPitch(),
                      COMPLETE );
    // D.max(0);
    nppiThreshold_LTVal_32f_C1IR( D.getBuffer(),
                                  D.getPitch(),
                                  COMPLETE,
                                  0.0f,   // if less-than this
                                  0.0f ); // set to this value
    // D.min(65534)
    nppiThreshold_GTVal_32f_C1IR( D.getBuffer(),
                                  D.getPitch(),
                                  COMPLETE,
                                  65534.0f,   // if greater-than this
                                  65534.0f ); // set to this value
}

/*************************************************************
 * Image
 *************************************************************/

Image::Image( )
    : ImageBase( 0, 0 )
{
}

Image::Image( int w, int h )
    : ImageBase( w, h )
{
    allocate( w, h );
}

Image::~Image( )
{
    if( _max_w == 0 ) return;

    destroyTexture( );
    _input_image_d.freeDev( );
    _input_image_h.freeHost( popsift::CudaAllocated );
}

void Image::load( void* input )
{
    /* The host memcpy may seem like a really stupid idea, but _input_image_h
     * is in CUDA-allocated pinned host memory, which makes the H2D copy
     * much faster.
     */
    memcpy( _input_image_h.data, input, _w*_h ); // assume that host Plane2D has no pitch
    _input_image_h.memcpyToDevice( _input_image_d );
}

void Image::resetDimensions( int w, int h )
{
    if( _max_w == 0 && _max_h == 0 ) {
        _max_w = _w = w;
        _max_h = _h = h;
        allocate( w, h );
        return;
    }

    if( w == _w && h == _h ) return;
        /* everything OK, nothing to do */

    _w = w;
    _h = h;

    if( w <= _max_w && h <= _max_h ) {
        _input_image_h.resetDimensionsHost( w, h );
        _input_image_d.resetDimensionsDev( w, h );

        destroyTexture( );
        createTexture( );
    } else {
        nvtxRangePushA( "reallocating host-side image memory" );

        _max_w = max( w, _max_w );
        _max_h = max( h, _max_h );
        _input_image_h.freeHost( popsift::CudaAllocated );
        _input_image_d.freeDev( );
        _input_image_h.allocHost( _max_w, _max_h, popsift::CudaAllocated );
        _input_image_d.allocDev(  _max_w, _max_h );
        _input_image_h.resetDimensionsHost( w, h );
        _input_image_d.resetDimensionsDev( w, h );

        destroyTexture( );
        createTexture( );

        nvtxRangePop(); // "reallocating host-side image memory"
    }
}

void Image::allocate( int w, int h )
{
    nvtxRangePushA( "allocating host-side image memory" );

    _input_image_h.allocHost( w, h, popsift::CudaAllocated );
    _input_image_d.allocDev( w, h );

    createTexture( );

    nvtxRangePop(); // "allocating host-side image memory"
}

void Image::destroyTexture( )
{
    cudaError_t err;
    err = cudaDestroyTextureObject( _input_image_tex );
    POP_CUDA_FATAL_TEST( err, "Could not destroy texture object: " );
}

void Image::createTexture( )
{
    /* initializing texture for upscaling
     */
    memset( &_input_image_texDesc, 0, sizeof(cudaTextureDesc) );
    _input_image_texDesc.normalizedCoords = 1; // address 0..1 instead of 0..width/height
    _input_image_texDesc.addressMode[0]   = cudaAddressModeClamp;
    _input_image_texDesc.addressMode[1]   = cudaAddressModeClamp;
    _input_image_texDesc.addressMode[2]   = cudaAddressModeClamp;
    _input_image_texDesc.readMode         = cudaReadModeNormalizedFloat; // automatic conversion from uchar to float
    _input_image_texDesc.filterMode       = cudaFilterModeLinear; // bilinear interpolation
    // _input_image_texDesc.filterMode       = cudaFilterModePoint; // nearest neighbour mode

    memset( &_input_image_resDesc, 0, sizeof(cudaResourceDesc) );
    _input_image_resDesc.resType                  = cudaResourceTypePitch2D;
    _input_image_resDesc.res.pitch2D.devPtr       = _input_image_d.data;
    _input_image_resDesc.res.pitch2D.desc.f       = cudaChannelFormatKindUnsigned;
    _input_image_resDesc.res.pitch2D.desc.x       = 8; // sizeof(uint8_t)*8
    _input_image_resDesc.res.pitch2D.desc.y       = 0;
    _input_image_resDesc.res.pitch2D.desc.z       = 0;
    _input_image_resDesc.res.pitch2D.desc.w       = 0;
    assert( _input_image_d.elemSize() == 1 );
    _input_image_resDesc.res.pitch2D.pitchInBytes = _input_image_d.getPitchInBytes();
    _input_image_resDesc.res.pitch2D.width        = _input_image_d.getCols();
    _input_image_resDesc.res.pitch2D.height       = _input_image_d.getRows();

    cudaError_t err;
    err = cudaCreateTextureObject( &_input_image_tex, &_input_image_resDesc, &_input_image_texDesc, 0 );
    POP_CUDA_FATAL_TEST( err, "Could not create texture object: " );
}

/*************************************************************
 * ImageFloat
 *************************************************************/

ImageFloat::ImageFloat( )
    : ImageBase( 0, 0 )
{
}

ImageFloat::ImageFloat( int w, int h )
    : ImageBase( w, h )
{
    allocate( w, h );
}

ImageFloat::~ImageFloat( )
{
    if( _max_w == 0 ) return;

    destroyTexture( );
    _input_image_d.freeDev( );
    _input_image_h.freeHost( popsift::CudaAllocated );
}

void ImageFloat::load( void* input )
{
    /* The host memcpy may seem like a really stupid idea, but _input_image_h
     * is in CUDA-allocated pinned host memory, which makes the H2D copy
     * much faster.
     */
    memcpy( _input_image_h.data, input, _w*_h*sizeof(float) ); // assume that host Plane2D has no pitch
    _input_image_h.memcpyToDevice( _input_image_d );
}

void ImageFloat::resetDimensions( int w, int h )
{
    if( _max_w == 0 && _max_h == 0 ) {
        _max_w = _w = w;
        _max_h = _h = h;
        allocate( w, h );
        return;
    }

    if( w == _w && h == _h ) return;
        /* everything OK, nothing to do */

    _w = w;
    _h = h;

    if( w <= _max_w && h <= _max_h ) {
        _input_image_h.resetDimensionsHost( w, h );
        _input_image_d.resetDimensionsDev( w, h );

        destroyTexture( );
        createTexture( );
    } else {
        nvtxRangePushA( "reallocating host-side image memory" );

        _max_w = max( w, _max_w );
        _max_h = max( h, _max_h );
        _input_image_h.freeHost( popsift::CudaAllocated );
        _input_image_d.freeDev( );
        _input_image_h.allocHost( _max_w, _max_h, popsift::CudaAllocated );
        _input_image_d.allocDev(  _max_w, _max_h );
        _input_image_h.resetDimensionsHost( w, h );
        _input_image_d.resetDimensionsDev( w, h );

        destroyTexture( );
        createTexture( );

        nvtxRangePop(); // "reallocating host-side image memory"
    }
}

void ImageFloat::allocate( int w, int h )
{
    nvtxRangePushA( "allocating host-side image memory" );

    _input_image_h.allocHost( w, h, popsift::CudaAllocated );
    _input_image_d.allocDev( w, h );

    createTexture( );

    nvtxRangePop(); // "allocating host-side image memory"
}

void ImageFloat::destroyTexture( )
{
    cudaError_t err;
    err = cudaDestroyTextureObject( _input_image_tex );
    POP_CUDA_FATAL_TEST( err, "Could not destroy texture object: " );
}

void ImageFloat::createTexture( )
{
    /* initializing texture for upscaling
     */
    memset( &_input_image_texDesc, 0, sizeof(cudaTextureDesc) );
    _input_image_texDesc.normalizedCoords = 1; // address 0..1 instead of 0..width/height
    _input_image_texDesc.addressMode[0]   = cudaAddressModeClamp;
    _input_image_texDesc.addressMode[1]   = cudaAddressModeClamp;
    _input_image_texDesc.addressMode[2]   = cudaAddressModeClamp;
    _input_image_texDesc.readMode         = cudaReadModeElementType; // no conversion
    _input_image_texDesc.filterMode       = cudaFilterModeLinear; // bilinear interpolation
    // _input_image_texDesc.filterMode       = cudaFilterModePoint; // nearest neighbour mode

    memset( &_input_image_resDesc, 0, sizeof(cudaResourceDesc) );
    _input_image_resDesc.resType                  = cudaResourceTypePitch2D;
    _input_image_resDesc.res.pitch2D.devPtr       = _input_image_d.data;
    _input_image_resDesc.res.pitch2D.desc.f       = cudaChannelFormatKindFloat;
    _input_image_resDesc.res.pitch2D.desc.x       = 32; // sizeof(float)*8
    _input_image_resDesc.res.pitch2D.desc.y       = 0;
    _input_image_resDesc.res.pitch2D.desc.z       = 0;
    _input_image_resDesc.res.pitch2D.desc.w       = 0;
    assert( _input_image_d.elemSize() == 4 );
    _input_image_resDesc.res.pitch2D.pitchInBytes = _input_image_d.getPitchInBytes();
    _input_image_resDesc.res.pitch2D.width        = _input_image_d.getCols();
    _input_image_resDesc.res.pitch2D.height       = _input_image_d.getRows();

    cudaError_t err;
    err = cudaCreateTextureObject( &_input_image_tex, &_input_image_resDesc, &_input_image_texDesc, 0 );
    POP_CUDA_FATAL_TEST( err, "Could not create texture object: " );
}

} // namespace popsift

