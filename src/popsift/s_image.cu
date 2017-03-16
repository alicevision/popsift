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

using namespace std;

namespace popsift {

Image::Image( )
    : _w(0), _h(0)
    , _max_w(0), _max_h(0)
{
}

Image::Image( int w, int h )
    : _w(w), _h(h)
    , _max_w(w), _max_h(h)
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

void Image::load( const unsigned char* input )
{
    /* The host memcpy may seem like a really stupid idea, but _input_image_h
     * is in CUDA-allocated pinned host memory, which makes the H2D copy
     * much faster.
     */
    memcpy( _input_image_h.data, input, _w*_h );
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
        _input_image_h.resetDimensions( w, h );
        _input_image_d.resetDimensions( w, h );

        destroyTexture( );
        createTexture( );
    } else {
        _max_w = max( w, _max_w );
        _max_h = max( h, _max_h );
        _input_image_h.freeHost( popsift::CudaAllocated );
        _input_image_d.freeDev( );
        _input_image_h.allocHost( _max_w, _max_h, popsift::CudaAllocated );
        _input_image_d.allocDev(  _max_w, _max_h );
        _input_image_h.resetDimensions( w, h );
        _input_image_d.resetDimensions( w, h );

        destroyTexture( );
        createTexture( );
    }
}

void Image::allocate( int w, int h )
{
    _input_image_h.allocHost( w, h, popsift::CudaAllocated );
    _input_image_d.allocDev( w, h );

    createTexture( );
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
    _input_image_resDesc.res.pitch2D.pitchInBytes = _input_image_d.step;
    _input_image_resDesc.res.pitch2D.width        = _input_image_d.getCols();
    _input_image_resDesc.res.pitch2D.height       = _input_image_d.getRows();

    cudaError_t err;
    err = cudaCreateTextureObject( &_input_image_tex, &_input_image_resDesc, &_input_image_texDesc, 0 );
    POP_CUDA_FATAL_TEST( err, "Could not create texture object: " );
}

} // namespace popsift

