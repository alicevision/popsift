/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/assist.h"
#include "common/debug_macros.h"
#include "s_image.h"
#include "sift_config.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>

using namespace std;

namespace popsift {

/*************************************************************
 * Image
 *************************************************************/

Image::Image( )
{ }

Image::Image( int w, int h )
{
    allocate( w, h );
}

Image::~Image( )
{
    _input_image_d.dealloc( );
    _hidden_conversion.dealloc( );
}

void Image::load( const void* input )
{
    /* We copy the input because it would be necessary for CUDA.
     * Eliminate eventually.
     */
    _input_image_d.memcpyFromBuffer( reinterpret_cast<const uint8_t*>(input) );

#if 1   // DEBUG PGM LOADING
    popsift::write_plane2D( "input-in-bytes.pgm", _input_image_d );
#endif
}

void Image::resetDimensions( int w, int h )
{
    _input_image_d.resetDimensions( w, h );
}

void Image::allocate( int w, int h )
{
    _input_image_d.alloc( w, h );
}

Plane2D_float& Image::getFloatPlane()
{
    sycl::queue q;  // Temporary queue for backward compatibility
    return getFloatPlane(q);
}

Plane2D_float& Image::getFloatPlane(sycl::queue& q)
{
    if( _hidden_conversion.isNull() )
    {
        _hidden_conversion.alloc( getWidth(), getHeight() );

        const int width = getWidth();
        const int height = getHeight();
        const int src_pitch = _input_image_d.getPitch();
        const int dst_pitch = _hidden_conversion.getPitch();
        
        // Simple CPU conversion (original code - works reliably)
        uint8_t* src_host = _input_image_d.getHostPtr();
        float* dst_host = _hidden_conversion.getHostPtr();
        
        for( int y = 0; y < height; y++ ) {
            for( int x = 0; x < width; x++ ) {
                uint8_t pixel = src_host[y * src_pitch + x];
                float f = (pixel - 0.5f) / 255.0f;
                dst_host[y * dst_pitch + x] = f;
            }
        }
    }
    return _hidden_conversion;
}

/*************************************************************
 * ImageFloat
 *************************************************************/

ImageFloat::ImageFloat( )
{ }

ImageFloat::ImageFloat( int w, int h )
{
    allocate( w, h );
}

ImageFloat::~ImageFloat( )
{
    _input_image_d.dealloc( );
}

void ImageFloat::load( const void* input )
{
    _input_image_d.memcpyFromBuffer( reinterpret_cast<const float*>(input) );
}

void ImageFloat::resetDimensions( int w, int h )
{
    _input_image_d.resetDimensions( w, h );
}

void ImageFloat::allocate( int w, int h )
{
    _input_image_d.alloc( w, h );
}

Plane2D_float& ImageFloat::getFloatPlane()
{
    return _input_image_d;
}

Plane2D_float& ImageFloat::getFloatPlane(sycl::queue& q)
{
    return _input_image_d;  // Already float, no conversion needed
}

} // namespace popsift

