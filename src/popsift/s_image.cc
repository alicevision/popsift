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
    if( _hidden_conversion.isNull() )
    {
        _hidden_conversion.alloc( getWidth(), getHeight() );

        for( int y=0; y<getHeight(); y++ )
        {
            for( int x=0; x<getWidth(); x++ )
            {
                uint8_t pixel = _input_image_d.get( y, x );
                
                const float f = (pixel - 0.5f) / 255.0f; 
                
                _hidden_conversion.set( y, x, f );
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

} // namespace popsift

