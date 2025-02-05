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
    _input_image_d.alloc( w, h );
}

Image::~Image( )
{
    if( _max_w == 0 ) return;

    _input_image_d.dealloc( );
}

void Image::load( void* input )
{
    /* We copy the input because it would be necessary for CUDA.
     * Eliminate eventually.
     */
    _input_image_d.copyToPlane( input );
}

void Image::resetDimensions( int w, int h )
{
    _input_image_d.resetDimensions( w, h );
}

void Image::allocate( int w, int h )
{
    _input_image_d.alloc( w, h );
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
    _input_image_d.alloc( w, h );
}

ImageFloat::~ImageFloat( )
{
    if( _max_w == 0 ) return;

    _input_image_d.dealloc( );
}

void ImageFloat::load( void* input )
{
    _input_image_d.copyToPlane( input );
}

void ImageFloat::resetDimensions( int w, int h )
{
    _input_image_d.resetDimensions( w, h );
}

void ImageFloat::allocate( int w, int h )
{
    _input_image_d.alloc( w, h );
}

} // namespace popsift

