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
    allocate( w, h );
}

Image::~Image( )
{
    if( _max_w == 0 ) return;

    _input_image_d.freeDev( );
}

void Image::load( void* input )
{
    /* We copy the input because it would be necessary for CUDA.
     * Eliminate eventually.
     */
    memcpy( _input_image_d.data, input, _w*_h );
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
        _input_image_d.resetDimensionsDev( w, h );
    } else {
        _max_w = max( w, _max_w );
        _max_h = max( h, _max_h );
        _input_image_d.freeDev( );
        _input_image_d.allocDev(  _max_w, _max_h );
        _input_image_d.resetDimensionsDev( w, h );
    }
}

void Image::allocate( int w, int h )
{
    _input_image_d.allocDev( w, h );
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
}

void ImageFloat::load( void* input )
{
    memcpy( _input_image_d.data, input, _w*_h*sizeof(float) );
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
        _input_image_d.resetDimensionsDev( w, h );
    } else {
        _max_w = max( w, _max_w );
        _max_h = max( h, _max_h );
        _input_image_d.freeDev( );
        _input_image_d.allocDev(  _max_w, _max_h );
        _input_image_d.resetDimensionsDev( w, h );
    }
}

void ImageFloat::allocate( int w, int h )
{
    _input_image_d.allocDev( w, h );
}

} // namespace popsift

