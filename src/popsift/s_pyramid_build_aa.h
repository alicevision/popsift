/*
 * Copyright 2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cuda_runtime.h>

#include "common/plane_2d.h"

namespace popsift {
namespace gauss {

/*********************************************************************************
 * AbsoluteSource
 *
 * @desc Class for Gaussian filtering.
 *  This class uses absolute source coordinates.
 *  The source texture is in point mode (no interpolation between point values).
 *  The Gaussian filter values are suitable for incremental filtering.
 *  Each cell in the Gaussian filter provides the weight of one pixel.
 *********************************************************************************/
class AbsoluteSource
{
    int          _width;
    int          _height;
    cudaStream_t _stream;
public:
    AbsoluteSource( int width, int height, cudaStream_t stream )
        : _width( width )
        , _height( height )
        , _stream( stream )
    { }

    /** 1D kernel for a horizontal sweep.
     *  @param src a 3D texture that is created with the following modes:
     *             coordinates are _not_ normalized to range [0:1]
     *             point values are read as floats normalized to [0:1]
     *             values are read from nearest point (not interpolated)
     *  @param dst A 3D CUDA Surface backed by a CUDA array
     *  @param level The third dimension to use for writing.
     *               The input is read from level-1.
     */
    void horiz( cudaTextureObject_t src,
                cudaSurfaceObject_t dst,
                int                 level );

    /** 1D kernel for a vertical sweep.
     *  @param src a 3D texture that is created with the following modes:
     *             coordinates are _not_ normalized to range [0:1]
     *             point values are read as floats normalized to [0:1]
     *             values are read from nearest point (not interpolated)
     *  @param dst A 3D CUDA Surface backed by a CUDA array
     *  @param level The third dimension to use for writing.
     *               The inp9ut is read from level.
     */
    void vert ( cudaTextureObject_t src,
                cudaSurfaceObject_t dst,
                int                 level );
};

/*********************************************************************************
 * AbsoluteSourceInterpolatedFilter
 *
 * @desc Class for Gaussian filtering.
 *  This class uses absolute source coordinates.
 *  The source texture is in linear interpolation mode.
 *  The Gaussian filter values are suitable for incremental filtering.
 *  Cells in the Gaussian filter alternate between a weight for attracting
 *  neighbouring pixels, and the weight they have in common.
 *  So, instead of computing pix[n]*f[n]+pix[n+1]*f[n+1]
 *  this filter computes (A*pix[n]+(1-A)*pix[n+1]*B,
 *  where A=f[n]/(f[n]+f[n+1]) and B=f[n]+f[n+1]
 *********************************************************************************/
class AbsoluteSourceInterpolatedFilter
{
    int          _width;
    int          _height;
    cudaStream_t _stream;
public:
    AbsoluteSourceInterpolatedFilter( int width, int height, cudaStream_t stream )
        : _width( width )
        , _height( height )
        , _stream( stream )
    { }

    /** @see AbsoluteSource::horiz
     */
    void horiz( cudaTextureObject_t src,
                cudaSurfaceObject_t dst,
                int                 level );

    /** @see AbsoluteSource::vert
     */
    void vert ( cudaTextureObject_t src,
                cudaSurfaceObject_t dst,
                int                 level );
};

/*********************************************************************************
 * AbsoluteSourceLevel0
 *
 * @desc Class for Gaussian filtering.
 *  This class uses absolute source coordinates.
 *  The source texture is in point mode (no interpolation between point values).
 *  The Gaussian filter values are suitable for absolute filtering from the first
 *  level of the octave.
 *********************************************************************************/
class AbsoluteSourceLevel0
{
    int          _width;
    int          _height;
    cudaStream_t _stream;
public:
    AbsoluteSourceLevel0( int width, int height, cudaStream_t stream )
        : _width( width )
        , _height( height )
        , _stream( stream )
    { }

    /** 1D kernel for a vertical sweep.
     *  @param src a 3D texture that is created with the following modes:
     *             coordinates are _not_ normalized to range [0:1]
     *             point values are read as floats normalized to [0:1]
     *             values are read from nearest point (not interpolated)
     *  @param dst A 3D CUDA Surface backed by a CUDA array
     *  @param level The third dimension to use for writing.
     *               The input is read from level.
     */
    void vert( cudaTextureObject_t src,
               cudaSurfaceObject_t dst,
               int                 level );

    /** 1D kernel for a vertical sweep.
     *  Works like the method vert applied to every level from
     *  start_level (inclusive) to max_level (exclusive).
     */
    void vert_all( cudaTextureObject_t src,
                   cudaSurfaceObject_t dst,
                   int                 start_level,
                   int                 max_level );
};

/*********************************************************************************
 * AbsoluteSourceInterpolatedFilterLevel0
 *
 * @desc Class for Gaussian filtering.
 *  This class uses absolute source coordinates.
 *  The source texture is in linear interpolation mode.
 *  The Gaussian filter values are suitable for absolute filtering from the first
 *  level of the octave.
 *  Cells in the Gaussian filter alternate between a weight for attracting
 *  neighbouring pixels, and the weight they have in common.
 *  So, instead of computing pix[n]*f[n]+pix[n+1]*f[n+1]
 *  this filter computes (A*pix[n]+(1-A)*pix[n+1]*B,
 *  where A=f[n]/(f[n]+f[n+1]) and B=f[n]+f[n+1]
 *********************************************************************************/
class AbsoluteSourceInterpolatedFilterLevel0
{
    int          _width;
    int          _height;
    cudaStream_t _stream;
public:
    AbsoluteSourceInterpolatedFilterLevel0( int width, int height, cudaStream_t stream )
        : _width( width )
        , _height( height )
        , _stream( stream )
    { }

    /** @see AbsoluteSourceLevel0::vert
     */
    void vert( cudaTextureObject_t src,
               cudaSurfaceObject_t dst,
               int                 level );

    /** @see AbsoluteSourceLevel0::vert_all
     */
    void vert_all( cudaTextureObject_t src,
                   cudaSurfaceObject_t dst,
                   int                 start_level,
                   int                 max_level );
};

} // namespace gauss
} // namespace popsift

