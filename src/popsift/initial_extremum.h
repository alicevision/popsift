/*
 * Copyright 2021, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

// #include "sift_constants.h"

namespace popsift {

/**
 * @brief This is an internal data structure.
 * Separated from the final Extremum data structure to implement
 * grid filtering in a space-efficient manner. In grid filtering,
 * extrema are first found, after that some may be discarded in
 * some spatial regions of the image. Avoid waste of space by
 * allocating Extremum structures only for the remaining ones.
 */
class InitialExtremum
{
public:
    __device__ inline
    void set( const int& oct, const float& x, const float& y, const float& z, const int& cellidx )
    {
        ignore = false;
        octave = oct;
        xpos   = x;
        ypos   = y;
        if( z == 0 )
        {
            lpos  = 0;
            sigma = 0.0f;
        }
        else
        {
            lpos  = (int)roundf(z);
            sigma = d_consts.sigma0 * pow(d_consts.sigma_k, z);
        }

        cell = cellidx;
    }

    __device__ inline const int&   getOctave( ) const { return octave; }
    __device__ inline const float& getX( )      const { return xpos; }
    __device__ inline const float& getY( )      const { return ypos; }
    __device__ inline const int&   getLevel( )  const { return lpos; }
    __device__ inline const float& getSigma( )  const { return sigma; }
    __device__ inline const int&   getCell( )   const { return cell; }

    /** Compute the accurate floating-point level within the image
     *  pyramid. It does not compensate for up- and downscaling of the
     *  input image but returns the value relative to the first octave
     *  that is actually computed.
     * note: Online used once in grid filtering. Therefore not pre-computed.
     */
    __device__ inline float getScale( )  const {
        return sigma * powf( 2.0f, octave );
    }

    /** Set the ignore flag. Only used in grid filtering. */
    __host__ __device__ inline void setIgnore( ) { ignore = true; }

    /** Test the ignore flag. Only used in grid filtering. */
    __host__ __device__ inline bool isIgnored( ) const { return ignore; }

private:
    /// true if this extremum has been filtered
    bool  ignore;

    /** The octave where this point was found. */
    int   octave;

    float xpos;
    float ypos;

    /** This is the closest level of the octave for this initial
     *  extremum. It is rounded from the computed continuous level.
     */
    int   lpos;

    /** This is the accurate floating-point level within this octave.
     *  It requires a transformation of the "z" value that is derived
     *  from refining the z-position in the octave.
     */
    float sigma;

    /// index into the grid for grid-based extrema filtering
    int   cell;
};

} // namespace popsift
