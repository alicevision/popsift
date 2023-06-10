/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "sift_conf.h"
#include "sift_constants.h"

namespace popsift {

struct GaussInfo;

template<int LEVELS>
struct GaussTable
{
    /* The filter that is computed from the sigma values of this level */
    float filter[ LEVELS * GAUSS_ALIGN ];

    /* The same filter as above, but recomputed for use with hardware
     * interpolation to implement half of the multiplications as hardware
     * access */
    float i_filter[ LEVELS * GAUSS_ALIGN ];

    /* The sigma used to generate the Gauss table for each level.
     * Meaning these are the differences between sigma0 and sigmaN.
     */
    float sigma [ LEVELS ];

    /* The span of the table that is generated for each level.  */
    int   span  [ LEVELS ];

    /* Alternative spans for i_filter, which must always be odd */
    int   i_span  [ LEVELS ];

    __host__
    void clearTables( );

    __host__
    void computeBlurTable( const GaussInfo* info );

private:
    __host__
    void transformBlurTable( ); // const GaussInfo* info );
};

struct GaussInfo
{
    int required_filter_stages;

    /* These are the 1D Gauss tables for all levels of an octave.
     * The first row is special:
     * - in octave 0 if initial blur is non-zero, contains the
     *   remaining blur that is required to reach sigma0
     * - in octave 0 if initial blur is zero, contains the
     *   filter for sigma0
     * - in all other octaves, row 0 is unused
     */
    GaussTable<GAUSS_LEVELS> inc;

    /* Compute the 1D Gauss tables for all levels of octave 0.
     * For octave 0, all of these tables derive from the input
     * image.
     */
    GaussTable<GAUSS_LEVELS> abs_o0;

    /* Compute the 1D Gauss tables for all levels of octaves 1 and up.
     * Level 0 is empty, since it is created by other means.
     * All other levels blur from level 0, not considering any
     * initial blur.
     */
    GaussTable<GAUSS_LEVELS> abs_oN;

    /* In theory, level 0 of octave 2 contains the same information
     * whether it is constructed by downscaling and blurring the
     * input image with sigma or by blurring the input image with 2*sigma
     * and downscaling afterwards.
     */
    GaussTable<MAX_OCTAVES> dd;

    __host__
    void clearTables( );

public:
    __host__
    void setSpanMode( Config::GaussMode m );

    __host__
    int getSpan( float sigma ) const;

private:
    Config::GaussMode _span_mode;

    __host__
    static int vlFeatSpan( float sigma );

    __host__
    static int vlFeatRelativeSpan( float sigma );

    __host__
    static int openCVSpan( float sigma );
};

extern __device__ __constant__ GaussInfo d_gauss;
extern thread_local            GaussInfo h_gauss;

/* init_filter must be called early to initialize the Gauss tables.
 */
void init_filter( const Config& conf,
                  float         sigma0,
                  int           levels );

} // namespace popsift

