/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "sift_constants.h"
#include "sift_conf.h"

#define SUPPORT_ABSOLUTE_SIGMA

namespace popart {

struct GaussInfo
{
    int required_filter_stages;

    /* If initial blur is used, then this is the 1D Gauss table
     * for blurring the remaining amount up to the blur value
     * of sigma0
     */
    float filter_initial_blur[ GAUSS_ALIGN ];

    /* These are the 1D Gauss tables for all levels, starting
     * sigma0 (always, even if sigma0 remain unused.
     */
    float filter[ GAUSS_ALIGN * GAUSS_LEVELS ];

#ifdef SUPPORT_ABSOLUTE_SIGMA
    /* An experimental 1D Gauss table. The idea is to blur
     * directly level 1. Level 1 is more suitable than level 0
     * because level 0 has special handling in octave 0.
     * For ease of use, level 0 exists in the table but is
     * initialized to 0.
     */
    float from_lvl_1[ GAUSS_ALIGN * GAUSS_LEVELS ];

    float abs_sigma[ GAUSS_LEVELS ];
#endif // SUPPORT_ABSOLUTE_SIGMA

    /* The sigma used to generate the Gauss table for each level.
     * Meaning these are the differences between sigma0 and sigmaN.
     */
    float sigma[ GAUSS_LEVELS ];

    /* The sigma value that is the difference between the assumed
     * initial blur (given blur of the input) and sigma0. Used to
     * generated filter_initial_blur.
     */
    float initial_sigma;

    /* The span of the table that is generated for each level.
     */
    int span[ GAUSS_LEVELS ];

#ifdef SUPPORT_ABSOLUTE_SIGMA
    /* Equivalent to span for from_lvl_1 tables.
     */
    int   abs_span[ GAUSS_LEVELS ];
#endif // SUPPORT_ABSOLUTE_SIGMA

    /* The span of the table that is generated for initial blur.
     */
    int initial_span;

    __host__
    void clearTables( );

    __host__
    void computeInitialBlurTable( int span, float sigma );

    __host__
    void computeBlurTable( int level, int span, float sigma );

    __host__
    void computeAbsBlurTable( int level, int span, float sigma );

    __host__
    static int vlFeatSpan( float sigma );

    __host__
    static int openCVSpan( float sigma );
};

extern __device__ __constant__ GaussInfo d_gauss;

void init_filter( const Config& conf,
                  float         sigma0,
                  int           levels );

} // namespace popart

