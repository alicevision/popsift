/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common/debug_macros.h"
#include "gauss_filter.h"

#include <algorithm>
#include <cstdio>

using namespace std;

namespace popsift {

__device__ __constant__
GaussInfo d_gauss;

__align__(128) thread_local GaussInfo h_gauss;


__global__
void print_gauss_filter_symbol( int columns )
{
    printf( "\n"
            "Gauss tables\n"
            "      level span sigma : center value -> edge value\n"
            "    relative sigma\n" );

    for( int lvl=0; lvl<d_gauss.required_filter_stages; lvl++ ) {
        int span = d_gauss.inc.span[lvl] + d_gauss.inc.span[lvl] - 1;

        printf("      %d %d ", lvl, span );
        printf("%2.6f: ", d_gauss.inc.sigma[lvl] );
        int m = min( d_gauss.inc.span[lvl], columns );
        for( int x=0; x<m; x++ ) {
            printf("%0.8f ", d_gauss.inc.filter[lvl*GAUSS_ALIGN+x] );
        }
        if( m < d_gauss.inc.span[lvl] )
            printf("...\n");
        else
            printf("\n");
    }
    printf("\n");

    printf( "\n"
            "Gauss tables for hardware interpolation\n"
            "      level span sigma : center value -> ( interpolation value, multiplier ) [one edge value] \n" );

    for( int lvl=0; lvl<d_gauss.required_filter_stages; lvl++ ) {
        int span = d_gauss.inc.i_span[lvl] + d_gauss.inc.i_span[lvl] - 1;

        printf("      %d %d ", lvl, span );
        printf("%2.6f: ", d_gauss.inc.sigma[lvl] );
        int m = min( d_gauss.inc.i_span[lvl], columns );
        for( int x=0; x<m; x++ ) {
            printf("%0.8f ", d_gauss.inc.i_filter[lvl*GAUSS_ALIGN+x] );
        }
        if( m < d_gauss.inc.i_span[lvl] )
            printf("...\n");
        else
            printf("\n");
    }
    printf("\n");

    printf( "\n"
            "Gauss tables\n"
            "    level 0-filters for direct downscaling\n");

    for( int lvl=0; lvl<MAX_OCTAVES; lvl++ ) {
        int span = d_gauss.dd.span[lvl] + d_gauss.dd.span[lvl] - 1;

        printf("      %d %d %2.6f: ", lvl, span, d_gauss.dd.sigma[lvl] );
        int m = min( d_gauss.dd.span[lvl], columns );
        for( int x=0; x<m; x++ ) {
            printf("%0.8f ", d_gauss.dd.filter[lvl*GAUSS_ALIGN+x] );
        }
        if( m < d_gauss.dd.span[lvl] )
            printf("...\n");
        else
            printf("\n");
    }
    printf("\n");
}

/*************************************************************
 * Initialize the Gauss filter table in constant memory
 *************************************************************/

void init_filter( const Config& conf,
                  float         sigma0,
                  int           levels )
{
    if( sigma0 > 2.0 )
    {
        stringstream ss;
        ss << "ERROR: "
           << " Sigma > 2.0 is not supported. Re-size __constant__ array and recompile.";
        POP_FATAL(ss.str());
    }
    if( levels > GAUSS_LEVELS )
    {
        stringstream ss;
        ss << "ERROR: "
           << " More than " << GAUSS_LEVELS << " levels not supported. Re-size __constant__ array and recompile.";
        POP_FATAL(ss.str());
    }

    if( conf.ifPrintGaussTables() ) {
        printf( "\n"
                "Upscaling factor: %f (i.e. original image is scaled by a factor of %f)\n"
                "\n"
                "Sigma computations\n"
                "    Initial sigma is %f\n"
                "    Input blurriness is assumed to be %f (scaled to %f)\n"
                ,
                conf.getUpscaleFactor(),
                pow( 2.0f, conf.getUpscaleFactor() ),
                sigma0,
                conf.getInitialBlur(),
                conf.getInitialBlur() * pow( 2.0f, conf.getUpscaleFactor() )
                );
        // printf("sigma is initially sigma0, afterwards the difference between previous 2 sigmas\n");
    }

    h_gauss.setSpanMode( conf.getGaussMode() );

    h_gauss.clearTables();

    h_gauss.required_filter_stages = levels + 3;

    const float initial_blur = conf.hasInitialBlur()
                             ? conf.getInitialBlur() * pow( 2.0f, conf.getUpscaleFactor() )
                             : 0.0f;

    /* inc :
     * The classical Gaussian blur tables for incremental blurring.
     * These do not rely on hardware interpolation.
     */
    h_gauss.inc.sigma[0] = conf.hasInitialBlur()
                         ? sqrt( fabsf( sigma0 * sigma0 - initial_blur * initial_blur ) )
                         : sigma0;

    for( int lvl=1; lvl<h_gauss.required_filter_stages; lvl++ ) {
        const float sigmaP = sigma0 * pow( 2.0f, (float)(lvl-1)/(float)levels );
        const float sigmaS = sigma0 * pow( 2.0f, (float)(lvl  )/(float)levels );

        h_gauss.inc.sigma[lvl] = sqrt( sigmaS * sigmaS - sigmaP * sigmaP );
    }

    h_gauss.inc.computeBlurTable( &h_gauss );

    /* dd :
     * The direct-downscaling kernels make use of the assumption that downscaling
     * from MAX_LEVEL-3 is identical to applying 2*sigma on the identical image
     * before downscaling, which would be identical to applying 1*sigma after
     * downscaling.
     * In reality, this is not true because images are not continuous, but we
     * support the options because it is interesting. Perhaps it works for the later
     * octaves, where it is also good for performance.
     * dd is only for creating level 0 of all octave directly from the input image.
     */

    // subtract initial blur
    const float b = sqrt( fabs( sigma0 * sigma0 - initial_blur * initial_blur ) );

    // sigma / 2^i
    h_gauss.dd.sigma[0] = b;
    h_gauss.dd.computeBlurTable( &h_gauss );

    cudaError_t err;
    err = cudaMemcpyToSymbol( d_gauss,
                              &h_gauss,
                              sizeof(GaussInfo),
                              0,
                              cudaMemcpyHostToDevice );
    POP_CUDA_FATAL_TEST( err, "cudaMemcpyToSymbol failed for Gauss kernel initialization: " );

    if( conf.ifPrintGaussTables() ) {
        print_gauss_filter_symbol
            <<<1,1>>>
            ( 10 );

        POP_SYNC_CHK;

        err = cudaGetLastError();
        POP_CUDA_FATAL_TEST( err, "Gauss Symbol info failed: " );
    }
}

__host__
void GaussInfo::clearTables( )
{
    inc            .clearTables();
    dd             .clearTables();
}

__host__
void GaussInfo::setSpanMode( Config::GaussMode m )
{
    _span_mode = m;
}

__host__
int GaussInfo::getSpan( float sigma ) const
{
    /* This is the VLFeat computation for choosing the Gaussian filter width.
     * In our case, we look at the half-sided filter including the center value.
     */
    return std::min<int>( ceilf( 4.0f * sigma ) + 1, GAUSS_ALIGN - 1 );
}

template<int LEVELS>
__host__
void GaussTable<LEVELS>::clearTables( )
{
    for( int i=0; i<GAUSS_ALIGN * LEVELS; i++ ) {
        filter[i]   = 0.0f;
        i_filter[i] = 0.0f;
    }
}

template<int LEVELS>
__host__
void GaussTable<LEVELS>::computeBlurTable( const GaussInfo* info )
{
    for( int level=0; level<LEVELS; level++ ) {
        span[level] = min( info->getSpan( sigma[level] ), GAUSS_ALIGN-1 );
    }

    for( int level=0; level<LEVELS; level++ ) {
        /* Should be:
         * kernel[x] = exp( -0.5 * (pow((x-mean)/sigma, 2.0) ) )
         *           / sqrt(2 * M_PI * sigma * sigma);
         * but the denominator is constant and we divide by sum anyway
         */
        const float sig = sigma[level];
        const int   spn = span[level];
        double sum = 1.0;
        filter[level*GAUSS_ALIGN + 0] = 1.0;
        for( int x = 1; x < spn; x++ ) {
            const float val = exp( -0.5 * (pow( double(x)/sig, 2.0) ) );
            filter[level*GAUSS_ALIGN + x] = val;
            sum += 2.0f * val;
        }
        for( int x = 0; x < spn; x++ ) {
            filter[level*GAUSS_ALIGN + x] /= sum;
        }
        for( int x = spn; x < GAUSS_ALIGN; x++ ) {
            filter[level*GAUSS_ALIGN + x] = 0;
        }
    }

    transformBlurTable();
}

template<int LEVELS>
__host__
void GaussTable<LEVELS>::transformBlurTable( )
{
    for( int level=0; level<LEVELS; level++ ) {
        i_span[level] = span[level];
        if( ! ( i_span[level] & 1 ) ) {
            i_span[level] += 1;
        }
    }

    for( int level=0; level<LEVELS; level++ ) {
        /* We want to use the hardware linear interpolation for one
         * multiplication, reducing software multiplications to half
         *
         * ax + by = v * ( ux + (1-u)y )
         * u = aa + ab
         * v = 1/(a+b)
         */
        const int   spn = i_span[level];
        for( int x = 1; x < spn; x += 2 ) {
            float a = filter[level*GAUSS_ALIGN + x];
            float b = filter[level*GAUSS_ALIGN + x + 1];
            float u = a / (a+b);
            float v = a+b;
            i_filter[level*GAUSS_ALIGN + x]     = u; // ratios are odd
            i_filter[level*GAUSS_ALIGN + x + 1] = v; // multipliers are even
        }

        // center stays the same
        i_filter[level*GAUSS_ALIGN] = filter[level*GAUSS_ALIGN];

        // outside of span is 0
        for( int x = spn; x < GAUSS_ALIGN; x++ ) {
            i_filter[level*GAUSS_ALIGN + x] = 0;
        }
    }
}

} // namespace popsift

