/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <stdio.h>
#include <algorithm>

#include "gauss_filter.h"
#include "common/debug_macros.h"

using namespace std;

namespace popsift {

__device__ __constant__
GaussInfo d_gauss;

__align__(128) GaussInfo h_gauss;


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
        int span = d_gauss.inc_relative.span[lvl] + d_gauss.inc_relative.span[lvl] - 1;

        printf("      %d %d ", lvl, span );
        printf("%2.6f: ", d_gauss.inc_relative.sigma[lvl] );
        int m = min( d_gauss.inc_relative.span[lvl], columns );
        for( int x=0; x<m; x++ ) {
            printf("%0.8f ", d_gauss.inc_relative.filter[lvl*GAUSS_ALIGN+x] );
        }
        if( m < d_gauss.inc_relative.span[lvl] )
            printf("...\n");
        else
            printf("\n");
    }
    printf("\n");

    printf( "\n"
            "Relative Gauss tables\n"
            "      level span sigma : center value -> edge value\n"
            "    relative sigma\n" );

    for( int lvl=0; lvl<d_gauss.required_filter_stages; lvl++ ) {
        int span = d_gauss.inc_relative.span[lvl] + d_gauss.inc_relative.span[lvl] - 1;

        printf("      %d %d ", lvl, span );
        printf("%2.6f: ", d_gauss.inc_relative.sigma[lvl] );
        int m = min( d_gauss.inc_relative.span[lvl], columns );
        for( int x=0; x<m; x++ ) {
            printf("%0.8f ", d_gauss.inc_relative.filter[lvl*GAUSS_ALIGN+x] );
        }
        if( m < d_gauss.inc_relative.span[lvl] )
            printf("...\n");
        else
            printf("\n");
    }
    printf("\n");

    printf("    absolute filters octave 0\n");

    for( int lvl=0; lvl<d_gauss.required_filter_stages; lvl++ ) {
        int span = d_gauss.abs_o0.span[lvl] + d_gauss.abs_o0.span[lvl] - 1;

        printf("      %d %d %2.6f: ", lvl, span, d_gauss.abs_o0.sigma[lvl] );
        int m = min( d_gauss.abs_o0.span[lvl], columns );
        for( int x=0; x<m; x++ ) {
            printf("%0.8f ", d_gauss.abs_o0.filter[lvl*GAUSS_ALIGN+x] );
        }
        if( m < d_gauss.abs_o0.span[lvl] )
            printf("...\n");
        else
            printf("\n");
    }
    printf("\n");

    printf("    absolute filters other octaves\n");

    for( int lvl=0; lvl<d_gauss.required_filter_stages; lvl++ ) {
        int span = d_gauss.abs_oN.span[lvl] + d_gauss.abs_oN.span[lvl] - 1;

        printf("      %d %d %2.6f: ", lvl, span, d_gauss.abs_oN.sigma[lvl] );
        int m = min( d_gauss.abs_oN.span[lvl], columns );
        for( int x=0; x<m; x++ ) {
            printf("%0.8f ", d_gauss.abs_oN.filter[lvl*GAUSS_ALIGN+x] );
        }
        if( m < d_gauss.abs_oN.span[lvl] )
            printf("...\n");
        else
            printf("\n");
    }
    printf("\n");

    printf("    level 0-filters for direct downscaling\n");

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
        cerr << __FILE__ << ":" << __LINE__ << ", ERROR: "
             << " Sigma > 2.0 is not supported. Re-size __constant__ array and recompile."
             << endl;
        exit( -__LINE__ );
    }
    if( levels > GAUSS_LEVELS )
    {
        cerr << __FILE__ << ":" << __LINE__ << ", ERROR: "
             << " More than " << GAUSS_LEVELS << " levels not supported. Re-size __constant__ array and recompile."
             << endl;
        exit( -__LINE__ );
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
                pow( 2.0, conf.getUpscaleFactor() ),
                sigma0,
                conf.getInitialBlur(),
                conf.getInitialBlur() * pow( 2.0, conf.getUpscaleFactor() )
                );
        // printf("sigma is initially sigma0, afterwards the difference between previous 2 sigmas\n");
    }

    h_gauss.setSpanMode( conf.getGaussMode() );

    h_gauss.clearTables();

    h_gauss.required_filter_stages = levels + 3;

    if( not conf.hasInitialBlur() ) {
        h_gauss.inc.sigma[0] = sigma0;
    } else {
        const float initial_blur = conf.getInitialBlur() * pow( 2.0, conf.getUpscaleFactor() );
        h_gauss.inc.sigma[0] = sqrt( fabsf( sigma0 * sigma0 - initial_blur * initial_blur ) );
    }

    for( int lvl=1; lvl<h_gauss.required_filter_stages; lvl++ ) {
        const float sigmaP = sigma0 * pow( 2.0, (float)(lvl-1)/(float)levels );
        const float sigmaS = sigma0 * pow( 2.0, (float)(lvl  )/(float)levels );

        h_gauss.inc.sigma[lvl] = sqrt( sigmaS * sigmaS - sigmaP * sigmaP );
    }

    h_gauss.inc.computeBlurTable( &h_gauss );

    float initial_blur = 0.0f;
    if( conf.hasInitialBlur() ) {
        initial_blur = conf.getInitialBlur() * pow( 2.0, conf.getUpscaleFactor() );
    }
    for( int lvl=0; lvl<h_gauss.required_filter_stages; lvl++ ) {
        const float sigmaS = sigma0 * pow( 2.0, (float)(lvl)/(float)levels );
        h_gauss.abs_o0.sigma[lvl]  = sqrt( fabs( sigmaS * sigmaS - initial_blur * initial_blur ) );
    }

    h_gauss.abs_o0.computeBlurTable( &h_gauss );

    if( not conf.hasInitialBlur() ) {
        h_gauss.inc_relative.sigma[0] = sigma0;
    } else {
        const float initial_blur = conf.getInitialBlur() * pow( 2.0, conf.getUpscaleFactor() );
        h_gauss.inc_relative.sigma[0] = sqrt( fabsf( sigma0 * sigma0 - initial_blur * initial_blur ) );
    }

    for( int lvl=1; lvl<h_gauss.required_filter_stages; lvl++ ) {
        const float sigmaP = sigma0 * pow( 2.0, (float)(lvl-1)/(float)levels );
        const float sigmaS = sigma0 * pow( 2.0, (float)(lvl  )/(float)levels );

        h_gauss.inc_relative.sigma[lvl] = sqrt( sigmaS * sigmaS - sigmaP * sigmaP );
    }

    h_gauss.inc_relative.computeBlurTable( &h_gauss );
    h_gauss.inc_relative.transformBlurTable( &h_gauss );

#if 0
    if( conf.ifPrintGaussTables() ) {
        for( int lvl=0; lvl<h_gauss.required_filter_stages; lvl++ ) {
            float sigmaP = sigma0 * pow( 2.0, (float)(lvl-1)/(float)levels );
            float sigmaS = sigma0 * pow( 2.0, (float)(lvl  )/(float)levels );
            if( lvl == 0 ) {
                sigmaP = conf.getInitialBlur() * pow( 2.0, conf.getUpscaleFactor() );
            }
            printf("    Sigma (rel) for level %d: %2.6f = sqrt(sigmaS(%2.6f)^2 - sigmaP(%2.6f)^2)\n", lvl, h_gauss.inc.sigma[lvl], sigmaS, sigmaP );
        }

        for( int lvl=0; lvl<h_gauss.required_filter_stages; lvl++ ) {
            const float sigmaS = sigma0 * pow( 2.0, (float)(lvl)/(float)levels );
            printf("    Sigma (abs0) for level %d: %2.6f = sqrt(sigmaS(%2.6f)^2 - sigmaP(%2.6f)^2)\n", lvl, h_gauss.abs_o0.sigma[lvl], sigmaS, initial_blur );
        }
    }
#endif

    for( int lvl=0; lvl<h_gauss.required_filter_stages; lvl++ ) {
        const float sigmaS = sigma0 * pow( 2.0, (float)(lvl)/(float)levels );
        h_gauss.abs_oN.sigma[lvl]  = sigmaS;
    }
    h_gauss.abs_oN.computeBlurTable( &h_gauss );

    for( int oct=0; oct<MAX_OCTAVES; oct++ ) {
        // sigma * 2^i
        float oct_sigma = scalbnf( sigma0, oct );

        // subtract initial blur
        float b = sqrt( fabs( oct_sigma * oct_sigma - initial_blur * initial_blur ) );

        // sigma / 2^i
        h_gauss.dd.sigma[oct] = scalbnf( b, -oct );
        h_gauss.dd.computeBlurTable( &h_gauss );
    }

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
        err = cudaGetLastError();
        POP_CUDA_FATAL_TEST( err, "Gauss Symbol info failed: " );
    }
}

__host__
void GaussInfo::clearTables( )
{
    inc         .clearTables();
    inc_relative.clearTables();
    abs_o0      .clearTables();
    abs_oN      .clearTables();
    dd          .clearTables();
}

__host__
void GaussInfo::setSpanMode( Config::GaussMode m )
{
    _span_mode = m;
}

__host__
int GaussInfo::getSpan( float sigma ) const
{
    switch( _span_mode )
    {
    case Config::VLFeat_Compute :
        return GaussInfo::vlFeatSpan( sigma );
    case Config::VLFeat_Relative :
        return GaussInfo::vlFeatRelativeSpan( sigma );
    case Config::OpenCV_Compute :
        return GaussInfo::openCVSpan( sigma );
    case Config::Fixed9 :
        return 5;
    case Config::Fixed15 :
        return 8;
    default :
        cerr << __FILE__ << ":" << __LINE__ << ", ERROR: "
             << " The mode for computing Gauss filter scan is invalid"
             << endl;
        exit( -__LINE__ );
    }
}

__host__
int GaussInfo::vlFeatSpan( float sigma )
{
    /* This is the VLFeat computation for choosing the Gaussian filter width.
     * In our case, we look at the half-sided filter including the center value.
     */
    return std::min<int>( ceilf( 4.0f * sigma ) + 1, GAUSS_ALIGN - 1 );
}

__host__
int GaussInfo::vlFeatRelativeSpan( float sigma )
{
    /* We want the width of the VLFeat computation, but always the next equal
     * or larger odd span, because we need pairs of weights.
     */
    int spn = vlFeatSpan( sigma );
    if( ( spn & 1 ) == 0 ) spn += 1;
    return spn;
}

__host__
int GaussInfo::openCVSpan( float sigma )
{
    int span = int( roundf( 2.0f * 4.0f * sigma + 1.0f ) ) | 1;
    span >>= 1;
    span  += 1;
    return std::min<int>( span, GAUSS_ALIGN - 1 );
}

template<int LEVELS>
__host__
void GaussTable<LEVELS>::clearTables( )
{
    for( int i=0; i<GAUSS_ALIGN * LEVELS; i++ ) {
        filter[i] = 0.0f;
    }
}

template<int LEVELS>
__host__
void GaussTable<LEVELS>::computeBlurTable( const GaussInfo* info )
{
    for( int level=0; level<LEVELS; level++ ) {
        span[level] = info->getSpan( sigma[level] );
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
    }
}

template<int LEVELS>
__host__
void GaussTable<LEVELS>::transformBlurTable( const GaussInfo* info )
{
    for( int level=0; level<LEVELS; level++ ) {
        span[level] = info->getSpan( sigma[level] );
        if( not ( span[level] & 1 ) ) {
            span[level] -= 1;
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
        const int   spn = span[level];
        for( int x = 1; x < spn; x += 2 ) {
            float a = filter[level*GAUSS_ALIGN + x];
            float b = filter[level*GAUSS_ALIGN + x + 1];
            float u = a / (a+b);
            float v = a+b;
            filter[level*GAUSS_ALIGN + x]     = u; // ratios are odd
            filter[level*GAUSS_ALIGN + x + 1] = v; // multipliers are even
        }
    }
}

} // namespace popsift

