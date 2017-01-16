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
        int span = d_gauss.inc_span[lvl] + d_gauss.inc_span[lvl] - 1;

        printf("      %d %d ", lvl, span );
        printf("%2.6f: ", d_gauss.inc_sigma[lvl] );
        int m = min( d_gauss.inc_span[lvl], columns );
        for( int x=0; x<m; x++ ) {
            printf("%0.8f ", d_gauss.inc_filter[lvl*GAUSS_ALIGN+x] );
        }
        if( m < d_gauss.inc_span[lvl] )
            printf("...\n");
        else
            printf("\n");
    }
    printf("\n");

#ifdef SUPPORT_ABSOLUTE_SIGMA
    printf("    absolute filters octave 0\n");

    for( int lvl=0; lvl<d_gauss.required_filter_stages; lvl++ ) {
        int span = d_gauss.abs_span_o0[lvl] + d_gauss.abs_span_o0[lvl] - 1;

        printf("      %d %d %2.6f: ", lvl, span, d_gauss.abs_sigma_o0[lvl] );
        int m = min( d_gauss.abs_span_o0[lvl], columns );
        for( int x=0; x<m; x++ ) {
            printf("%0.8f ", d_gauss.abs_filter_o0[lvl*GAUSS_ALIGN+x] );
        }
        if( m < d_gauss.abs_span_o0[lvl] )
            printf("...\n");
        else
            printf("\n");
    }
    printf("\n");

    printf("    absolute filters other octaves\n");

    for( int lvl=0; lvl<d_gauss.required_filter_stages; lvl++ ) {
        int span = d_gauss.abs_span_oN[lvl] + d_gauss.abs_span_oN[lvl] - 1;

        printf("      %d %d %2.6f: ", lvl, span, d_gauss.abs_sigma_oN[lvl] );
        int m = min( d_gauss.abs_span_oN[lvl], columns );
        for( int x=0; x<m; x++ ) {
            printf("%0.8f ", d_gauss.abs_filter_oN[lvl*GAUSS_ALIGN+x] );
        }
        if( m < d_gauss.abs_span_oN[lvl] )
            printf("...\n");
        else
            printf("\n");
    }
    printf("\n");
#endif
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
        h_gauss.inc_sigma[0] = sigma0;
    } else {
        const float initial_blur = conf.getInitialBlur() * pow( 2.0, conf.getUpscaleFactor() );
        h_gauss.inc_sigma[0] = sqrt( fabsf( sigma0 * sigma0 - initial_blur * initial_blur ) );
    }

    for( int lvl=1; lvl<h_gauss.required_filter_stages; lvl++ ) {
        const float sigmaP = sigma0 * pow( 2.0, (float)(lvl-1)/(float)levels );
        const float sigmaS = sigma0 * pow( 2.0, (float)(lvl  )/(float)levels );

        h_gauss.inc_sigma[lvl] = sqrt( sigmaS * sigmaS - sigmaP * sigmaP );
    }

    for( int lvl=0; lvl<h_gauss.required_filter_stages; lvl++ ) {
        h_gauss.inc_span[lvl]  = h_gauss.getSpan( h_gauss.inc_sigma[lvl] );

        h_gauss.computeBlurTable( lvl, h_gauss.inc_span[lvl], h_gauss.inc_sigma[lvl] );

        if( conf.ifPrintGaussTables() ) {
            float sigmaP = sigma0 * pow( 2.0, (float)(lvl-1)/(float)levels );
            float sigmaS = sigma0 * pow( 2.0, (float)(lvl  )/(float)levels );
            if( lvl == 0 ) {
                sigmaP = conf.getInitialBlur() * pow( 2.0, conf.getUpscaleFactor() );
            }
            printf("    Sigma for level %d: %2.6f = sqrt(sigmaS(%2.6f)^2 - sigmaP(%2.6f)^2)\n", lvl, h_gauss.inc_sigma[lvl], sigmaS, sigmaP );
        }
    }

#ifdef SUPPORT_ABSOLUTE_SIGMA
    float initial_blur = 0.0f;
    if( conf.hasInitialBlur() ) {
        initial_blur = conf.getInitialBlur() * pow( 2.0, conf.getUpscaleFactor() );
    }
    for( int lvl=0; lvl<h_gauss.required_filter_stages; lvl++ ) {
        const float sigmaS = sigma0 * pow( 2.0, (float)(lvl)/(float)levels );
        h_gauss.abs_sigma_o0[lvl]  = sqrt( fabs( sigmaS * sigmaS - initial_blur * initial_blur ) );
        h_gauss.abs_span_o0[lvl]   = h_gauss.getSpan( h_gauss.abs_sigma_o0[lvl] );
        h_gauss.computeAbsBlurTable_o0( lvl, h_gauss.abs_span_o0[lvl], h_gauss.abs_sigma_o0[lvl] );
    }

    for( int lvl=0; lvl<h_gauss.required_filter_stages; lvl++ ) {
        const float sigmaS = sigma0 * pow( 2.0, (float)(lvl)/(float)levels );
        h_gauss.abs_sigma_oN[lvl]  = sigmaS;
        h_gauss.abs_span_oN[lvl]   = h_gauss.getSpan( h_gauss.abs_sigma_oN[lvl] );
        h_gauss.computeAbsBlurTable_oN( lvl, h_gauss.abs_span_oN[lvl], h_gauss.abs_sigma_oN[lvl] );
    }
#endif // SUPPORT_ABSOLUTE_SIGMA

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
    for( int i=0; i<GAUSS_ALIGN * GAUSS_LEVELS; i++ ) {
        inc_filter[i] = 0.0f;
    }

#ifdef SUPPORT_ABSOLUTE_SIGMA
    for( int i=0; i<GAUSS_ALIGN * GAUSS_LEVELS; i++ ) {
        abs_filter_o0[i] = 0.0f;
        abs_filter_oN[i] = 0.0f;
    }
#endif // SUPPORT_ABSOLUTE_SIGMA
}

__host__
void GaussInfo::computeBlurTable( int level, int span, float sigma )
{
    /* Should be:
     * kernel[x] = exp( -0.5 * (pow((x-mean)/sigma, 2.0) ) )
     *           / sqrt(2 * M_PI * sigma * sigma);
     * but the denominator is constant and we divide by sum anyway
     */
    double sum = 1.0;
    inc_filter[level*GAUSS_ALIGN + 0] = 1.0;
    for( int x = 1; x < span; x++ ) {
        const float val = exp( -0.5 * (pow( double(x)/sigma, 2.0) ) );
        inc_filter[level*GAUSS_ALIGN + x] = val;
        sum += 2.0f * val;
    }
    for( int x = 0; x < span; x++ ) {
        inc_filter[level*GAUSS_ALIGN + x] /= sum;
    }
}

#ifdef SUPPORT_ABSOLUTE_SIGMA
__host__
void GaussInfo::computeAbsBlurTable_o0( int level, int span, float sigma )
{
    double sum = 1.0;
    abs_filter_o0[level*GAUSS_ALIGN + 0] = 1.0;
    for( int x = 1; x < span; x++ ) {
        const float val = exp( -0.5 * (pow( double(x)/sigma, 2.0) ) );
        abs_filter_o0[level*GAUSS_ALIGN + x] = val;
        sum += 2.0f * val;
    }
    for( int x = 0; x < span; x++ ) {
        abs_filter_o0[level*GAUSS_ALIGN + x] /= sum;
    }
}

__host__
void GaussInfo::computeAbsBlurTable_oN( int level, int span, float sigma )
{
    double sum = 1.0;
    abs_filter_oN[level*GAUSS_ALIGN + 0] = 1.0;
    for( int x = 1; x < span; x++ ) {
        const float val = exp( -0.5 * (pow( double(x)/sigma, 2.0) ) );
        abs_filter_oN[level*GAUSS_ALIGN + x] = val;
        sum += 2.0f * val;
    }
    for( int x = 0; x < span; x++ ) {
        abs_filter_oN[level*GAUSS_ALIGN + x] /= sum;
    }
}
#endif // SUPPORT_ABSOLUTE_SIGMA

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
    case Config::OpenCV_Compute :
        return GaussInfo::openCVSpan( sigma );
    case Config::Fixed4 :
        return 4;
    case Config::Fixed8 :
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
int GaussInfo::openCVSpan( float sigma )
{
    int span = int( roundf( 2.0f * 4.0f * sigma + 1.0f ) ) | 1;
    span >>= 1;
    span  += 1;
    return std::min<int>( span, GAUSS_ALIGN - 1 );
}

} // namespace popsift

