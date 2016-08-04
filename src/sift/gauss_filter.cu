#include <stdio.h>

#include "gauss_filter.h"
#include "debug_macros.h"

using namespace std;

namespace popart {

__device__ __constant__
GaussInfo d_gauss;

__global__
void print_gauss_filter_symbol( int columns )
{
    printf( "\n"
            "Gauss tables\n"
            "      level span sigma : center value -> edge value\n"
            "    initial blur\n" );

    int span = d_gauss.initial_span + d_gauss.initial_span - 1;

    printf("      %d %d %2.6f: ", 0, span, d_gauss.initial_sigma );
    int m = min( d_gauss.initial_span, columns );
    for( int x=0; x<m; x++ ) {
        printf("%0.8f ", d_gauss.filter_initial_blur[x] );
    }
    if( m < d_gauss.initial_span )
        printf("...\n");
    else
        printf("\n");

    printf("    relative sigma\n");

    for( int lvl=0; lvl<d_gauss.required_filter_stages; lvl++ ) {
        span = d_gauss.span[lvl] + d_gauss.span[lvl] - 1;

        printf("      %d %d %2.6f: ", lvl, span, d_gauss.sigma[lvl] );
        int m = min( d_gauss.span[lvl], columns );
        for( int x=0; x<m; x++ ) {
            printf("%0.8f ", d_gauss.filter[lvl*GAUSS_ALIGN+x] );
        }
        if( m < d_gauss.span[lvl] )
            printf("...\n");
        else
            printf("\n");
    }
    printf("\n");

#ifdef SUPPORT_ABSOLUTE_SIGMA
    printf("    absolute filters\n");

    for( int lvl=0; lvl<d_gauss.required_filter_stages; lvl++ ) {
        span = d_gauss.abs_span[lvl] + d_gauss.abs_span[lvl] - 1;

        printf("      %d %d %2.6f: ", lvl, span, d_gauss.abs_sigma[lvl] );
        int m = min( d_gauss.abs_span[lvl], columns );
        for( int x=0; x<m; x++ ) {
            printf("%0.8f ", d_gauss.from_lvl_1[lvl*GAUSS_ALIGN+x] );
        }
        if( m < d_gauss.abs_span[lvl] )
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

    GaussInfo h_gauss;

    h_gauss.clearTables();

    h_gauss.required_filter_stages = levels + 3;

    // float local_filter_initial_blur[ GAUSS_ALIGN ];

    if( conf.hasInitialBlur() ) {
        const float initial_blur = conf.getInitialBlur() * pow( 2.0, conf.getUpscaleFactor() );

        h_gauss.initial_sigma = sqrt( fabsf( sigma0 * sigma0 - initial_blur * initial_blur ) );
        h_gauss.initial_span  = ( conf.getSiftMode() == Config::OpenCV )
                              ? GaussInfo::openCVSpan( h_gauss.initial_sigma )
                              : GaussInfo::vlFeatSpan( h_gauss.initial_sigma );

        h_gauss.computeInitialBlurTable( h_gauss.initial_span, h_gauss.initial_sigma );

        if( conf.ifPrintGaussTables() ) {
            printf("    Sigma for remaining top level blur: %f = sqrt(sigma0(%f)^2 , initial_blur(%f)^2)\n", h_gauss.initial_sigma, sigma0, initial_blur );
        }
    }

    h_gauss.sigma[0] = sigma0;
    h_gauss.span[0]  = ( conf.getSiftMode() == Config::OpenCV )
                     ? GaussInfo::openCVSpan( sigma0 )
                     : GaussInfo::vlFeatSpan( sigma0 );
    h_gauss.computeBlurTable( 0, h_gauss.span[0], h_gauss.sigma[0] );
    if( conf.ifPrintGaussTables() ) {
        printf("    Sigma for level 0: %2.6f = sigma0(%2.6f)\n", h_gauss.sigma[0], sigma0 );
    }

    for( int lvl=1; lvl<h_gauss.required_filter_stages; lvl++ ) {
        const float sigmaP = sigma0 * pow( 2.0, (float)(lvl-1)/(float)levels );
        const float sigmaS = sigma0 * pow( 2.0, (float)(lvl  )/(float)levels );

        h_gauss.sigma[lvl] = sqrt( sigmaS * sigmaS - sigmaP * sigmaP );
        h_gauss.span[lvl]  = ( conf.getSiftMode() == Config::OpenCV )
                           ? GaussInfo::openCVSpan( h_gauss.sigma[lvl] )
                           : GaussInfo::vlFeatSpan( h_gauss.sigma[lvl] );
        h_gauss.computeBlurTable( lvl, h_gauss.span[lvl], h_gauss.sigma[lvl] );

        if( conf.ifPrintGaussTables() ) {
            printf("    Sigma for level %d: %2.6f = sqrt(sigmaS(%2.6f)^2 - sigmaP(%2.6f)^2)\n", lvl, h_gauss.sigma[lvl], sigmaS, sigmaP );
        }
    }

#ifdef SUPPORT_ABSOLUTE_SIGMA
    for( int lvl=1; lvl<h_gauss.required_filter_stages; lvl++ ) {
        const float sigmaP = sigma0;
        const float sigmaS = sigma0 * pow( 2.0, (float)(lvl)/(float)levels );

        h_gauss.abs_sigma[lvl] = sqrt( sigmaS * sigmaS - sigmaP * sigmaP );
        h_gauss.from_lvl_1[lvl] = sigma0 * pow( 2.0, (float)(lvl)/(float)levels );
        h_gauss.abs_span[lvl]  = ( conf.getSiftMode() == Config::OpenCV )
                               ? GaussInfo::openCVSpan( h_gauss.abs_sigma[lvl] )
                               : GaussInfo::vlFeatSpan( h_gauss.abs_sigma[lvl] );
        h_gauss.computeAbsBlurTable( lvl, h_gauss.span[lvl], h_gauss.abs_sigma[lvl] );
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
    initial_span = 1;

    for( int i=0; i<GAUSS_ALIGN; i++ ) {
        filter_initial_blur[i] = 0.0f;
    }

    for( int i=0; i<GAUSS_ALIGN * GAUSS_LEVELS; i++ ) {
        filter[i] = 0.0f;
    }

#ifdef SUPPORT_ABSOLUTE_SIGMA
    for( int i=0; i<GAUSS_ALIGN * GAUSS_LEVELS; i++ ) {
        from_lvl_1[i] = 0.0f;
    }
#endif // SUPPORT_ABSOLUTE_SIGMA
}

__host__
void GaussInfo::computeInitialBlurTable( int span, float sigma )
{
    double sum = 1.0;
    filter_initial_blur[0] = 1.0;
    for( int x = 1; x < span; x++ ) {
        const float val = exp( -0.5 * (pow( double(x)/sigma, 2.0) ) );
        filter_initial_blur[x] = val;
        sum += 2.0f * val;
    }
    for( int x = 0; x < span; x++ ) {
        filter_initial_blur[x] /= sum;
    }
}

__host__
void GaussInfo::computeBlurTable( int level, int span, float sigma )
{
        /* Should be:
         * kernel[x] = exp( -0.5 * (pow((x-mean)/sigma, 2.0) ) )
         *           / sqrt(2 * M_PI * sigma * sigma);
         * _w /= 2;
         * _h /= 2;
         * but the denominator is constant and we divide by sum anyway
         */
    double sum = 1.0;
    filter[level*GAUSS_ALIGN + 0] = 1.0;
    for( int x = 1; x < span; x++ ) {
        const float val = exp( -0.5 * (pow( double(x)/sigma, 2.0) ) );
        filter[level*GAUSS_ALIGN + x] = val;
        sum += 2.0f * val;
    }
    for( int x = 0; x < span; x++ ) {
        filter[level*GAUSS_ALIGN + x] /= sum;
    }
}

__host__
void GaussInfo::computeAbsBlurTable( int level, int span, float sigma )
{
    double sum = 1.0;
    from_lvl_1[level*GAUSS_ALIGN + 0] = 1.0;
    for( int x = 1; x < span; x++ ) {
        const float val = exp( -0.5 * (pow( double(x)/sigma, 2.0) ) );
        from_lvl_1[level*GAUSS_ALIGN + x] = val;
        sum += 2.0f * val;
    }
    for( int x = 0; x < span; x++ ) {
        from_lvl_1[level*GAUSS_ALIGN + x] /= sum;
    }
}

__host__
int GaussInfo::vlFeatSpan( float sigma )
{
        /* This is the VLFeat computation for choosing the Gaussian filter width.
         * In our case, we look at the half-sided filter including the center value.
         */
    return min<int>( ceilf( 4.0f * sigma ) + 1, GAUSS_ALIGN - 1 );
}

__host__
int GaussInfo::openCVSpan( float sigma )
{
    int span = int( roundf( 2.0f * 4.0f * sigma + 1.0f ) ) | 1;
    // span = span | 1;   // don't make odd like original code, because
    span >>= 1;           // we divide by two anyway
    span  += 1;           // add the center node
    return min<int>( span, GAUSS_ALIGN - 1 );
}

} // namespace popart

