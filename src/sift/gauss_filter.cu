#include <stdio.h>

#include "gauss_filter.h"
// #include "s_pyramid.h"
#include "debug_macros.h"

#undef PRINT_GAUSS_FILTER_SYMBOL

using namespace std;

namespace popart {

__device__ __constant__
float d_gauss_filter_initial_blur[ GAUSS_ALIGN ];

__device__ __constant__
float d_gauss_filter[ GAUSS_ALIGN * GAUSS_LEVELS ];

__device__ __constant__
float d_gauss_from_lvl_1[ GAUSS_ALIGN * GAUSS_LEVELS ];

#ifdef PRINT_GAUSS_FILTER_SYMBOL
__global__
void print_gauss_filter_symbol( int columns )
{
    printf("Gauss tables for initial blur\n");
    for( int x=0; x<columns; x++ ) {
        printf("%0.3f ", d_gauss_filter_initial_blur[x] );
    }
    printf("\n");

    printf("Gauss tables with relative differences\n");
    for( int lvl=0; lvl<GAUSS_LEVELS; lvl++ ) {
        for( int x=0; x<columns; x++ ) {
            printf("%0.3f ", d_gauss_filter[lvl*GAUSS_ALIGN+x] );
        }
        printf("\n");
    }
    printf("\n");
    printf("Gauss tables with absolute filters\n");
    for( int lvl=0; lvl<GAUSS_LEVELS; lvl++ ) {
        for( int x=0; x<columns; x++ ) {
            printf("%0.3f ", d_gauss_from_lvl_1[lvl*GAUSS_ALIGN+x] );
        }
        printf("\n");
    }
    printf("\n");
}
#endif // PRINT_GAUSS_FILTER_SYMBOL

/*************************************************************
 * Initialize the Gauss filter table in constant memory
 *************************************************************/

void init_filter( float sigma0, int levels, bool vlfeat_mode, bool assume_initial_blur, float initial_blur, float downsampling_factor )
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

    initial_blur *= pow( 2.0, -1.0f * downsampling_factor );

    float sigma;

    float local_filter_initial_blur[ GAUSS_ALIGN ];

    if( assume_initial_blur ) {
        sigma = sqrt( fabsf( sigma0 * sigma0 - initial_blur * initial_blur ) );
        printf("Creating table for initial blur %f\n", sigma);

        double sum = 1.0;
        local_filter_initial_blur[0] = 1.0;
        for( int x = 1; x <= GAUSS_SPAN; x++ ) {
            const float val = exp( -0.5 * (pow( double(x)/sigma, 2.0) ) );
            local_filter_initial_blur[x] = val;
            sum += 2.0f * val;
        }
        for( int x = 0; x <= GAUSS_SPAN; x++ ) {
            local_filter_initial_blur[x] /= sum;
        }
    } else {
        for( int x = 0; x <= GAUSS_SPAN; x++ ) {
            local_filter_initial_blur[x] = 0;
        }
    }

    float local_filter_rel[ GAUSS_ALIGN * GAUSS_LEVELS ];
    float local_filter_abs[ GAUSS_ALIGN * GAUSS_LEVELS ];
    // const int W = GAUSS_SPAN; // no filter wider than 25; 32 is just for alignment
    // assert( W % 2 == 1 ); // filters should be symmetric, i.e. odd-sized
    // const double mean = the center value

    sigma = sigma0;
    if( vlfeat_mode == true ) {
        printf("We are in VLFeat mode\n");
    } else {
        printf("We are in OpenCV mode\n");
#ifdef PRINT_GAUSS_FILTER_SYMBOL
        printf("sigma is initially sigma0, afterwards the difference between previous 2 sigmas\n");
#endif
    }
#ifdef PRINT_GAUSS_FILTER_SYMBOL
    printf( "Sigma values for creating Gauss tables:\n" );
    printf( "sigma for 1 filter is %f\n", sigma );
#endif
    for( int lvl=0; lvl<GAUSS_LEVELS; lvl++ ) {
        for( int x = 1; x < GAUSS_ALIGN; x++ ) {
            local_filter_rel[lvl * GAUSS_ALIGN + x] = 0.0;
            local_filter_abs[lvl * GAUSS_ALIGN + x] = 0.0;
        }

        local_filter_rel[lvl * GAUSS_ALIGN + 0] = 1.0;
        local_filter_abs[lvl * GAUSS_ALIGN + 0] = 1.0;

        float absSigma;
        if( vlfeat_mode == true ) {
            absSigma = sigma;
            float multiplier = pow( 2.0, 1.0/levels );
            for( int l=0; l<lvl; l++ ) {
                absSigma *= multiplier;
            }
        } else {
            absSigma = sigma0 * pow( 2.0, (float)(lvl)/(float)levels );
        }
        double sum_rel = 1.0;
        double sum_abs = 1.0;
        for( int x = 1; x <= GAUSS_SPAN; x++ ) {
                /* Should be:
                 * kernel[x] = exp( -0.5 * (pow((x-mean)/sigma, 2.0) ) )
                 *           / sqrt(2 * M_PI * sigma * sigma);
                 * _w /= 2;
                 * _h /= 2;
                 * but the denominator is constant and we divide by sum anyway
                 */
            const float val_rel = exp( -0.5 * (pow( double(x)/sigma, 2.0) ) );
            const float val_abs = exp( -0.5 * (pow( double(x)/absSigma, 2.0) ) );
            local_filter_rel[lvl * GAUSS_ALIGN + x] = val_rel;
            local_filter_abs[lvl * GAUSS_ALIGN + x] = val_abs;
            sum_rel += 2 * val_rel;
            sum_abs += 2 * val_abs;
        }

        for( int x = 0; x <= GAUSS_SPAN; x++ ) {
            local_filter_rel[lvl * GAUSS_ALIGN + x] /= sum_rel;
            local_filter_abs[lvl * GAUSS_ALIGN + x] /= sum_abs;
        }

        if( vlfeat_mode == true ) {
            sigma *= pow( 2.0, 1.0/levels );
        } else {
            // OpenCV sigma computation
            const float sigmaP = sigma0 * pow( 2.0, (float)(lvl+0)/(float)levels );
            const float sigmaS = sigma0 * pow( 2.0, (float)(lvl+1)/(float)levels );

            sigma = sqrt( sigmaS * sigmaS - sigmaP * sigmaP );
#ifdef PRINT_GAUSS_FILTER_SYMBOL
            printf("    sigmaP is %f - sigmaS is %f - sigma is %f\n", sigmaP, sigmaS, sigma );
#endif
        }
    }

    cudaError_t err;
    err = cudaMemcpyToSymbol( d_gauss_filter_initial_blur,
                              local_filter_initial_blur,
                              GAUSS_ALIGN * sizeof(float),
                              0,
                              cudaMemcpyHostToDevice );
    POP_CUDA_FATAL_TEST( err, "cudaMemcpyToSymbol failed for Gauss kernel initialization: " );
    err = cudaMemcpyToSymbol( d_gauss_filter,
                              local_filter_rel,
                              GAUSS_ALIGN * GAUSS_LEVELS * sizeof(float),
                              0,
                              cudaMemcpyHostToDevice );
    POP_CUDA_FATAL_TEST( err, "cudaMemcpyToSymbol failed for Gauss kernel initialization: " );
    err = cudaMemcpyToSymbol( d_gauss_from_lvl_1,
                              local_filter_abs,
                              GAUSS_ALIGN * GAUSS_LEVELS * sizeof(float),
                              0,
                              cudaMemcpyHostToDevice );
    POP_CUDA_FATAL_TEST( err, "cudaMemcpyToSymbol failed for Gauss kernel initialization: " );

#ifdef PRINT_GAUSS_FILTER_SYMBOL
    cerr << "Initial sigma is " << sigma0 << endl;
    print_gauss_filter_symbol
        <<<1,1>>>
        ( GAUSS_SPAN );
    err = cudaGetLastError();
    POP_CUDA_FATAL_TEST( err, "print_gauss_filter_symbol failed: " );
#endif // PRINT_GAUSS_FILTER_SYMBOL
}

} // namespace popart

