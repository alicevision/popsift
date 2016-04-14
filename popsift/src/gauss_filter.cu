#include <stdio.h>

#include "gauss_filter.h"
// #include "s_pyramid.h"
#include "debug_macros.h"

#undef PRINT_GAUSS_FILTER_SYMBOL

using namespace std;

namespace popart {

__device__ __constant__ float d_gauss_filter[ GAUSS_ALIGN * GAUSS_LEVELS ];

#ifdef PRINT_GAUSS_FILTER_SYMBOL
__global__
void print_gauss_filter_symbol( int columns )
{
    printf("Entering print_gauss_filter_symbol\n");
    for( int lvl=0; lvl<GAUSS_LEVELS; lvl++ ) {
        for( int x=0; x<GAUSS_ALIGN; x++ ) {
            printf("%0.3f ", d_gauss_filter[lvl*GAUSS_ALIGN+x] );
        }
        printf("\n");
    }
    printf("\n");
}
#endif // PRINT_GAUSS_FILTER_SYMBOL

/*************************************************************
 * Initialize the Gauss filter table in constant memory
 *************************************************************/

void init_filter( float sigma0, int levels, bool vlfeat_mode )
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

    float local_filter[ GAUSS_ALIGN * GAUSS_LEVELS ];
    // const int W = GAUSS_SPAN; // no filter wider than 25; 32 is just for alignment
    // assert( W % 2 == 1 ); // filters should be symmetric, i.e. odd-sized
    // const double mean = the center value

    float sigma = sigma0;
    for( int lvl=0; lvl<GAUSS_LEVELS; lvl++ ) {
        for( int x = 1; x < GAUSS_ALIGN; x++ ) {
            local_filter[lvl * GAUSS_ALIGN + x] = 0.0;
        }

        local_filter[lvl * GAUSS_ALIGN + 0] = 1.0;
        double sum = 1.0;
        for( int x = 1; x <= GAUSS_SPAN; x++ ) {
                /* Should be:
                 * kernel[x] = exp( -0.5 * (pow((x-mean)/sigma, 2.0) ) )
                 *           / sqrt(2 * M_PI * sigma * sigma);
                 * _w /= 2;
                 * _h /= 2;
                 * but the denominator is constant and we divide by sum anyway
                 */
            local_filter[lvl * GAUSS_ALIGN + x] = exp( -0.5 * (pow( double(x)/sigma, 2.0) ) );
            sum += 2 * local_filter[lvl * GAUSS_ALIGN + x];
        }

        for( int x = 0; x <= GAUSS_SPAN; x++ ) 
            local_filter[lvl * GAUSS_ALIGN + x] /= sum;

        if( vlfeat_mode == true ) {
            sigma *= pow( 2.0, 1.0/levels );
        } else {
            // OpenCV sigma computation
            const float sigmaP = sigma0 * pow( 2.0, (float)(lvl+0)/(float)levels );
            const float sigmaS = sigma0 * pow( 2.0, (float)(lvl+1)/(float)levels );

            sigma = sqrt( sigmaS * sigmaS - sigmaP * sigmaP );
        }
    }

    cudaError_t err;
    err = cudaMemcpyToSymbol( d_gauss_filter,
                              local_filter,
                              GAUSS_ALIGN * GAUSS_LEVELS * sizeof(float),
                              0,
                              cudaMemcpyHostToDevice );
    POP_CUDA_FATAL_TEST( err, "cudaMemcpyToSymbol failed for Gauss kernel initialization: " );

#ifdef PRINT_GAUSS_FILTER_SYMBOL
    print_gauss_filter_symbol
        <<<1,1>>>
        ( GAUSS_SPAN );
    err = cudaGetLastError();
    POP_CUDA_FATAL_TEST( err, "print_gauss_filter_symbol failed: " );
#endif // PRINT_GAUSS_FILTER_SYMBOL
}

} // namespace popart

