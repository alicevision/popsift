#include "s_pyramid.h"
#include "debug_macros.h"

#define PRINT_GAUSS_FILTER_SYMBOL

using namespace std;

namespace popart {

__device__ __constant__ float d_gauss_filter[32];

#ifdef PRINT_GAUSS_FILTER_SYMBOL
__global__
void print_gauss_filter_symbol( uint32_t columns )
{
    printf("Entering print_gauss_filter_symbol\n");
    for( uint32_t x=0; x<columns; x++ ) {
        printf("%0.3f ", d_gauss_filter[x] );
    }
    printf("\n");
}
#endif // PRINT_GAUSS_FILTER_SYMBOL

/*************************************************************
 * Initialize the Gauss filter table in constant memory
 *************************************************************/

void Pyramid::init_filter( float sigma0, uint32_t levels )
{
    cerr << "Entering " << __FUNCTION__ << endl;
    if( sigma0 > 2.0 )
    {
        cerr << __FILE__ << ":" << __LINE__ << ", ERROR: "
             << " Sigma > 2.0 is not supported. Re-size __constant__ array and recompile."
             << endl;
        exit( -__LINE__ );
    }
    if( levels > 12 )
    {
        cerr << __FILE__ << ":" << __LINE__ << ", ERROR: "
             << " More than 12 levels not supported. Re-size __constant__ array and recompile."
             << endl;
        exit( -__LINE__ );
    }

    float local_filter[32];
    // const int W = GAUSS_SPAN; // no filter wider than 25; 32 is just for alignment
    // assert( W % 2 == 1 ); // filters should be symmetric, i.e. odd-sized
    // const double mean = GAUSS_ONE_SIDE_RANGE; // is always (GAUSS_SPAN-1)/2

    float sigma = sigma0;
    double sum = 0.0;
    for (int x = 0; x < GAUSS_SPAN; ++x) {
            /* Should be:
             * kernel[x] = exp( -0.5 * (pow((x-mean)/sigma, 2.0) ) )
             *           / sqrt(2 * M_PI * sigma * sigma);
             _w /= 2;
             _h /= 2;
             * but the denominator is constant and we divide by sum anyway
             */
        local_filter[x] = exp( -0.5 * (pow( double(x-GAUSS_ONE_SIDE_RANGE)/sigma, 2.0) ) );
        sum += local_filter[x];
    }

    for (int x = 0; x < GAUSS_SPAN; ++x) 
        local_filter[x] /= sum;

    cudaError_t err;
    err = cudaMemcpyToSymbol( d_gauss_filter,
                              local_filter,
                              32*sizeof(float),
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

