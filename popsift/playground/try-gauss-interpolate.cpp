#include <math.h>
#include <assert.h>
#include <iostream>
#include <iomanip>
#include <algorithm>

using namespace std;

float make_kernel1D( float sigma, float* kernel, int width )
{
    int W = width;
    float mean = W/2;
    float sum = 0.0; // For accumulating the kernel values
    for (int x = 0; x < W; ++x) {
        /*
        kernel[x] = exp( -0.5 * (pow((x-mean)/sigma, 2.0) ) )
                  / sqrt(2 * M_PI * sigma * sigma);
        */
        kernel[x] = exp( -0.5 * (pow((x-mean)/sigma, 2.0) ) );

        // Accumulate the kernel values
        sum += kernel[x];
    }

    // Normalize the kernel
    for (int x = 0; x < W; ++x)
        kernel[x] /= sum;
    return sum;
}

void plotkernel( float* kernel, int width )
{
    cout << "    ";
    for( int i=0; i<width; i++ ) {
        cout << setw(4) << setprecision(3) << kernel[i] << " ";
    }
    cout << endl;
}

void ratios( float* kernel, int width )
{
    cout << "      ";
    for( int i=1; i<width; i++ ) {
        cout << setw(4) << setprecision(3) << kernel[i-1]/kernel[i] << " ";
    }
    cout << endl;
}

int main( )
{
    float  sigma  = 1.0;
    int    width  = 2*ceil(sigma*3)+1;
    float* kernel = new float[width];
    make_kernel1D( sigma, kernel, width );
    cout << "Kernel for sigma " << sigma << ":" << endl;
    plotkernel( kernel, width );
    ratios( kernel, width );

    float* k2 = new float[width];
    for( int i=0; i<width; i++ ) k2[i] = 1;
    plotkernel( k2, width );
    ratios( k2, width );

    int center = width/2;
    cout << "center: " << center << endl;
    for( int i=1; i<=width/2; i++ ) {
        k2[center-i] =
        k2[center+i] = kernel[center+i] / kernel[center+i-1];
    }
    plotkernel( k2, width );
}

