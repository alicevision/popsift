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
    for( int i=0; i<width/2; i++ ) {
        cout << setw(4) << setprecision(3) << kernel[i] << " ";
    }
    for( int i=width/2; i>=0; i-- ) {
        cout << setw(4) << setprecision(3) << kernel[i] << " ";
    }
    cout << endl;
}

void ratios( float* kernel, int width )
{
    cout << "      ";
    for( int i=1; i<=width/2; i+=2 ) {
        cout << setw(4) << setprecision(3) << kernel[i-1]/kernel[i] << " ";
        cout << "   - ";
    }
    cout << endl;
}

void call( float sigma, int width )
{
    float* kernel = new float[width];
    make_kernel1D( sigma, kernel, width );
    cout << "Kernel for sigma " << sigma << ":" << endl;
    plotkernel( kernel, width );
    ratios( kernel, width );

#if 0
    float* k2 = new float[width];
    for( int i=0; i<width; i++ ) k2[i] = 1;
    plotkernel( k2, width );
    ratios( k2, width );
#endif

    for( int i=0; i<ceil(width/2.0f); i++ ) {
        cout << kernel[i] / kernel[i+1] << " ";
    }
    cout << endl;
}

int main( )
{
    float  sigma  = 1.0;
    // int    width  = 2*ceil(sigma*3)+1;
    int    width  = 9;

    call( sigma, 7 );
    call( sigma, 9 );
    call( sigma, 11 );
}

#if 0
kernel size: 5
avg size: 3
avg array: 0-1 2 1-0
kernel size: 7
avg size: 4
avg array: 0-1 2-3 3-2 1-0
kernel size: 9
avg size: 5
avg array: 0-1 2-3 4 3-2 1-0
kernel size: 11
avg size: 6
kernel size: 13
avg size: 7
#endif
