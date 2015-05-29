#include <math.h>
#include <assert.h>
#include <iostream>
#include <iomanip>
#include <algorithm>

using namespace std;


float apply( float sigma )
{
    return sqrt( sigma*sigma + sigma*sigma );
}

float apply2( float sigma )
{
    return sqrt(2.0) * sigma;
}

void test( float sigma )
{
    cout << "Sigma is " << sigma << endl;
    int width = 2*ceil(sigma*3)+1;
    cout << "    a filter should have at least width " << width << endl;

    // float sigma2 = apply( sigma );
    // float sigma3 = apply( sigma2 );
    // float sigma4 = apply( sigma3 );
    // float sigma2_ = apply2( sigma );
    // float sigma3_ = apply2( sigma2 );
    // float sigma4_ = apply2( sigma3 );

    float sigmaX = sigma;
    cout << setprecision(3);
    for( int i=1; i<5; i++ ) {
        cout << "    filter applied " << i << "x is equivalent with sigma "
             << setw(4) << sigmaX
             << " (width " << 2*ceil(sigmaX*3)+1 << ")"
             << endl;
        sigmaX = sqrt(2.0) * sigmaX;
    }

    // difference = sqrt( square( result ) + square( sigma ) );
    // sqrt( square(difference) - square(sigma) ) = result

#if 0
    // float diff   = 2;
    // float sigma2 = sqrt( diff * diff - sigma * sigma );
    float  sigmaX = sqrt(4.0) * sigma;
    cout << "    applying a 2D filter 4 times yields 2*sigma for sigma 1.0" << endl
         << "    for this sigma, it yields " << sigmaX << endl;
    cout << endl;
#endif
}

int main( )
{
    test( 0.8 );
    test( 0.9 );
    test( 1.0 );
    test( 1.1 );
    test( 1.2 );
    test( 1.41 );
}

