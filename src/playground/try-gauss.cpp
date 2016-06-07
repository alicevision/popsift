#include <math.h>
#include <assert.h>
#include <iostream>
#include <iomanip>
#include <algorithm>

using namespace std;

double make_kernel2D( double sigma, double* kernel, int width )
{
    int W = width;
    double mean = W/2;
    double sum = 0.0; // For accumulating the kernel values
    for (int x = 0; x < W; ++x) 
        for (int y = 0; y < W; ++y) {
            /*
            kernel[y*width+x] = exp( -0.5 * (pow((x-mean)/sigma, 2.0) + pow((y-mean)/sigma,2.0)) )
                              / (2 * M_PI * sigma * sigma);
            */
            kernel[y*width+x] = exp( -0.5 * (pow((x-mean)/sigma, 2.0) + pow((y-mean)/sigma,2.0)) );

            // Accumulate the kernel values
            sum += kernel[y*width+x];
        }

    // Normalize the kernel
    for (int x = 0; x < W; ++x) 
        for (int y = 0; y < W; ++y)
            kernel[y*width+x] /= sum;
    return sum;
}

double make_kernel1D( double sigma, double* kernel, int width )
{
    int W = width;
    double mean = W/2;
    double sum = 0.0; // For accumulating the kernel values
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

void apply2D( const double* kernel, int kwidth, double* dst, const double* src, int iwidth )
{
    for( int y=0; y<iwidth; y++ ) {
        for( int x=0; x<iwidth; x++ ) {
// cout << "Dest is (" << y << "," << x << ")" << endl;
            dst[y*iwidth+x] = 0;
            assert( 2*(kwidth/2) < kwidth );
            for( int h=-kwidth/2; h<=kwidth/2; h++ ) {
                for( int w=-kwidth/2; w<=kwidth/2; w++ ) {
                    int xw = std::min<int>( std::max<int>(x+w,0), iwidth-1 );
                    int yh = std::min<int>( std::max<int>(y+h,0), iwidth-1 );
                    const double srcval = src[ yh*iwidth + xw ];

// cout << "(" << y+h << "," << x+w << "): " << setprecision(3) << srcval << " ";
                    const double kerval = kernel[ (h+kwidth/2)*kwidth + (w+kwidth/2) ];
                    dst[y*iwidth+x] += ( srcval * kerval );
                }
// cout << endl;
            }
        }
    }
    cout << endl;
}

void apply1D_row( const double* kernel, int kwidth, double* dst, const double* src, int iwidth )
{
    for( int y=0; y<iwidth; y++ ) {
        for( int x=0; x<iwidth; x++ ) {
            dst[y*iwidth+x] = 0;
            assert( 2*(kwidth/2) < kwidth );
            for( int w=-kwidth/2; w<=kwidth/2; w++ ) {
                int xw = std::min<int>( std::max<int>(x+w,0), iwidth-1 );
                const double srcval = src[ y*iwidth + xw ];

                const double kerval = kernel[ w+kwidth/2 ];
                dst[y*iwidth+x] += ( srcval * kerval );
            }
        }
    }
    cout << endl;
}

void apply1D_col( const double* kernel, int kwidth, double* dst, const double* src, int iwidth )
{
    for( int y=0; y<iwidth; y++ ) {
        for( int x=0; x<iwidth; x++ ) {
            dst[y*iwidth+x] = 0;
            assert( 2*(kwidth/2) < kwidth );
            for( int h=-kwidth/2; h<=kwidth/2; h++ ) {
                int yh = std::min<int>( std::max<int>(y+h,0), iwidth-1 );
                const double srcval = src[ yh*iwidth + x ];

                const double kerval = kernel[ h+kwidth/2 ];
                dst[y*iwidth+x] += ( srcval * kerval );
            }
        }
    }
    cout << endl;
}

void printit( double* image, int width )
{
    for( int h=0; h<width; h++ ) {
        for( int w=0; w<width; w++ ) {
            cout << setw(4) << setprecision(3) << image[h*width+w] << " ";
        }
        cout << endl;
    }
}

int main( )
{
    const double sigma=1.0;
    const int width=9;    // width should be ceil(6*sigma)^2

    int     w1D_1 = ceil(sigma)*6+1;
    double* k = new double[width*width];
    cout << sigma << " -> " << w1D_1 << endl;
    make_kernel2D( sigma, k, width );

                    // printit( k, width );
                    // cout << "Sum: " << sum << endl;
    double* k1D = new double[width];
    make_kernel1D( sigma, k1D, width );

    double  sigma2 = sqrt(2.0) * sigma;
    int     w1D_2 = ceil(sigma2)*6+1;
    cout << sigma2 << " -> " << w1D_2 << endl;
    double* k1D_2 = new double[w1D_2];
    make_kernel1D( sigma2, k1D_2, w1D_2 );

    double  sigma3 = sqrt(3.0) * sigma;
    int     w1D_3 = ceil(sigma3)*6+1;
    cout << sigma3 << " -> " << w1D_3 << endl;
    double* k1D_3 = new double[w1D_3];
    make_kernel1D( sigma3, k1D_3, w1D_3 );

    double  sigma4 = sqrt(4.0) * sigma;
    int     w1D_4 = ceil(sigma4)*6+1;
    cout << sigma4 << " -> " << w1D_4 << endl;
    double* k1D_4 = new double[w1D_4];
    make_kernel1D( sigma4, k1D_4, w1D_4 );

    double  sigma6 = sqrt(6.0) * sigma;
    int     w1D_6 = ceil(sigma6)*6+1;
    cout << sigma6 << " -> " << w1D_6 << endl;
    double* k1D_6 = new double[w1D_6];
    make_kernel1D( sigma6, k1D_6, w1D_6 );

    cout << endl
         << "Applying kernel to a dummy image of 1s" << endl;
    const int imagewidth = 3 * width;
    double* image = new double[imagewidth*imagewidth];
    for( int h=0; h<imagewidth; h++ ) {
        for( int w=0; w<imagewidth; w++ ) {
            image[h*imagewidth+w] = 1.0;
        }
    }
    image[10*imagewidth+10] = 255;

    double* out_i2D_1 = new double[imagewidth*imagewidth];
    double* out_i2D_2 = new double[imagewidth*imagewidth];
    double* out_i2D_3 = new double[imagewidth*imagewidth];
    double* out_i2D_4 = new double[imagewidth*imagewidth];
    double* out_i2D_5 = new double[imagewidth*imagewidth];
    double* out_i2D_6 = new double[imagewidth*imagewidth];

    double* out_i1D_1 = new double[imagewidth*imagewidth];
    double* out_i1D_2 = new double[imagewidth*imagewidth];
    double* out_i1D_3 = new double[imagewidth*imagewidth];
    double* out_i1D_4 = new double[imagewidth*imagewidth];
    double* out_i1D_5 = new double[imagewidth*imagewidth];
    double* out_i1D_6 = new double[imagewidth*imagewidth];
    double* out_i1D_7 = new double[imagewidth*imagewidth];
    double* out_i1D_8 = new double[imagewidth*imagewidth];

    apply2D( k, width, out_i2D_1, image, imagewidth );
    apply2D( k, width, out_i2D_2, out_i2D_1, imagewidth );
    apply2D( k, width, out_i2D_3, out_i2D_2, imagewidth );
    apply2D( k, width, out_i2D_4, out_i2D_3, imagewidth );
    apply2D( k, width, out_i2D_5, out_i2D_4, imagewidth );
    apply2D( k, width, out_i2D_6, out_i2D_5, imagewidth );
    cout << "Applying 2D kernel with sigma " << sigma << " 6 times (kernel size " << width << "x" << width << ")" << endl;
    printit( out_i2D_6, imagewidth );

    // apply1D_row( k1D, width, out_i1D_1, image, imagewidth );
    // apply1D_row( k1D, width, out_i1D_2, out_i1D_1, imagewidth );
    // apply1D_col( k1D, width, out_i1D_3, out_i1D_2, imagewidth );
    // apply1D_col( k1D, width, out_i1D_4, out_i1D_3, imagewidth );
    // printit( out_i1D_4, imagewidth );
    // apply1D_row( k1D, width, out_i1D_1, image, imagewidth );
    // apply1D_row( k1D, width, out_i1D_2, out_i1D_1, imagewidth );
    // apply1D_row( k1D, width, out_i1D_3, out_i1D_2, imagewidth );
    // apply1D_col( k1D, width, out_i1D_4, out_i1D_3, imagewidth );
    // apply1D_col( k1D, width, out_i1D_5, out_i1D_4, imagewidth );
    // apply1D_col( k1D, width, out_i1D_6, out_i1D_5, imagewidth );
    // printit( out_i1D_6, imagewidth );

    // apply1D_row( k1D_3, w1D_3, out_i1D_7, image, imagewidth );
    // apply1D_col( k1D_3, w1D_3, out_i1D_8, out_i1D_7, imagewidth );
    // cout << "Applying 1D kernel with sigma " << sigma3 << " once horiz, once vert (kernel size " << w1D_3 << ")" << endl;
    // printit( out_i1D_8, imagewidth );

    apply1D_row( k1D_6, w1D_6, out_i1D_7, image, imagewidth );
    apply1D_col( k1D_6, w1D_6, out_i1D_8, out_i1D_7, imagewidth );
    cout << "Applying 1D kernel with sigma " << sigma3 << " once horiz, once vert (kernel size " << w1D_3 << ")" << endl;
    printit( out_i1D_8, imagewidth );
#if 0
    apply2D( k, width, out_image2, out_image, imagewidth );
    printit( out_image2, imagewidth );

    double* k2 = new double[width*width];
    apply2D( k, width, k2, k, width );
    apply2D( k2, width, out_image3, image, imagewidth );
    cout << "First kernel" << endl;
    printit( k, width );
    cout << "Second kernel" << endl;
    printit( k2, width );
    cout << "Final image kernel" << endl;
    printit( out_image3, imagewidth );
#endif
}

