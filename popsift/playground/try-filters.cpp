#include <math.h>
#include <iostream>
#include <iomanip>

using namespace std;

template<int levels,int sigma0x10>
class GaussFilter
{
public:
    // const float sigma0 = sigma0x10 / 10.0;
    static const float sigma0;
    static const float k;
    static const float sigma0n;

    static float* fil[levels+3];
    static int    width[levels+3];

public:
    GaussFilter( );
};

template<int levels,int sigma0x10>
const float GaussFilter<levels,sigma0x10>::sigma0 = sigma0x10 / 10.0;

template<int levels,int sigma0x10>
const float GaussFilter<levels,sigma0x10>::k = powf(2.0f, 1.0f / levels);

template<int levels,int sigma0x10>
const float GaussFilter<levels,sigma0x10>::sigma0n = 0.5f;

template<int levels,int sigma0x10>
float* GaussFilter<levels,sigma0x10>::fil[levels+3];

template<int levels,int sigma0x10>
int GaussFilter<levels,sigma0x10>::width[levels+3];

template<int levels,int sigma0x10,int s>
struct Sig
{
    static const float val;
};

/* sa   = sigma0 * pow(k, s - 1);
 * stot = sa * k; 
 * sig[s] = sqrt(stot * stot - sa * sa)
 *        = sqrt( sa * k * sa * k - sa * sa )
 *        = sqrt( sa * sa * ( k * k - 1 ) )
 */
template<int levels,int sigma0x10,int s>
const float  Sig<levels,sigma0x10,s>::val = sqrt( GaussFilter<levels,sigma0x10>::sigma0 * pow(GaussFilter<levels,sigma0x10>::k, s - 1)
                                                * GaussFilter<levels,sigma0x10>::sigma0 * pow(GaussFilter<levels,sigma0x10>::k, s - 1)
                                                * ( GaussFilter<levels,sigma0x10>::k * GaussFilter<levels,sigma0x10>::k - 1 ) );


template<int levels,int sigma0x10>
struct Sig<levels,sigma0x10,0>
{
    static const float val;
};

template<int levels,int sigma0x10>
const float  Sig<levels,sigma0x10,0>::val = sqrt( GaussFilter<levels,sigma0x10>::sigma0 * GaussFilter<levels,sigma0x10>::sigma0
                                                - GaussFilter<levels,sigma0x10>::sigma0n * GaussFilter<levels,sigma0x10>::sigma0n * 4);

template<int levels,int sigma0x10>
float sigVal( int x )
{
    switch( x )
    {
    case  0 : return Sig<levels,sigma0x10, 0>::val;
    case  1 : return Sig<levels,sigma0x10, 1>::val;
    case  2 : return Sig<levels,sigma0x10, 2>::val;
    case  3 : return Sig<levels,sigma0x10, 3>::val;
    case  4 : return Sig<levels,sigma0x10, 4>::val;
    case  5 : return Sig<levels,sigma0x10, 5>::val;
    case  6 : return Sig<levels,sigma0x10, 6>::val;
    case  7 : return Sig<levels,sigma0x10, 7>::val;
    case  8 : return Sig<levels,sigma0x10, 8>::val;
    case  9 : return Sig<levels,sigma0x10, 9>::val;
    case 10 : return Sig<levels,sigma0x10,10>::val;
    case 11 : return Sig<levels,sigma0x10,11>::val;
    case 12 : return Sig<levels,sigma0x10,12>::val;
    case 13 : return Sig<levels,sigma0x10,13>::val;
    case 14 : return Sig<levels,sigma0x10,14>::val;
    case 15 : return Sig<levels,sigma0x10,15>::val;
    case 16 : return Sig<levels,sigma0x10,16>::val;
    case 17 : return Sig<levels,sigma0x10,17>::val;
    case 18 : return Sig<levels,sigma0x10,18>::val;
    case 19 : return Sig<levels,sigma0x10,19>::val;
    default :
        std::cerr << "undefined s" << std::endl;
        exit(-__LINE__);
        return 0;
    }
}

template<int levels,int sigma0x10>
GaussFilter<levels,sigma0x10>::GaussFilter( )
{
    cout << "Creating filters with " << levels << " levels and sigma " << sigma0 << endl;
    int j;
    float sig[levels + 3];

#if 1
    /* precompute Gaussian sigmas */
    sig[0] = sqrt(sigma0 * sigma0 - sigma0n * sigma0n * 4);
    for (int s = 1; s < levels + 3; s++) {		
        const float sa   = sigma0 * pow(k, s - 1);
        const float stot = sa * k; 
        sig[s] = sqrt(stot * stot - sa * sa);
    }
#else
    for (int s = 0; s < levels + 3; s++) {		
        sig[s] = sigVal<levels,sigma0x10>( s );
    }
#endif

    // float** fil   = new float*[levels+3];
    // int*    width = new int[levels + 3];

    /* precompute gaussian kernels */
    for (int s = 0; s < levels + 3; s++) {
        float sum = 0;
        const int W = (int)(ceil(float(4.0) * sig[s]));
        fil[s] = new float[ 2 * W + 1 ];
        for (j = 0; j < W; j++) {
            const float v = float(exp
                       (float
                        (-0.5 * (j - W) * (j - W) / (sig[s] * sig[s]))));
            fil[s][j] = fil[s][2 * W - j] = v;
            sum += 2 * v;
        }
        fil[s][j] = 1;
        sum += 1;

        /* normalize */
        const float v = 1.0f / sum;
        for (j = 0; j < 2 * W + 1; j++)
            fil[s][j] *= v;

        /* set half width */
        width[s] = W;

        cout << "iteration: " << s << " width: " << width[s] << endl;
    }
    for (int s = 0; s < levels + 3; s++) {
        cout << "  kernel " << s << ": < ";
        const int W=width[s];
        for( int x=W; x < 2*W+1; x++ ) {
            cout << setprecision(3) << fil[s][x] << " ";
        }
        cout << ">" << endl;
    }
}


int main( )
{
    int   levels = 3;
    float sigma  = 1.6f;

    // prep_gauss_filter( 1.6f, 2 );
    // prep_gauss_filter( 1.6f, 3 );
    // prep_gauss_filter( 1.6f, 4 );
    GaussFilter<2,16> g1;
    GaussFilter<3,16> g2;
    GaussFilter<4,16> g3;

    GaussFilter<3,15> g4;
    GaussFilter<3,16> g5;
    GaussFilter<3,17> g6;
}

