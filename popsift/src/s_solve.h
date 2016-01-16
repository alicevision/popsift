#pragma once

#include <cuda_runtime.h>

#define CLOSED_FORM_SOLVE

#ifdef CLOSED_FORM_SOLVE
__device__
inline bool solve( float i[3][3], float b[3] )
{
    float det0b = - i[1][2] * i[1][2];
    float det0a =   i[1][1] * i[2][2];
    float det0 = det0b + det0a;

    float det1b = - i[0][1] * i[2][2];
    float det1a =   i[1][2] * i[0][2];
    float det1 = det1b + det1a;

    float det2b = - i[1][1] * i[0][2];
    float det2a =   i[0][1] * i[1][2];
    float det2 = det2b + det2a;

    float det3b = - i[0][2] * i[0][2];
    float det3a =   i[0][0] * i[2][2];
    float det3 = det3b + det3a;

    float det4b = - i[0][0] * i[1][2];
    float det4a =   i[0][1] * i[0][2];
    float det4 = det4b + det4a;

    float det5b = - i[0][1] * i[0][1];
    float det5a =   i[0][0] * i[1][1];
    float det5 = det5b + det5a;

    float det;
    det  = ( i[0][0] * det0 );
    det += ( i[0][1] * det1 );
    det += ( i[0][2] * det2 );

    if( det == 0 ) {
        return false;
    }

    // float rsd = 1.0 / det;
    float rsd = __frcp_rn( det );

    i[0][0] = det0 * rsd;
    i[1][0] = det1 * rsd;
    i[2][0] = det2 * rsd;
    i[1][1] = det3 * rsd;
    i[1][2] = det4 * rsd;
    i[2][2] = det5 * rsd;
    i[0][1] = i[1][0];
    i[0][2] = i[2][0];
    i[2][1] = i[1][2];

    float vout[3];
    vout[0] = vout[1] = vout[2] = 0;
    for (   int y = 0;  y < 3;  y ++ ) {
        for ( int x = 0;  x < 3;  x ++ ) {
            vout[y] += ( i[y][x] * b[x] );
        }
    }
    b[0] = vout[0];
    b[1] = vout[1];
    b[2] = vout[2];

    return true;
}

#else // not CLOSED_FORM_SOLVE

__device__
inline bool solve( float A[3][3], float b[3] )
{
    // Gauss elimination
    for( int j = 0 ; j < 3 ; j++ ) {
            // look for leading pivot
            float maxa    = 0;
            float maxabsa = 0;
            int   maxi    = -1;
            for( int i = j ; i < 3 ; i++ ) {
                float a    = A[j][i];
                float absa = fabs( a );
                if ( absa > maxabsa ) {
                    maxa    = a;
                    maxabsa = absa;
                    maxi    = i;
                }
            }

            // singular?
            if( maxabsa < 1e-15 ) {
                return false;
            }

            int i = maxi;

            // swap j-th row with i-th row and
            // normalize j-th row
            for(int jj = j ; jj < 3 ; ++jj) {
                float tmp = A[jj][j];
                A[jj][j]  = A[jj][i];
                A[jj][i]  = tmp;
                A[jj][j] /= maxa;
            }
            float tmp = b[j];
            b[j]  = b[i];
            b[i]  = tmp;
            b[j] /= maxa;

            // elimination
            for(int ii = j+1 ; ii < 3 ; ++ii) {
                float x = A[j][ii];
                for( int jj = j ; jj < 3 ; jj++ ) {
                    A[jj][ii] -= x * A[jj][j];
                }
                b[ii] -= x * b[j] ;
            }
    }

    // backward substitution
    for( int i = 2 ; i > 0 ; i-- ) {
        float x = b[i] ;
        for( int ii = i-1 ; ii >= 0 ; ii-- ) {
            b[ii] -= x * A[i][ii];
        }
    }
    return true;
}
#endif // not CLOSED_FORM_SOLVE

