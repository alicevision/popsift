#pragma once

#include <cuda_runtime.h>
#include <stdio.h>

__device__
inline bool solve( float i[3][3], float3& b )
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
        vout[y] += ( i[y][0] * b.x );
        vout[y] += ( i[y][1] * b.y );
        vout[y] += ( i[y][2] * b.z );
    }
    b.x = vout[0];
    b.y = vout[1];
    b.z = vout[2];

    return true;
}

