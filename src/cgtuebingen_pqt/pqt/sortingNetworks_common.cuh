/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



#ifndef SORTINGNETWORKS_COMMON_CUH
#define SORTINGNETWORKS_COMMON_CUH



#include "sortingNetworks_common.h"

//Enables maximum occupancy
#define SHARED_SIZE_LIMIT 1024U

//Map to single instructions on G8x / G9x / G100
#define    UMUL(a, b) __umul24((a), (b))
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )



__device__ inline void Comparator(
	float &keyA,
    uint &valA,
    float &keyB,
    uint &valB,
    uint dir
)
{
    uint t;
    float keyT;

    if ((keyA > keyB) == dir)
    {
        keyT = keyA;
        keyA = keyB;
        keyB = keyT;
        t = valA;
        valA = valB;
        valB = t;
    }
}



#endif
