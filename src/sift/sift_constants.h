#pragma once

#ifndef INF
#define INF               (1<<29)
#endif
#ifndef NINF
#define NINF              (-INF)
#endif
#ifndef M_PI
#define M_PI  3.1415926535897932384626433832F
#endif
#ifndef M_PI2
#define M_PI2 (2.0F * M_PI)
#endif

#define M_4RPI               (4.0f / M_PI)

#define DESC_MIN_FLOAT               1E-15F

#define GAUSS_ALIGN  16
#define GAUSS_LEVELS 12

#define ORI_V1_NUM_THREADS 16
#define ORI_NBINS          36
#define ORI_WINFACTOR      1.5F

#define DESC_BINS        8
#define DESC_MAGNIFY           3.0f
// #define DESC_MAGNIFY           6.0f

/* Define this to prefer L1 norm in descriptor normalization
 * instead of default L2 norm.
 */
#undef DESC_USE_ROOT_SIFT

/* CUDA Dynamic Parallelism does not work with all devices.
 * Try to maintain the possibility to compile without this.
 */
#define USE_DYNAMIC_PARALLELISM


/*
 * USE_OPENCV_INTERPRETATION
 * - RGB conversion relies on integer arithmetic
 * - resize function is nearest neighbour instead of interpolation
 * - we modified the clamping behaviour in OpenCV to something that CUDA can do
 */
#define USE_OPENCV_INTERPRETATION

#define DEBUG_ORI_GRID

