// HIP-only shim for CUDA's <math_constants.h>. ROCm ships no equivalent header,
// so provide the CUDART_* constants PopSift uses, with CUDA's values. On the
// HIP include path only; the NVIDIA build uses the real CUDA header.
#pragma once

#include <math.h>

#ifndef CUDART_INF_F
#define CUDART_INF_F  __int_as_float(0x7f800000)
#endif
#ifndef CUDART_NAN_F
#define CUDART_NAN_F  __int_as_float(0x7fffffff)
#endif
#ifndef CUDART_PI_F
#define CUDART_PI_F   3.141592654f
#endif
