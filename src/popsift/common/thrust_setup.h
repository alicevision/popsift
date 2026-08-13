/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

/* Everything a translation unit needs before it can call a Thrust algorithm on
 * one of PopSift's streams. Grid filtering is the only user today, but Thrust is
 * usable elsewhere.
 */

#include <thrust/execution_policy.h>
#include <thrust/version.h>

/* The stream-bound parallel execution policy. It lives in thrust::cuda on
 * NVIDIA and in thrust::hip in the AMD build of Thrust. Fully qualified from
 * the global namespace, because inside namespace popsift the name cuda would
 * otherwise resolve to popsift::cuda (common/debug_macros.h).
 */
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
#define POPSIFT_THRUST_PAR ::thrust::hip::par
#else
#define POPSIFT_THRUST_PAR ::thrust::cuda::par
#endif
