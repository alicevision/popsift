/*
 * Copyright 2016-2017, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "sift_config.h"
#include "sift_extremum.h"
#include "sift_pyramid.h"

namespace popsift
{
/* do nothing unless we have CUDA v 8 or newer */
int Pyramid::extrema_filter_grid( const Config& conf, int ext_total )
{
    return ext_total;
}
}; // namespace popsift

