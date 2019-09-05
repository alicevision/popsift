/*
 * Copyright 2019, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

// #include <iomanip>
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <algorithm>
// #include <limits>
// #include <set>
// #include <iterator>

// #include <stdlib.h> // for rand()

// #include <stdlib.h>
// #include <errno.h>
// #include <math_constants.h>

#include "features.h"
#include "sift_extremum.h"
// #include "common/assist.h"
// #include "common/debug_macros.h"

namespace popsift {

/*************************************************************
 * class PqtAnn
 * Inspired by the paper 10.1109/CVPR.2016.223
 *************************************************************/

struct Level;

class PqtAnn
{
    Level* _level1;

public:
    PqtAnn( const std::vector<Descriptor*>& descriptorList );
    ~PqtAnn( );

    void run( );

    void findMatch( const Descriptor& desc );
};

} // namespace popsift

