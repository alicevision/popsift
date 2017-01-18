/*
* Copyright 2017, Simula Research Laboratory
*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "sift_matching.h"
#include "assist.h"

namespace popsift {

Matching::Matching(Config& config) {

}

Matching::~Matching() {

}

__host__
tmp_ret Matching::Match(Features* a, Features* b) {
    for (auto it = a->begin(); it != a->end(); it++) {
        
    }
}

}
