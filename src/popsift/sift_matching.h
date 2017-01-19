/*
* Copyright 2017, Simula Research Laboratory
*
* This Source Code Form is subject to the terms of the Mozilla Public
* License, v. 2.0. If a copy of the MPL was not distributed with this
* file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include "sift_conf.h"
#include "sift_extremum.h"
#include <tuple>
#include <vector>

class PopSift;

namespace popsift {

using Pair = std::tuple<unsigned, unsigned>;
using PairList = std::vector<Pair>;

PairList cpu_matching(const Features& fa, const Features& fb);

struct tmp_ret {

};

class Matching
{
public:
    Matching(Config& config);
    ~Matching();

    tmp_ret Match(PopSift& a, PopSift& b);

private:
    const Config& config;

    void getFlatDeviceDesc(PopSift& ps, Descriptor*& desc_out_device, int* desc_count);
};

}
