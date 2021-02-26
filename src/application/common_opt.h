/*
 * Copyright 2021, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <boost/program_options.hpp>

#include <popsift/sift_conf.h>

void option_init_parameters( popsift::Config& config,
                             boost::program_options::options_description& parameters );
void option_init_modes( popsift::Config& config,
                        boost::program_options::options_description& modes );

