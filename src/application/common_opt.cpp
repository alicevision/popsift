/*
 * Copyright 2021, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "common_opt.h"

// #include <popsift/common/device_prop.h>
// #include <popsift/features.h>
// #include <popsift/popsift.h>
// #include <popsift/sift_config.h>
// #include <popsift/version.hpp>

// #include <boost/filesystem.hpp>

// #include <cmath>
// #include <cstdlib>
// #include <fstream>
// #include <iomanip>
// #include <iostream>
// #include <list>
// #include <sstream>
// #include <stdexcept>
#include <string>

using namespace std;
using namespace boost::program_options;

void option_init_parameters( popsift::Config& config, options_description& parameters )
{
    parameters.add_options()
        ("octaves",
         value<int>(&config.octaves)->default_value(config.getOctaves()),
         "Number of octaves")
        ("levels",
         value<int>(&config.levels)->default_value(config.getLevels()),
         "Number of levels per octave")
        ("sigma",
         value<float>()->notifier([&](float f) { config.setSigma(f); })->default_value(config.getSigma()),
         "Initial sigma value")
        ("threshold",
         value<float>()->notifier([&](float f) { config.setThreshold(f); })->default_value(config.getThreshold()),
         popsift::Config::getPeakThreshUsage().c_str() )
        ("edge-threshold",
         value<float>()->notifier([&](float f) { config.setEdgeLimit(f); })->default_value(config.getEdgeLimit()),
         popsift::Config::getEdgeThreshUsage().c_str() )
        ("edge-limit",
         value<float>()->notifier([&](float f) { config.setEdgeLimit(f); }),
         "synonym to --edge-threshold" )
        ("downsampling",
         value<float>()->notifier([&](float f) { config.setDownsampling(f); })->default_value(config.getDownsampling()),
         "Downscale width and height of input by 2^N")
        ("initial-blur",
         value<float>()->notifier([&](float f) {config.setInitialBlur(f); })->default_value(config.getInitialBlur()),
         "Assume initial blur, subtract when blurring first time");
}

void option_init_modes( popsift::Config& config, options_description& modes )
{
    modes.add_options()
        ( "gauss-mode",
          value<std::string>()->notifier([&](const std::string& s) { config.setGaussMode(s); }),
          popsift::Config::getGaussModeUsage() )
        // "Choice of span (1-sided) for Gauss filters. Default is VLFeat-like computation depending on sigma. "
        // "Options are: vlfeat, relative, relative-all, opencv, fixed9, fixed15"
        ( "desc-mode",
          value<std::string>()->notifier([&](const std::string& s) { config.setDescMode(s); }),
          popsift::Config::getDescModeUsage() )
        ( "popsift-mode",
          bool_switch()->notifier([&](bool b) { if(b) config.setMode(popsift::Config::PopSift); }),
         "During the initial upscale, shift pixels by 1. In extrema refinement, steps up to 0.6, do not reject points when reaching max iterations, "
         "first contrast threshold is .8 * peak thresh. Shift feature coords octave 0 back to original pos.")
        ( "vlfeat-mode",
          bool_switch()->notifier([&](bool b) { if(b) config.setMode(popsift::Config::VLFeat); }),
          "During the initial upscale, shift pixels by 1. That creates a sharper upscaled image. "
          "In extrema refinement, steps up to 0.6, levels remain unchanged, "
          "do not reject points when reaching max iterations, "
          "first contrast threshold is .8 * peak thresh.")
        ( "opencv-mode",
          bool_switch()->notifier([&](bool b) { if(b) config.setMode(popsift::Config::OpenCV); }),
         "During the initial upscale, shift pixels by 0.5. "
         "In extrema refinement, steps up to 0.5, "
         "reject points when reaching max iterations, "
         "first contrast threshold is floor(.5 * peak thresh). "
         "Computed filter width are lower than VLFeat/PopSift")
        ( "direct-scaling",
          bool_switch()->notifier([&](bool b) { if(b) config.setScalingMode(popsift::Config::ScaleDirect); }),
          "Direct each octave from upscaled orig instead of blurred level.")
        ( "norm-multi",
          value<int>()->notifier([&](int i) {config.setNormalizationMultiplier(i); }),
          "Multiply the descriptor by pow(2,<int>).")
        ( "norm-mode",
          value<std::string>()->notifier([&](const std::string& s) { config.setNormMode(s); }),
          popsift::Config::getNormModeUsage() )
        ( "root-sift",
          bool_switch()->notifier([&](bool b) { if(b) config.setNormMode(popsift::Config::RootSift); }),
          "synonym to --norm-mode=RootSift" )
        ( "filter-max-extrema",
          value<int>()->notifier([&](int f) {config.setFilterMaxExtrema(f); }),
          "Approximate max number of extrema.")
        ( "filter-grid",
          value<int>()->notifier([&](int f) {config.setFilterGridSize(f); }),
          "Grid edge length for extrema filtering (ie. value 4 leads to a 4x4 grid)")
        ( "filter-sort",
          value<std::string>()->notifier([&](const std::string& s) {config.setFilterSorting(s); }),
          popsift::Config::getFilterGridModeUsage() );

}

