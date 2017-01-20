/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <iomanip>
#include <stdlib.h>
#include <stdexcept>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <popsift/popsift.h>
#include <popsift/sift_conf.h>
#include <popsift/common/device_prop.h>

#include "pgmread.h"
#include "popsift/sift_pyramid.h"
#include "popsift/sift_matching.h"

using namespace std;

static bool print_dev_info  = false;
static bool print_time_info = false;
static bool write_as_uchar  = false;

static void parseargs(int argc, char** argv, popsift::Config& config, string& inputFile, string& matchFile) {
    using namespace boost::program_options;

    options_description options("Options");
    options.add_options()
        ("help,h", "Print usage")
        ("verbose,v", bool_switch()->notifier([&](bool v) { if (v) config.setVerbose(); }), "")
        ("log,l", bool_switch()->notifier([&](bool v) { if(v) config.setLogMode(popsift::Config::All); }), "Write debugging files")
        ("input-file,i", value<std::string>(&inputFile)->required(), "Input file")
        ("match-file,m", value<std::string>(&matchFile), "Match file. If provided PopSIFT will attempt to match the two input-file with match-file.");
    

    options_description parameters("Parameters");
    parameters.add_options()
        ("octaves", value<int>(&config.octaves), "Number of octaves")
        ("levels", value<int>(&config.levels), "Number of levels per octave")
        ("sigma", value<float>()->notifier([&](float f) { config.setSigma(f); }), "Initial sigma value")

        ("threshold", value<float>()->notifier([&](float f) { config.setThreshold(f); }), "Contrast threshold")
        ("edge-threshold", value<float>()->notifier([&](float f) { config.setEdgeLimit(f); }), "On-edge threshold")
        ("edge-limit", value<float>()->notifier([&](float f) { config.setEdgeLimit(f); }), "On-edge threshold")
        ("downsampling", value<float>()->notifier([&](float f) { config.setDownsampling(f); }), "Downscale width and height of input by 2^N")
        ("initial-blur", value<float>()->notifier([&](float f) {config.setInitialBlur(f); }), "Assume initial blur, subtract when blurring first time");

    options_description modes("Modes");
    modes.add_options()
        ("popsift-mode", bool_switch()->notifier([&](bool v) { if(v) config.setMode(popsift::Config::PopSift); }),
            "During the initial upscale, shift pixels by 1. In extrema refinement, steps up to 0.6, do not reject points when reaching max iterations, "
            "first contrast threshold is .8 * peak thresh. Shift feature coords octave 0 back to original pos.")
        ("vlfeat-mode", bool_switch()->notifier([&](bool v) { if (v) config.setMode(popsift::Config::VLFeat); }),
            "During the initial upscale, shift pixels by 1. That creates a sharper upscaled image. "
            "In extrema refinement, steps up to 0.6, levels remain unchanged, "
            "do not reject points when reaching max iterations, "
            "first contrast threshold is .8 * peak thresh.")
        ("opencv-mode", bool_switch()->notifier([&](bool v) { if (v) config.setMode(popsift::Config::OpenCV); }),
            "During the initial upscale, shift pixels by 0.5. "
            "In extrema refinement, steps up to 0.5, "
            "reject points when reaching max iterations, "
            "first contrast threshold is floor(.5 * peak thresh). "
            "Computed filter width are lower than VLFeat/PopSift")
        ("root-sift", bool_switch()->notifier([&](bool v) { if (v) config.setUseRootSift(true); }),
            "Use the L1-based norm for OpenMVG rather than L2-based as in OpenCV")
        ("norm-multi", value<int>()->notifier([&](int i) {config.setNormalizationMultiplier(i); }), "Multiply the descriptor by pow(2,<int>).")
        ("dp-off", bool_switch()->notifier([&](bool v) { if (v) config.setDPOrientation(false); config.setDPDescriptors(false); }), "Switch all CUDA Dynamic Parallelism off.")
        ("dp-ori-off", bool_switch()->notifier([&](bool v) { if (v) config.setDPOrientation(false); }), "Switch DP off for orientation computation")
        ("dp-desc-off", bool_switch()->notifier([&](bool v) { if (v) config.setDPDescriptors(false); }), "Switch DP off for descriptor computation");


    options_description informational("Informational");
    informational.add_options()
        ("print-gauss-tables", bool_switch()->notifier([&](bool v) { if (v) config.setPrintGaussTables(); }), "A debug output printing Gauss filter size and tables")
        ("print-dev-info", bool_switch(&print_dev_info)->default_value(false), "A debug output printing CUDA device information")
        ("print-time-info", bool_switch(&print_time_info)->default_value(false), "A debug output printing image processing time after load()")
        ("write-as-uchar", bool_switch(&write_as_uchar)->default_value(false), "Output descriptors rounded to int Scaling to sensible ranges is not automatic, should be combined with --norm-multi=9 or similar");
        
        //("test-direct-scaling")

    options_description all("Allowed options");
    all.add(options).add(parameters).add(modes).add(informational);
    variables_map vm;
    store(parse_command_line(argc, argv, all), vm);

    if (vm.count("help")) {
        std::cout << all << '\n';
        exit(1);
    }
    // Notify does processing (e.g., raise exceptions if required args are missing)
    notify(vm);
}

popsift_ptr extractFeatures(string& img, popsift::Config& config) {
    int w;
    int h;
    auto image_data = readPGMfile(img, w, h);
    if (!image_data) {
        exit(-1);
    }

    // cerr << "Input image size: "
    //      << w << "X" << h
    //      << " filename: " << boost::filesystem::path(inputFile).filename() << endl;

    popsift::cuda::device_prop_t deviceInfo;
    deviceInfo.set(0, print_dev_info);
    if (print_dev_info) deviceInfo.print();

    
    popsift_ptr popsift(new PopSift(config), popsift_deleter());

    popsift->init(0, w, h, print_time_info);
    popsift->execute(0, image_data.get(), print_time_info);
    
    popsift::Features* feature_list = popsift->getFeatures();
    std::ofstream of("output-features.txt");
    feature_list->print(of, write_as_uchar);

    return popsift;
}

int main(int argc, char **argv)
{
    cudaDeviceReset();

    popsift::Config config;
    string         inputFile = "";
    string         matchFile = "";
    const char*    appName   = argv[0];

    try {
        parseargs(argc, argv, config, inputFile, matchFile); // Parse command line
        std::cout << inputFile << std::endl;
    }
    catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        exit(1);
    }
    
    
    popsift_ptr sift_a = extractFeatures(inputFile, config);
    if (!matchFile.empty()) {
        popsift_ptr sift_b = extractFeatures(matchFile, config);

        popsift::Matching matcher(config);
        matcher.Match(*sift_a, *sift_b);
        /*
        auto matches = popsift::cpu_matching(*sift_a->getFeatures(), *sift_b->getFeatures());
        for (const auto& m : matches)
            cout << std::get<0>(m) << " " << std::get<1>(m) << endl;
        */
    }
    system("Pause");
    return 0;
}
