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
#include <list>
#include <string>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <popsift/popsift.h>
#include <popsift/features.h>
#include <popsift/sift_conf.h>
#include <popsift/common/device_prop.h>

#ifdef USE_DEVIL
#include <devil_cpp_wrapper.hpp>
#endif
#include "pgmread.h"

#ifdef USE_NVTX
#include <nvToolsExtCuda.h>
#else
#define nvtxRangePushA(a)
#define nvtxRangePop()
#endif

using namespace std;

static bool print_dev_info  = false;
static bool print_time_info = false;
static bool write_as_uchar  = false;
static bool dont_write      = false;
static bool pgmread_loading = false;

static void parseargs(int argc, char** argv, popsift::Config& config, string& lFile, string& rFile) {
    using namespace boost::program_options;

    options_description options("Options");
    {
        options.add_options()
            ("help,h", "Print usage")
            ("verbose,v", bool_switch()->notifier([&](bool i) {if(i) config.setVerbose(); }), "")
            ("log", bool_switch()->notifier([&](bool i) {if(i) config.setLogMode(popsift::Config::All); }), "Write debugging files")

            ("left,l",  value<std::string>(&lFile)->required(), "\"Left\"  input file")
            ("right,r", value<std::string>(&rFile)->required(), "\"Right\" input file");
    
    }
    options_description parameters("Parameters");
    {
        parameters.add_options()
            ("octaves", value<int>(&config.octaves), "Number of octaves")
            ("levels", value<int>(&config.levels), "Number of levels per octave")
            ("sigma", value<float>()->notifier([&](float f) { config.setSigma(f); }), "Initial sigma value")

            ("threshold", value<float>()->notifier([&](float f) { config.setThreshold(f); }), "Contrast threshold")
            ("edge-threshold", value<float>()->notifier([&](float f) { config.setEdgeLimit(f); }), "On-edge threshold")
            ("edge-limit", value<float>()->notifier([&](float f) { config.setEdgeLimit(f); }), "On-edge threshold")
            ("downsampling", value<float>()->notifier([&](float f) { config.setDownsampling(f); }), "Downscale width and height of input by 2^N")
            ("initial-blur", value<float>()->notifier([&](float f) {config.setInitialBlur(f); }), "Assume initial blur, subtract when blurring first time");
    }
    options_description modes("Modes");
    {
    modes.add_options()
        ("gauss-mode", value<std::string>()->notifier([&](const std::string& s) { config.setGaussMode(s); }),
        "Choice of span (1-sided) for Gauss filters. Default is VLFeat-like computation depending on sigma. "
        "Options are: vlfeat, relative, opencv, fixed9, fixed15")
        ("desc-mode", value<std::string>()->notifier([&](const std::string& s) { config.setDescMode(s); }),
        "Choice of descriptor extraction modes:\n"
        "loop, iloop, grid, igrid, notile\n"
	"Default is loop\n"
        "loop is OpenCV-like horizontal scanning, computing only valid points, grid extracts only useful points but rounds them, iloop uses linear texture and rotated gradiant fetching. igrid is grid with linear interpolation. notile is like igrid but avoids redundant gradiant fetching.")
        ("popsift-mode", bool_switch()->notifier([&](bool b) { if(b) config.setMode(popsift::Config::PopSift); }),
        "During the initial upscale, shift pixels by 1. In extrema refinement, steps up to 0.6, do not reject points when reaching max iterations, "
        "first contrast threshold is .8 * peak thresh. Shift feature coords octave 0 back to original pos.")
        ("vlfeat-mode", bool_switch()->notifier([&](bool b) { if(b) config.setMode(popsift::Config::VLFeat); }),
        "During the initial upscale, shift pixels by 1. That creates a sharper upscaled image. "
        "In extrema refinement, steps up to 0.6, levels remain unchanged, "
        "do not reject points when reaching max iterations, "
        "first contrast threshold is .8 * peak thresh.")
        ("opencv-mode", bool_switch()->notifier([&](bool b) { if(b) config.setMode(popsift::Config::OpenCV); }),
        "During the initial upscale, shift pixels by 0.5. "
        "In extrema refinement, steps up to 0.5, "
        "reject points when reaching max iterations, "
        "first contrast threshold is floor(.5 * peak thresh). "
        "Computed filter width are lower than VLFeat/PopSift")
        ("direct-scaling", bool_switch()->notifier([&](bool b) { if(b) config.setScalingMode(popsift::Config::ScaleDirect); }),
         "Direct each octave from upscaled orig instead of blurred level.")
        ("root-sift", bool_switch()->notifier([&](bool b) { if(b) config.setUseRootSift(true); }),
        "Use the L1-based norm for OpenMVG rather than L2-based as in OpenCV")
        ("norm-multi", value<int>()->notifier([&](int i) {config.setNormalizationMultiplier(i); }), "Multiply the descriptor by pow(2,<int>).")
        ("filter-max-extrema", value<int>()->notifier([&](int f) {config.setFilterMaxExtrema(f); }), "Approximate max number of extrema.")
        ("filter-grid", value<int>()->notifier([&](int f) {config.setFilterGridSize(f); }), "Grid edge length for extrema filtering (ie. value 4 leads to a 4x4 grid)")
        ("filter-sort", value<std::string>()->notifier([&](const std::string& s) {config.setFilterSorting(s); }), "Sort extrema in each cell by scale, either random (default), up or down");

    }
    options_description informational("Informational");
    {
        informational.add_options()
        ("print-gauss-tables", bool_switch()->notifier([&](bool b) { if(b) config.setPrintGaussTables(); }), "A debug output printing Gauss filter size and tables")
        ("print-dev-info", bool_switch(&print_dev_info)->default_value(false), "A debug output printing CUDA device information")
        ("print-time-info", bool_switch(&print_time_info)->default_value(false), "A debug output printing image processing time after load()")
        ("write-as-uchar", bool_switch(&write_as_uchar)->default_value(false), "Output descriptors rounded to int Scaling to sensible ranges is not automatic, should be combined with --norm-multi=9 or similar")
        ("dont-write", bool_switch(&dont_write)->default_value(false), "Suppress descriptor output")
        ("pgmread-loading", bool_switch(&pgmread_loading)->default_value(false), "Use the old image loader instead of LibDevIL")
        ;
        
        //("test-direct-scaling")
    }

    options_description all("Allowed options");
    all.add(options).add(parameters).add(modes).add(informational);
    variables_map vm;
    
    try
    {    
       store(parse_command_line(argc, argv, all), vm);

       if (vm.count("help")) {
           std::cout << all << '\n';
           exit(1);
       }

        notify(vm); // Notify does processing (e.g., raise exceptions if required args are missing)
    }
    catch(boost::program_options::error& e)
    {
        std::cerr << "Error: " << e.what() << std::endl << std::endl;
        std::cerr << "Usage:\n\n" << all << std::endl;
        exit(EXIT_FAILURE);
    }
}


static void collectFilenames( list<string>& inputFiles, const boost::filesystem::path& inputFile )
{
    vector<boost::filesystem::path> vec;
    std::copy( boost::filesystem::directory_iterator( inputFile ),
               boost::filesystem::directory_iterator(),
               std::back_inserter(vec) );
    for( auto it = vec.begin(); it!=vec.end(); it++ ) {
        if( boost::filesystem::is_regular_file( *it ) ) {
            string s( it->c_str() );
            inputFiles.push_back( s );
        } else if( boost::filesystem::is_directory( *it ) ) {
            collectFilenames( inputFiles, *it );
        }
    }
}

SiftJob* process_image( const string& inputFile, PopSift& PopSift )
{
    int w;
    int h;
    unsigned char* image_data;
    SiftJob* job;

    nvtxRangePushA( "load and convert image" );
#ifdef USE_DEVIL
    if( not pgmread_loading )
    {
        ilImage img;
        if( img.Load( inputFile.c_str() ) == false ) {
            cerr << "Could not load image " << inputFile << endl;
            return 0;
        }
        if( img.Convert( IL_LUMINANCE ) == false ) {
            cerr << "Failed converting image " << inputFile << " to unsigned greyscale image" << endl;
            exit( -1 );
        }
        w = img.Width();
        h = img.Height();
        cout << "Loading " << w << " x " << h << " image " << inputFile << endl;
        image_data = img.GetData();

        nvtxRangePop( );

        // PopSift.init( w, h );
        job = PopSift.enqueue( w, h, image_data );

        img.Clear();
    }
    else
#endif
    {
        image_data = readPGMfile( inputFile, w, h );
        if( image_data == 0 ) {
            exit( -1 );
        }

        nvtxRangePop( );

        // PopSift.init( w, h );
        job = PopSift.enqueue( w, h, image_data );

        delete [] image_data;
    }

    return job;
}

int main(int argc, char **argv)
{
    cudaDeviceReset();

    popsift::Config config;
    string         lFile = "";
    string         rFile = "";
    const char*    appName   = argv[0];

    try {
        parseargs( argc, argv, config, lFile, rFile ); // Parse command line
        std::cout << lFile << " <-> " << rFile << std::endl;
    }
    catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        exit(1);
    }

    if( boost::filesystem::exists( lFile ) ) {
        if( not boost::filesystem::is_regular_file( lFile ) ) {
            cout << "Input file " << lFile << " is not a regular file, nothing to do" << endl;
            exit( -1 );
        }
    }

    if( boost::filesystem::exists( rFile ) ) {
        if( not boost::filesystem::is_regular_file( rFile ) ) {
            cout << "Input file " << rFile << " is not a regular file, nothing to do" << endl;
            exit( -1 );
        }
    }

    popsift::cuda::device_prop_t deviceInfo;
    deviceInfo.set( 0, print_dev_info );
    if( print_dev_info ) deviceInfo.print( );

    PopSift PopSift( config, popsift::Config::MatchingMode );

    SiftJob* lJob = process_image( lFile, PopSift );
    SiftJob* rJob = process_image( rFile, PopSift );

    popsift::DeviceFeatures* lFeatures = dynamic_cast<popsift::DeviceFeatures*>( lJob->get() );
    cout << "Number of features:    " << lFeatures->getFeatureCount() << endl;
    cout << "Number of descriptors: " << lFeatures->getDescriptorCount() << endl;

    popsift::DeviceFeatures* rFeatures = dynamic_cast<popsift::DeviceFeatures*>( rJob->get() );
    cout << "Number of features:    " << rFeatures->getFeatureCount() << endl;
    cout << "Number of descriptors: " << rFeatures->getDescriptorCount() << endl;

    lFeatures->match( rFeatures );

    delete lFeatures;
    delete rFeatures;

    PopSift.uninit( );
}

