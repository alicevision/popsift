/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <popsift/common/device_prop.h>
#include <popsift/features.h>
#include <popsift/popsift.h>
#include <popsift/sift_conf.h>
#include <popsift/sift_config.h>
#include <popsift/version.hpp>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef USE_DEVIL
#include <devil_cpp_wrapper.hpp>
#endif
#include "pgmread.h"

#if POPSIFT_IS_DEFINED(POPSIFT_USE_NVTX)
#include <nvToolsExtCuda.h>
#else
#define nvtxRangePushA(a)
#define nvtxRangePop()
#endif

using namespace std;

static bool print_dev_info  = false;
static bool print_time_info = false;
static bool write_as_uchar  = false;
static bool write_with_ori  = false;
static bool dont_write      = false;
static bool pgmread_loading = false;
static bool float_mode      = false;

static void parseargs(int argc, char** argv, popsift::Config& config, string& inputFile) {
    using namespace boost::program_options;

    options_description options("Options");
    {
        options.add_options()
            ("help,h", "Print usage")
            ("verbose,v", bool_switch()->notifier([&](bool i) {if(i) config.setVerbose(); }), "")
            ("log,l", bool_switch()->notifier([&](bool i) {if(i) config.setLogMode(popsift::Config::All); }), "Write debugging files")

            ("input-file,i", value<std::string>(&inputFile)->required(), "Input file");
    
    }
    options_description parameters("Parameters");
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
             "Contrast threshold")
            ("edge-threshold",
             value<float>()->notifier([&](float f) { config.setEdgeLimit(f); })->default_value(config.getEdgeLimit()),
             "On-edge threshold")
            ("edge-limit",
             value<float>()->notifier([&](float f) { config.setEdgeLimit(f); }),
             "On-edge threshold")
            ("downsampling",
             value<float>()->notifier([&](float f) { config.setDownsampling(f); })->default_value(config.getDownsampling()),
             "Downscale width and height of input by 2^N")
            ("initial-blur",
             value<float>()->notifier([&](float f) {config.setInitialBlur(f); })->default_value(config.getInitialBlur()),
             "Assume initial blur, subtract when blurring first time");
    }
    options_description modes("Modes");
    {
    modes.add_options()
        ( "gauss-mode", value<std::string>()->notifier([&](const std::string& s) { config.setGaussMode(s); }),
          popsift::Config::getGaussModeUsage() )
        // "Choice of span (1-sided) for Gauss filters. Default is VLFeat-like computation depending on sigma. "
        // "Options are: vlfeat, relative, relative-all, opencv, fixed9, fixed15"
        ( "desc-mode", value<std::string>()->notifier([&](const std::string& s) { config.setDescMode(s); }),
          popsift::Config::getDescModeUsage() )
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
        ("norm-multi", value<int>()->notifier([&](int i) {config.setNormalizationMultiplier(i); }), "Multiply the descriptor by pow(2,<int>).")
        ( "norm-mode", value<std::string>()->notifier([&](const std::string& s) { config.setNormMode(s); }),
          popsift::Config::getNormModeUsage() )
        ( "root-sift", bool_switch()->notifier([&](bool b) { if(b) config.setNormMode(popsift::Config::RootSift); }),
          "synonym to --norm-mode=RootSift" )
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
        ("write-as-uchar", bool_switch(&write_as_uchar)->default_value(false), "Output descriptors rounded to int.\n"
         "Scaling to sensible ranges is not automatic, should be combined with --norm-multi=9 or similar")
        ("write-with-ori", bool_switch(&write_with_ori)->default_value(false), "Output points are written with sigma and orientation.\n")
        ("dont-write", bool_switch(&dont_write)->default_value(false), "Suppress descriptor output")
        ("pgmread-loading", bool_switch(&pgmread_loading)->default_value(false), "Use the old image loader instead of LibDevIL")
        ("float-mode", bool_switch(&float_mode)->default_value(false), "Upload image to GPU as float instead of byte")
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
           exit(EXIT_SUCCESS);
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
    for (const auto& currPath : vec)
    {
        if( boost::filesystem::is_regular_file(currPath) )
        {
            inputFiles.push_back( currPath.string() );
        }
        else if( boost::filesystem::is_directory(currPath) )
        {
            collectFilenames( inputFiles, currPath);
        }
    }
}

SiftJob* process_image( const string& inputFile, PopSift& PopSift )
{
    SiftJob* job;
    unsigned char* image_data;

#ifdef USE_DEVIL
    if( ! pgmread_loading )
    {
        if( float_mode )
        {
            cerr << "Cannot combine float-mode test with DevIL image reader" << endl;
            exit( -1 );
        }

        nvtxRangePushA( "load and convert image - devil" );

        ilImage img;
        if( img.Load( inputFile.c_str() ) == false ) {
            cerr << "Could not load image " << inputFile << endl;
            return 0;
        }
        if( img.Convert( IL_LUMINANCE ) == false ) {
            cerr << "Failed converting image " << inputFile << " to unsigned greyscale image" << endl;
            exit( -1 );
        }
        const auto w = img.Width();
        const auto h = img.Height();
        cout << "Loading " << w << " x " << h << " image " << inputFile << endl;

        image_data = img.GetData();

        nvtxRangePop( ); // "load and convert image - devil"

        job = PopSift.enqueue( w, h, image_data );

        img.Clear();
    }
    else
#endif
    {
        nvtxRangePushA( "load and convert image - pgmread" );
        int w{};
        int h{};
        image_data = readPGMfile( inputFile, w, h );
        if( image_data == nullptr ) {
            exit( EXIT_FAILURE );
        }

        nvtxRangePop( ); // "load and convert image - pgmread"

        if( ! float_mode )
        {
            // PopSift.init( w, h );
            job = PopSift.enqueue( w, h, image_data );

            delete [] image_data;
        }
        else
        {
            auto f_image_data = new float [w * h];
            for( int i=0; i<w*h; i++ )
            {
                f_image_data[i] = float( image_data[i] ) / 256.0f;
            }
            job = PopSift.enqueue( w, h, f_image_data );

            delete [] image_data;
            delete [] f_image_data;
        }
    }

    return job;
}

void read_job( SiftJob* job, bool really_write )
{
    popsift::Features* feature_list = job->get();
    cerr << "Number of feature points: " << feature_list->getFeatureCount()
         << " number of feature descriptors: " << feature_list->getDescriptorCount()
         << endl;

    if( really_write ) {
        nvtxRangePushA( "Writing features to disk" );

        std::ofstream of( "output-features.txt" );
        feature_list->print( of, write_as_uchar, write_with_ori );
    }
    delete feature_list;

    if( really_write ) {
        nvtxRangePop( ); // Writing features to disk
    }
}

int main(int argc, char **argv)
{
    cudaDeviceReset();

    popsift::Config config;
    list<string>   inputFiles;
    string         inputFile{};

    std::cout << "PopSift version: " << POPSIFT_VERSION_STRING << std::endl;

    try {
        parseargs( argc, argv, config, inputFile ); // Parse command line
        std::cout << inputFile << std::endl;
    }
    catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    if( boost::filesystem::exists( inputFile ) ) {
        if( boost::filesystem::is_directory( inputFile ) ) {
            cout << "BOOST " << inputFile << " is directory" << endl;
            collectFilenames( inputFiles, inputFile );
            if( inputFiles.empty() ) {
                cerr << "No files in directory, nothing to do" << endl;
                return EXIT_SUCCESS;
            }
        } else if( boost::filesystem::is_regular_file( inputFile ) ) {
            inputFiles.push_back( inputFile );
        } else {
            cout << "Input file is neither regular file nor directory, nothing to do" << endl;
            return EXIT_FAILURE;
        }
    }

    popsift::cuda::device_prop_t deviceInfo;
    deviceInfo.set( 0, print_dev_info );
    if( print_dev_info ) deviceInfo.print( );

    PopSift PopSift( config,
                     popsift::Config::ExtractingMode,
                     float_mode ? PopSift::FloatImages : PopSift::ByteImages );

    std::queue<SiftJob*> jobs;
    for(const auto& currFile : inputFiles)
    {
        SiftJob* job = process_image( currFile, PopSift );
        jobs.push( job );
    }

    while( !jobs.empty() )
    {
        SiftJob* job = jobs.front();
        jobs.pop();
        if( job ) {
            read_job( job, ! dont_write );
            delete job;
        }
    }

    PopSift.uninit( );

    return EXIT_SUCCESS;
}
