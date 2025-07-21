/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
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

#include <OpenImageIO/imageio.h>

#include "pgmread.h"

using namespace OIIO;
using namespace std;

static bool print_dev_info  = false;
static bool print_time_info = false;
static bool write_as_uchar  = false;
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
        ( "gauss-mode", value<std::string>()->notifier([&](const std::string& s) { config.setGaussMode(s); }),
          popsift::Config::getGaussModeUsage() )
        // "Choice of span (1-sided) for Gauss filters. Default is VLFeat-like computation depending on sigma. "
        // "Options are: vlfeat, relative, relative-all, opencv"
        ("desc-mode", value<std::string>()->notifier([&](const std::string& s) { config.setDescMode(s); }),
         popsift::Config::getDescModeUsage())
        ("popsift-mode", bool_switch()->notifier([&](bool b) { if(b) config.setMode(popsift::Config::RefineInOctave); }),
        "During the initial upscale, shift pixels by 1. In extrema refinement, steps up to 0.6, do not reject points when reaching max iterations, "
        "first contrast threshold is .8 * peak thresh. Shift feature coords octave 0 back to original pos.")
        ("vlfeat-mode", bool_switch()->notifier([&](bool b) { if(b) config.setMode(popsift::Config::RefineInLevel); }),
        "During the initial upscale, shift pixels by 1. That creates a sharper upscaled image. "
        "In extrema refinement, steps up to 0.6, levels remain unchanged, "
        "do not reject points when reaching max iterations, "
        "first contrast threshold is .8 * peak thresh.")
        ("norm-multi", value<int>()->notifier([&](int i) {config.setNormalizationMultiplier(i); }), "Multiply the descriptor by pow(2,<int>).")
        ( "norm-mode", value<std::string>()->notifier([&](const std::string& s) { config.setNormMode(s); }),
          popsift::Config::getNormModeUsage() )
        ( "root-sift", bool_switch()->notifier([&](bool b) { if(b) config.setNormMode(popsift::Config::RootSift); }),
          popsift::Config::getNormModeUsage() )
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

#ifdef USE_OIIO
    if( ! pgmread_loading )
    {
        auto inp = ImageInput::open( inputFile.c_str() );
        if( !inp )
        {
            cerr << "Could not load image " << inputFile << endl;
            return 0;
        }

        const ImageSpec& spec = inp->spec();
        int w = spec.width;
        int h = spec.height;
        int nchannels = spec.nchannels;

        if( nchannels == 3 )
        {
            cout << "Loading " << w << " x " << h << " image " << inputFile << endl;

            uint8_t* load_data  = new unsigned char[w * h * nchannels];
            image_data = new unsigned char[w * h];

            inp->read_image( 0, 0, 0, nchannels, TypeDesc::UINT8, load_data );
            inp->close();

            for( int i=0; i<w*h; i++ )
            {
                image_data[i] = (uint8_t)( 0.299f * load_data[3*i]
                                         + 0.587f * load_data[3*i+1]
                                         + 0.114f * load_data[3*i+2] );
            }

            job = PopSift.enqueue( w, h, image_data );
        }
        else if( nchannels == 1 )
        {
            cout << "Loading " << w << " x " << h << " image " << inputFile << endl;

            image_data = new unsigned char[w * h];

            inp->read_image( 0, 0, 0, 1, TypeDesc::UINT8, image_data );
            inp->close();

            job = PopSift.enqueue( w, h, image_data );
        }
        else
        {
            cerr << "Cannot read images that don't have 1 or 3 channels, " << inputFile << endl;
            return 0;
        }
    }
    else
#endif
    {
        int w{};
        int h{};
        image_data = readPGMfile( inputFile, w, h );
        if( image_data == nullptr ) {
            exit( EXIT_FAILURE );
        }

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
        std::ofstream of( "output-features.txt" );
        feature_list->print( of, write_as_uchar );
    }
    delete feature_list;
}

int main(int argc, char **argv)
{
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

