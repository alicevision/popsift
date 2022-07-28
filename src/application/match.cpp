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
#include "common_opt.h"

#if POPSIFT_IS_DEFINED(POPSIFT_USE_NVTX)
#include <nvToolsExtCuda.h>
#else
#define nvtxRangePushA(a)
#define nvtxRangePop()
#endif

using namespace std;

static bool print_dev_info  {false};
static bool print_time_info {false};
static bool write_as_uchar  {false};
static bool dont_write      {false};
static bool pgmread_loading {false};

static void parseargs(int argc, char** argv, popsift::Config& config, string& lFile, string& rFile)
{
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
    options_description modes("Modes");

    option_init_parameters( config, parameters );
    option_init_modes( config, modes );

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
    unsigned char* image_data;
    SiftJob* job;

    nvtxRangePushA( "load and convert image" );
#ifdef USE_DEVIL
    if( ! pgmread_loading )
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
        const auto w = img.Width();
        const auto h = img.Height();
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
        int h{};
        int w{};
        image_data = readPGMfile( inputFile, w, h );
        if( image_data == nullptr ) {
            exit( EXIT_FAILURE );
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
    string         lFile{};
    string         rFile{};

    std::cout << "PopSift version: " << POPSIFT_VERSION_STRING << std::endl;

    config.setDescMode( popsift::Config::VLFeat_Desc );

    try {
        parseargs( argc, argv, config, lFile, rFile ); // Parse command line
        std::cout << lFile << " <-> " << rFile << std::endl;
    }
    catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        return EXIT_SUCCESS;
    }

    if( boost::filesystem::exists( lFile ) ) {
        if( ! boost::filesystem::is_regular_file( lFile ) ) {
            cout << "Input file " << lFile << " is not a regular file, nothing to do" << endl;
            return EXIT_FAILURE;
        }
    }

    if( boost::filesystem::exists( rFile ) ) {
        if( ! boost::filesystem::is_regular_file( rFile ) ) {
            cout << "Input file " << rFile << " is not a regular file, nothing to do" << endl;
            return EXIT_FAILURE;
        }
    }

    popsift::cuda::device_prop_t deviceInfo;
    deviceInfo.set( 0, print_dev_info );
    if( print_dev_info ) deviceInfo.print( );

    PopSift PopSift( config, popsift::Config::MatchingMode );

    SiftJob* lJob = process_image( lFile, PopSift );
    SiftJob* rJob = process_image( rFile, PopSift );

    popsift::FeaturesDev* lFeatures = lJob->getDev();
    cout << "Number of features:    " << lFeatures->getFeatureCount() << endl;
    cout << "Number of descriptors: " << lFeatures->getDescriptorCount() << endl;

    popsift::FeaturesDev* rFeatures = rJob->getDev();
    cout << "Number of features:    " << rFeatures->getFeatureCount() << endl;
    cout << "Number of descriptors: " << rFeatures->getDescriptorCount() << endl;

    int3* matches = lFeatures->matchAndReturn( rFeatures );
    cudaDeviceSynchronize();

    for( int i=0; i<lFeatures->getDescriptorCount(); i++ )
    {
        int3& match = matches[i];
        if( match.z )
        {
            const popsift::Feature* l_f = lFeatures->getFeatureForDescriptor( i );
            const popsift::Feature* r_f = rFeatures->getFeatureForDescriptor( match.x );
            cout << setprecision(5) << showpoint
                 << "point (" << l_f->xpos << "," << l_f->ypos << ") in l matches "
                 << "point (" << r_f->xpos << "," << r_f->ypos << ") in r" << endl;
        }
    }

    cudaFree( matches );

    delete lFeatures;
    delete rFeatures;

    PopSift.uninit( );

    return EXIT_SUCCESS;
}

