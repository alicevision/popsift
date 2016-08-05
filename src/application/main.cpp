#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <iomanip>
#include <stdlib.h>

#include <getopt.h>
#include <boost/filesystem.hpp>

#include "libavformat/avformat.h"
#include "libavutil/avutil.h"

#include "popsift.h"
#include "sift_conf.h"
#include "device_prop.h"
#include "c_util_img.h"

using namespace std;

static void validate( const char* appName, popart::Config& config );

/* User parameters */
// int    verbose         = false;
static bool print_info      = false;

string keyFilename     = "";
string inputFilename   = "";
string realName        = ""; 
string prefix          = "";

static void usage( const char* argv )
{
    cout << argv
         << "     <filename>"
         << endl << endl
         << "* Options *" << endl
         << " --help / -h / -?            Print usage" << endl
         << " --verbose / -v" << endl
         << " --log / -l                  Write debugging files" << endl
         << " --print-info / -p           Print info about GPUs" << endl
         << endl
         << "* Parameters *" << endl
         << " --octaves=<int>             Number of octaves" << endl
         << " --levels=<int>              Number of levels per octave" << endl
         << " --sigma=<float>             Initial sigma value" << endl
         << " --threshold=<float>         Keypoint strength threshold" << endl
         << " --edge-threshold=<float> or" << endl
         << " --edge-limit=<float>        On-edge threshold" << endl
         << " --downsampling=<float>      Downscale width and height of input by 2^N (default N=-1)" << endl
         << " --initial-blur=<float>      Assume initial blur, subtract when blurring first time" << endl
         << endl
         << "* Modes *" << endl
         << " --vlfeat-mode               During the initial upscale, shift pixels by 1." << endl
         << "                             That creates a sharper upscaled image. " << endl
         << "                             In extrema refinement, steps up to 0.6, levels remain unchanged," << endl
         << "                             do not reject points when reaching max iterations," << endl
         << "                             first contrast threshold is .8 * peak thresh." << endl
         << " --opencv-mode               During the initial upscale, shift pixels by 0.5." << endl
         << "                             In extrema refinement, steps up to 0.5," << endl
         << "                             reject points when reaching max iterations," << endl
         << "                             first contrast threshold is floor(.5 * peak thresh)." << endl
         << "                             Computed filter width are lower than VLFeat/PopSift" << endl
         << " --popsift-mode (default)    During the initial upscale, shift pixels by 1." << endl
         << "                             In extrema refinement, steps up to 0.6," << endl
         << "                             do not reject points when reaching max iterations," << endl
         << "                             first contrast threshold is .8 * peak thresh." << endl
         << "                             Shift feature coords octave 0 back to original pos." << endl
         << " --test-direct-scaling       Direct each octave from upscaled orig instead of blurred level" << endl
         << "                             Does not work yet" << endl
         << " --group-gauss=<int>         Gauss-filter N levels at once (N=2, 3 or 8)" << endl
         << "                             3 is accurate for default sigmas of VLFeat and OpenCV mode" << endl
         << " --dp-off                    Switch all CUDA Dynamic Parallelism off" << endl
         << " --dp-ori-off                Switch DP off for orientation computation" << endl
         << " --dp-desc-off               Switch DP off for descriptor computation" << endl
         << endl
         << "* Informational *" << endl
         << " --print-gauss-tables        A debug output printing Gauss filter size and tables" << endl
         << endl;
    exit(0);
}

static struct option longopts[] = {
    { "help",                no_argument,            NULL, 'h' },
    { "verbose",             no_argument,            NULL, 'v' },
    { "log",                 no_argument,            NULL, 'l' },
    { "print-info",          no_argument,            NULL, 'p' },

    { "octaves",             required_argument,      NULL, 1000 },
    { "levels",              required_argument,      NULL, 1001 },
    { "downsampling",        required_argument,      NULL, 1002 },
    { "threshold",           required_argument,      NULL, 1003 },
    { "edge-threshold",      required_argument,      NULL, 1004 },
    { "edge-limit",          required_argument,      NULL, 1004 },
    { "sigma",               required_argument,      NULL, 1005 },
    { "initial-blur",        required_argument,      NULL, 1006 },

    { "vlfeat-mode",         no_argument,            NULL, 1100 },
    { "opencv-mode",         no_argument,            NULL, 1101 },
    { "popsift-mode",        no_argument,            NULL, 1102 },
    { "test-direct-scaling", no_argument,            NULL, 1103 },
    { "group-gauss",         required_argument,      NULL, 1104 },
    { "dp-off",              no_argument,            NULL, 1105 },
    { "dp-ori-off",          no_argument,            NULL, 1106 },
    { "dp-desc-off",         no_argument,            NULL, 1107 },

    { "print-gauss-tables",  no_argument,            NULL, 1200 },

    { NULL,                  0,                      NULL, 0  }
};

static void parseargs( int argc, char**argv, popart::Config& config, string& inputFile )
{
    const char* appName = argv[0];
    if( argc == 0 ) usage( "<program>" );
    if( argc == 1 ) usage( argv[0] );

    int opt;
    bool applySigma = false;
    float sigma;

    while( (opt = getopt_long(argc, argv, "?hvlp", longopts, NULL)) != -1 )
    {
        switch (opt)
        {
        case '?' :
        case 'h' : usage( appName ); break;
        case 'v' : config.setVerbose(); break;
        case 'l' : config.setLogMode( popart::Config::All ); break;
        case 'p' : print_info = true; break;

        case 1000 : config.setOctaves( strtol( optarg, NULL, 0 ) ); break;
        case 1001 : config.setLevels(  strtol( optarg, NULL, 0 ) ); break;
        case 1002 : config.setDownsampling( strtof( optarg, NULL ) ); break;
        case 1003 : config.setThreshold(  strtof( optarg, NULL ) ); break;
        case 1004 : config.setEdgeLimit(  strtof( optarg, NULL ) ); break;
        case 1005 : applySigma = true; sigma = strtof( optarg, NULL ); break;
        case 1006 : config.setInitialBlur( strtof( optarg, NULL ) ); break;

        case 1100 : config.setMode( popart::Config::VLFeat ); break;
        case 1101 : config.setMode( popart::Config::OpenCV ); break;
        case 1102 : config.setMode( popart::Config::PopSift ); break;
        case 1103 : config.setScalingMode( popart::Config::ScaleDirect ); break;
        case 1104 : config.setGaussGroup( strtol( optarg, NULL, 0 ) ); break;
        case 1105 : config.setDPOrientation( false ); config.setDPDescriptors( false ); break;
        case 1106 : config.setDPOrientation( false ); break;
        case 1107 : config.setDPDescriptors( false ); break;

        case 1200 : config.setPrintGaussTables( ); break;
        default   : usage( appName );
        }
    }

    if( applySigma ) config.setSigma( sigma );

    validate( appName, config );

    argc -= optind;
    argv += optind;

    if( argc == 0 ) usage( appName );

    inputFile = argv[0];
}

int main(int argc, char **argv)
{
    cudaDeviceReset();

    popart::Config config;
    string         inputFile = "";
    const char*    appName   = argv[0];

    parseargs( argc, argv, config, inputFile ); // Parse command line

    if( inputFile == "" ) {
        cerr << "No input filename given" << endl;
        usage( appName );
    }

    if( not boost::filesystem::exists( inputFile ) ) {
        cerr << "File " << inputFile << " not found" << endl;
        usage( appName );
    }

    imgStream inp;

    realName = extract_filename( inputFile, prefix );
    read_gray( inputFile, inp );
    cerr << "Input image size: "
         << inp.width << "X" << inp.height
         << " filename: " << realName << endl;

    device_prop_t deviceInfo;
    deviceInfo.set( 0 );
    if( print_info ) {
        deviceInfo.print( );
    }

    PopSift PopSift( config );

    PopSift.init( 0, inp.width, inp.height );
    PopSift.execute( 0, &inp );
    PopSift.uninit( 0 );
    return 0;
}

static void validate( const char* appName, popart::Config& config )
{
    switch( config.getGaussGroup() )
    {
    case 1 :
    case 2 :
    case 3 :
    case 8 :
        break;
    default :
        cerr << "Only 2, 3 or 8 Gauss levels can be combined at this time" << endl;
        usage( appName );
        exit( -1 );
    }
}

