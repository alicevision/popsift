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
// #include "libavutil/log.h"
// #include "libavutil/pixfmt.h"

#include "SIFT.h"
#include "sift_conf.h"
#include "device_prop.h"

using namespace std;

/* User parameters */
int    verbose         = false;

string keyFilename     = "";
string inputFilename   = "";
string realName        = ""; 
string prefix          = "";

// int    upsampling      = 1;
// int    octaves         = -1;
// int    levels          = 3;
// float  sigma           = 1.6f;

// float edgeLimit = 16.0; // from Celebrandil
// float edgeLimit = 10.0; // from Bemap

// float threshold = 0.0; // default of vlFeat
// float threshold =  5.0 / 256.0;
// float threshold = 15.0 / 256.0;  // it seems our DoG is really small ???
// float threshold = 10.0 / 256.0;
// float threshold = 5.0;  // from Celebrandil, not happening in our data
// float threshold = 0.04 / (_levels-3.0) / 2.0f * 255;
//                   from Bemap -> 1.69 (makes no sense)

// int    display         = false;
// int    log_to_file     = false;
// int    vlfeat_mode     = false;

static void usage( const char* argv )
{
    cout << argv
         << "     <filename>"
         << endl << endl
         << "* Options *" << endl
         << " --help / -h / -?            Print usage" << endl
         << " --verbose / -v" << endl
         << endl
         << " --vlfeat-mode               Compute Gauss filter like VLFeat instead of like OpenCV" << endl
         << " --log                       Write debugging files" << endl
         << endl
         << " --octaves=<int>             Number of octaves" << endl
         << " --levels=<int>              Number of levels per octave" << endl
         << " --sigma=<float>             Initial sigma value" << endl
         << " --threshold=<float>         Keypoint strength threshold" << endl
         << " --edge-threshold=<float> or" << endl
         << " --edge-limit=<float>        On-edge threshold" << endl
         << endl;
    exit(0);
}

static struct option longopts[] = {
    { "help",            no_argument,            NULL, 'h' },
    { "verbose",         no_argument,            NULL, 'v' },

    { "octaves",         required_argument,      NULL, 1000 },
    { "levels",          required_argument,      NULL, 1001 },
    { "upsampling",      required_argument,      NULL, 1002 },
    { "threshold",       required_argument,      NULL, 1003 },
    { "edge-threshold",  required_argument,      NULL, 1004 },
    { "edge-limit",      required_argument,      NULL, 1004 },
    { "sigma",           required_argument,      NULL, 1005 },

    { "vlfeat-mode",     no_argument,            NULL, 1100 },
    { "log",             no_argument,            NULL, 1101 },
    { NULL,              0,                      NULL, 0  }
};

static void parseargs( int argc, char**argv, popart::Config& config, string& inputFile )
{
    const char* appName = argv[0];
    if( argc == 0 ) usage( "<program>" );
    if( argc == 1 ) usage( argv[0] );

    int opt;

    while( (opt = getopt_long(argc, argv, "?hv", longopts, NULL)) != -1 )
    {
        switch (opt)
        {
        case '?' :
        case 'h' : usage( appName );       break;
        case 'v' : config.setVerbose(); break;

        case 1100 : config.setModeVLFeat( popart::Config::VLFeat ); break;
        case 1101 : config.setLogMode( popart::Config::All );       break;

        case 1000 : config.setOctaves( strtol( optarg, NULL, 0 ) ); break;
        case 1001 : config.setLevels(  strtol( optarg, NULL, 0 ) ); break;
        case 1002 : config.setUpsampling( strtof( optarg, NULL ) ); break;
        case 1003 : config.setThreshold(  strtof( optarg, NULL ) ); break;
        case 1004 : config.setEdgeLimit(  strtof( optarg, NULL ) ); break;
        case 1005 : config.setSigma(      strtof( optarg, NULL ) ); break;
        default   : usage( appName );
        }
    }

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
    cerr << "Real name of input file is " << realName << endl
         << "Width: " << inp.width
         << " height: " << inp.height
         << endl;

    device_prop_t deviceInfo;
    // deviceInfo.set( 1 );
    // deviceInfo.print( );

    PopSift PopSift( config );

    PopSift.init( inp.width, inp.height );
    cerr << "Width: " << inp.width << " height: " << inp.height << endl;
    PopSift.execute( inp );
    PopSift.uninit( );
    return 0;
}

