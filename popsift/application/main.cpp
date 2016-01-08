/**
 * Copyright (c) 2012, Fixstars Corp.
 * All rights reserved.
 * 
 * Full copyright see end of file.
 */

/**
 * @file   main.cpp
 * @author Yuri Ardila <y_ardila@fixstars.com>
 * @date   Tue Oct 30 16:01:19 JST 2012
 * 
 * @brief  
 *    SIFT
 *    OpenCL Implementation
 *
 */

#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <iomanip>
#include "getopt.h"

#include "SIFT.h"
#include "device_prop.h"

#define N_ITER 1

using namespace std;

/* User parameters */
int    verbose         = false;

string keyFilename     = "";
string inputFilename   = "";
string realName        = ""; 
string prefix          = "";

int    upsampling      = 1;
int    octaves         = -1;
int    levels          = 3;
float  sigma           = 1.6f;

float edgeLimit = 16.0; // from Celebrandil
// float edgeLimit = 10.0; // from Bemap

// float threshold = 15.0 / 256.0;  // it seems our DoG is really small ???
float threshold = 10.0 / 256.0;
// float threshold = 5.0;  // from Celebrandil, not happening in our data
// float threshold = 0.04 / (_levels-3.0) / 2.0f * 255;
//                   from Bemap -> 1.69 (makes no sense)


int    saveGauss       = false;
int    saveDOG         = false;
int    saveMag         = false;
int    saveOri         = false;
int    display         = false;
int    log_to_file     = false;

static struct option longopts[] = {
    { "verbose",         no_argument,            NULL,              'v' },
    { "help",            no_argument,            NULL,              'h' },
    { "octaves",         required_argument,      NULL,              'O' },
    { "levels",          required_argument,      NULL,              'S' },
    { "upsampling",      required_argument,      NULL,              'u' },
    { "threshold",       required_argument,      NULL,              't' },
    { "edge-threshold",  required_argument,      NULL,              'e' },
    { "log",             no_argument,            &log_to_file,      true},
    { "save-gauss",      no_argument,            &saveGauss,        true},
    { "save-dog",        no_argument,            &saveDOG,          true},
    { "save-mag",        no_argument,            &saveMag,          true},
    { "save-ori",        no_argument,            &saveOri,          true},

    { NULL,              0,                      NULL,               0  }
};

/***************************/
/* @name help              */
/* @brief Print help       */
/* @param filename argv[0] */
/***************************/
void help(const string& filename)
{
    cout
        << filename
        // << " [--verbose|-v] [--help|-h]" << endl
        << " [--help|-h]" << endl
        // << "     [--octaves|-O INT] [--levels|-S INT] [--upsampling|-u INT]" << endl
        // << "     [--threshold|-t FLOAT] [--edge-threshold|-e FLOAT] [--floating-point]" << endl
        << "     [--threshold|-t FLOAT] [--log]" << endl
        // << "     [--save-gauss] [--save-dog] [--save-mag] [--save-ori] [--fast-comp]" << endl
        << "     FILENAME"
        << endl << endl
        << "* Options *" << endl
        // << " --verbose                  Be verbose"<< endl
        << " --help                     Print this message"<<endl
        // << " --octaves=INT              Number of octaves" << endl
        // << " --levels=INT               Number of levels per octave" << endl
        // << " --upsampling=INT           Number of upsamplings" << endl
        << " --threshold=FLOAT          Keypoint strength threshold" << endl
        << " --log                      Write debugging files" << endl
        // << " --edge-threshold=FLOAT     On-edge threshold" << endl
        // << " --save-gauss               Save Gaussian Scale pyramid"<<endl
        // << " --save-dog                 Save Difference of Gaussian pyramid"<<endl
        // << " --save-mag                 Save Magnitudes pyramid"<<endl
        // << " --save-ori                 Save Orientations pyramid"<<endl
        // << endl
        // << " The keypoints will be written to [filename].key" << endl
        // << endl
        // << " * Examples *" << endl
        // << filename << " [OPTS...] -v -u 2 --save-gauss --use-gpu test_data.pgm" << endl
        // << filename << " [OPTS...] -O 5 -S 2 --outkey=test_output.key --dev-info test_data.ppm" << endl
        << endl;
    exit(0);
}

/**********************************/
/* @name option                   */
/* @brief Process user parameters */
/* @param ac argc                 */
/*        av argv                 */
/**********************************/
void option(int ac, char **av)
{
    if (ac == 1) std::cout << av[0] << ": Execute with default parameter(s)..\n(--help for program usage)\n\n";
    int opt;
    while ((opt = getopt_long(ac, av, "vho:S:u:t:e:gdp:", longopts, NULL)) != -1) {
        switch (opt) {

        case '?' :
            ERROR_HANDLER(0, "Invalid option '" + std::string(av[optind-1]) + "'" );
            break;

        case ':' :
            ERROR_HANDLER(0, "Missing argument of option '" + std::string(av[optind-1]) + "'");
            break;

        case 'v' :
            verbose = true;
            break;

        case 'h' :
            help(av[0]);
            break;

        case 'O':
			{
                /* number of octaves */
				std::istringstream iss(optarg);
				int a = -1;
				iss >> a; octaves = a;
				ERROR_HANDLER((!iss.fail()), "Invalid argument '" + std::string(optarg) + "'");
				ERROR_HANDLER((octaves >= 0), "Octaves must be bigger than 0");
			}
			break;

        case 'S':
			{
			    /* number of levels */
				std::istringstream iss(optarg);
				int a = -1;
				iss >> a; levels = a; 
				ERROR_HANDLER((!iss.fail()), "Invalid argument '" + std::string(optarg) + "'");
				ERROR_HANDLER((levels >= 0), "Levels must be bigger than 0");
			}
			break;

        case 'u':
			{
			    /* number of upsamplings */
				std::istringstream iss(optarg);
				int a = -1;
				iss >> a; upsampling = a;
				ERROR_HANDLER((!iss.fail()), "Invalid argument '" + std::string(optarg) + "'");
				ERROR_HANDLER((upsampling >= 0), "Upsampling must be bigger than 0");
			}
			break;

        case 't':
			{
			    /* threshold */
				std::istringstream iss(optarg);
				float a = -1;
				iss >> a; threshold = a;
				ERROR_HANDLER((!iss.fail()), "Invalid argument '" + std::string(optarg) + "'");
			}
			break;

        case 'e':
			{
			    /* edge-threshold */
				std::istringstream iss(optarg);
				float a = -1;
				iss >> a; edgeLimit = a;
				ERROR_HANDLER((!iss.fail()), "Invalid argument '" + std::string(optarg) + "'");
			}
			break;

        case 0:
            break;

        default :
            ERROR_HANDLER(0, "Error parsing arguments");    
        }
    }

    ac -= optind;
    av += optind;

    if (ac == 0) {
        cerr << "An input file is needed" << endl;
        exit( -1 );
    }

    inputFilename = std::string(av[0]); ac--; av++;
    ERROR_HANDLER((ac==0), "Too many input files");
}

/*********************/
/* @name main        */
/* @brief main       */
/* @param argc, argv */
/*********************/
int main(int argc, char **argv)
{
    cudaDeviceReset();

    imgStream inp;

    /* Parse user input */
    option(argc, argv);
    realName = extract_filename(inputFilename, prefix);
    read_gray(inputFilename, inp);
    cerr << "Real name of input file is " << realName << endl;

    device_prop_t deviceInfo;
    deviceInfo.print( );

    PopSift PopSift( octaves,
                     levels,
                     upsampling,
                     threshold,
                     edgeLimit,
                     sigma );

    PopSift.init( inp.width, inp.height );
    PopSift.execute(inp);
    PopSift.uninit( );
    return 0;
}

/**
 * Copyright (c) 2012, Fixstars Corp.
 * All rights reserved.
 * 
 * The following patent has been issued for methods embodied in this 
 * software: "Method and apparatus for identifying scale invariant features 
 * in an image and use of same for locating an object in an image," David 
 * G. Lowe, US Patent 6,711,293 (March 23, 2004). Provisional application 
 * filed March 8, 1999. Asignee: The University of British Columbia. For 
 * further details, contact David Lowe (lowe@cs.ubc.ca) or the 
 * University-Industry Liaison Office of the University of British 
 * Columbia.
 * 
 * Note that restrictions imposed by this patent (and possibly others) 
 * exist independently of and may be in conflict with the freedoms granted 
 * in this license, which refers to copyright of the program, not patents 
 * for any methods that it implements.  Both copyright and patent law must 
 * be obeyed to legally use and redistribute this program and it is not the 
 * purpose of this license to induce you to infringe any patents or other 
 * property right claims or to contest validity of any such claims.  If you 
 * redistribute or use the program, then this license merely protects you 
 * from committing copyright infringement.  It does not protect you from 
 * committing patent infringement.  So, before you do anything with this 
 * program, make sure that you have permission to do so not merely in terms 
 * of copyright, but also in terms of patent law.
 * 
 * Please note that this license is not to be understood as a guarantee 
 * either.  If you use the program according to this license, but in 
 * conflict with patent law, it does not mean that the licensor will refund 
 * you for any losses that you incur if you are sued for your patent 
 * infringement.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are 
 * met:
 *     * Redistributions of source code must retain the above copyright and 
 *       patent notices, this list of conditions and the following 
 *       disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in 
 *       the documentation and/or other materials provided with the 
 *       distribution.
 *     * Neither the name of Fixstars Corp. nor the names of its 
 *       contributors may be used to endorse or promote products derived 
 *       from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS 
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED 
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
 * HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 */

