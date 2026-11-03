#include <iostream>
#include <cstdlib>

#include <boost/program_options.hpp>

#include "csf_options.h"

using namespace std;

void parseargs( int argc, char** argv, Parameters& param )
{
    using namespace boost::program_options;

    bool longinfo = false;
    bool use_l1_distance = false;

    positional_options_description positional;
    positional.add( "input", -1 );

    options_description hidden_options("Hidden options");
    options_description options("Options");
    options_description same_image("Only for same-image comparisons, where both "
                                   "files describe the same image");

    {
        hidden_options.add_options()
            ("input,i",   value<vector<string> >()->multitoken()->composing(),
                          "A descriptor file. 2 descriptor files must be passed");

        options.add_options()
            ("help,h", "Print usage")
            ("verbose,v", value<bool>( &longinfo )->default_value(false),
                          "verbose, longer output")
            ("output,o",  value<std::string>(&param.outfile_name),
                          "print essential diff info to <file> (default is cout)")
            ("desc,d",    value<std::string>(&param.descfile_name),
                          "print descriptor distance per cell to <file>")
            ("dv",        value<bool>(&param.descfile_verbose)->default_value(false),
                          "prints every comparison into descfile, instead of summary only")
            ("l1",        bool_switch(&use_l1_distance)->default_value(false),
                          "use L1 for distance computation instead of L2" )
            /* max-avg-dist should be renamed max-avg-desc-dist */
            ("max-avg-dist", value<float>(&param.max_avg_desc_dist)->default_value(-1.0f),
                          "If >= 0, fail (nonzero exit code) when the average best-match "
                          "descriptor distance across all input features exceeds this value. "
                          "Left unset (-1), the tool always exits successfully." )
            ("allow-empty", bool_switch(&param.allow_empty)->default_value(false),
                          "Accept two inputs that both contain no features. Without "
                          "this, that is an error (nonzero exit code). One empty "
                          "input is always an error." )
            ;

        same_image.add_options()
            ("match-dir", value<std::string>(&param.match_directory),
                          "Pair the features of the two files by position, scale and "
                          "orientation, and write matched.csv, only_a.csv, "
                          "only_b.csv and desc_bins.csv into this directory "
                          "(a = first file)." )
            ("match-radius-sigma", value<float>(&param.match_radius_sigma)->default_value(0.5f),
                          "Matching: maximum position distance as a multiple of the "
                          "first feature's sigma ..." )
            ("match-radius-min", value<float>(&param.match_radius_min)->default_value(1.0f),
                          "... but at least this many pixels." )
            ("match-max-scale-ratio", value<float>(&param.match_max_scale_ratio)->default_value(1.5f),
                          "Matching: maximum ratio between the two sigmas." )
            /* same-pose-px should be renamed fail-pix-dist */
            ("same-pose-px", value<float>(&param.same_pose_px)->default_value(0.25f),
                          "desc_bins.csv: matched pairs at most this many pixels ..." )
            /* same-pose-deg shold be renamed fail-deg-dist */
            ("same-pose-deg", value<float>(&param.same_pose_deg)->default_value(2.0f),
                          "... and degrees apart have the same pose." )
            ;
    }

    options_description visible;
    visible.add(options)
           .add(same_image);

    options_description all("Allowed options");

    all.add(hidden_options)
       .add(visible);

    variables_map vm;

    try
    {

        store( command_line_parser(argc, argv).options( all )
                                              .positional( positional )
                                              .run( ),
                vm );

        if( vm.count("help") )
        {
            usage( argv[0], visible );
            exit(EXIT_SUCCESS);
        }

        if( vm.count("input") )
        {
            param.input = vm["input"].as<vector<string> >();
        }

        if( param.input.size() != 2 )
        {
            cerr << "Error: Exactly 2 input file for comparison must be provided." << endl;
            usage( argv[0], visible );
            exit(EXIT_FAILURE);
        }

        notify(vm); // Notify does processing (e.g., raise exceptions if required args are missing)
    }
    catch(boost::program_options::error& e)
    {
        std::cerr << "Error: " << e.what() << std::endl << std::endl;
        usage( argv[0], visible );
        exit(EXIT_FAILURE);
    }

    param.briefinfo = !longinfo;

    feat_t::setUseL2Distance( !use_l1_distance );
}

void usage( char* name, const boost::program_options::options_description& options )
{
    cerr << endl
         << "Usage: " << name << " [Options] <descriptorfile> <descriptorfile>" << endl
         << endl
         << "Description\n"
            "    For every feature of the first file, find the feature of the\n"
            "    second file with the closest descriptor (L2, or L1 with --l1),\n"
            "    and report the descriptor distance, the pixel distance, and the\n"
            "    scale and angle difference. The average descriptor distance is\n"
            "    printed at the end." << endl
         << endl
         << options << endl;
}

