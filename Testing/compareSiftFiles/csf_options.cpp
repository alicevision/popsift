#include <iostream>
#include <cstdlib>

#include <boost/program_options.hpp>

#include "csf_options.h"

using namespace std;

void parseargs( int argc, char** argv, Parameters& param )
{
    using namespace boost::program_options;

    bool longinfo = false;

    positional_options_description positional;
    positional.add( "input", -1 );

    options_description hidden_options("Hidden options");
    options_description options("Options");

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
            ("l1",        bool_switch()->notifier( [&](bool b)
                                                   {
                                                       feat_t::setL2Distance( !b );
                                                   } )
                                       ->default_value( false ),
                          "use L1 for distance computation instead of L2" )
            ("max-avg-dist", value<float>(&param.max_avg_desc_dist)->default_value(-1.0f),
                          "If >= 0, fail (nonzero exit code) when the average best-match\n"
                          "descriptor distance across all input features exceeds this value.\n"
                          "Left unset (-1), the tool always exits successfully, which is\n"
                          "useful for a first run to find out what a sane value is." )
            ("allow-empty", bool_switch(&param.allow_empty)->default_value(false),
                          "Accept two inputs that both contain no features. Without\n"
                          "this, that is an error (nonzero exit code). One empty\n"
                          "input is always an error." )
            ("match-dir", value<std::string>(&param.match_dir),
                          "Pair the features of the two files by position, scale and\n"
                          "orientation, and write matched.csv, only_a.csv and\n"
                          "only_b.csv into this directory (a = first file)." )
            ("match-radius-sigma", value<float>(&param.match_radius_sigma)->default_value(0.5f),
                          "Matching: maximum position distance as a multiple of the\n"
                          "first feature's sigma ..." )
            ("match-radius-min", value<float>(&param.match_radius_min)->default_value(1.0f),
                          "... but at least this many pixels." )
            ("match-max-scale-ratio", value<float>(&param.match_max_scale_ratio)->default_value(1.5f),
                          "Matching: maximum ratio between the two sigmas." )
            ;
    }

    options_description all("Allowed options");

    all.add(hidden_options)
       .add(options);

    variables_map vm;

    try
    {

        store( command_line_parser(argc, argv).options( all )
                                              .positional( positional )
                                              .run( ),
                vm );

        if( vm.count("help") )
        {
            usage( argv[0], options );
            exit(EXIT_SUCCESS);
        }

        if( vm.count("input") )
        {
            param.input = vm["input"].as<vector<string> >();
        }

        if( param.input.size() != 2 )
        {
            cerr << "Error: Exactly 2 input file for comparison must be provided." << endl;
            usage( argv[0], options );
            exit(EXIT_FAILURE);
        }

        notify(vm); // Notify does processing (e.g., raise exceptions if required args are missing)
    }
    catch(boost::program_options::error& e)
    {
        std::cerr << "Error: " << e.what() << std::endl << std::endl;
        usage( argv[0], options );
        exit(EXIT_FAILURE);
    }

    param.briefinfo = !longinfo;
}

void usage( char* name, const boost::program_options::options_description& options )
{
    cerr << endl
         << "Usage: " << name << " [Options] <descriptorfile> <descriptorfile>" << endl
         << endl
         << "Description\n"
            "    Compute the L1 or L2 distance between the descriptors of closest\n"
            "    coordinate pairs.\n"
            "    When a coordinate has several orientations, the closest distance\n"
            "    is reported. Summary information at the end." << endl
         << endl
         << options << endl;
}

