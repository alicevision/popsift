#include <iostream>

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

        param.input = vm["input"].as<vector<string> >();

        if( vm.count("help") )
        {
            usage( argv[0], options );
            exit(EXIT_SUCCESS);
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

