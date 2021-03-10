#pragma once

#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include "csf_feat.h"

struct Parameters
{
    std::vector<std::string> input;
    std::string              outfile_name;
    std::string              descfile_name;
    bool                     descfile_verbose;
    bool                     briefinfo;
};

void parseargs( int argc, char** argv, Parameters& param );
void usage( char* name, const boost::program_options::options_description& all );

