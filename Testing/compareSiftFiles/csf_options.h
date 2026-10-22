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
    float                    max_avg_desc_dist; // CTest pass/fail gate; <0 disables the check
    bool                     allow_empty;       // an input without features is not an error
    std::string              match_dir;         // write matched.csv, only_a.csv, only_b.csv here
    float                    match_radius_sigma;
    float                    match_radius_min;
    float                    match_max_scale_ratio;
};

void parseargs( int argc, char** argv, Parameters& param );
void usage( char* name, const boost::program_options::options_description& all );

