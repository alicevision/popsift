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
    std::string              pair_directory;    // write colocated.csv, only_a.csv, only_b.csv, ... here
    float                    pair_radius_sigma;
    float                    pair_radius_min;
    float                    pair_max_scale_ratio;
    float                    ori_tolerance_deg;
    float                    desc_tolerance;
};

void parseargs( int argc, char** argv, Parameters& param );
void usage( char* name, const boost::program_options::options_description& all );

