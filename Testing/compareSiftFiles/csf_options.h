#pragma once

#include <string>
#include <vector>
#include <filesystem>

#include <boost/program_options.hpp>

#include "csf_feat.h"

/** The command line options. An empty file or directory name means that
 * output is not requested.
 */
struct Parameters
{
    std::vector<std::string> input;             ///< the two descriptor files
    std::filesystem::path    outfile_name;      ///< -o: per-feature report; empty: cout
    std::filesystem::path    descfile_name;     ///< -d: descriptor difference summary
    bool                     descfile_verbose;  ///< --dv: also every difference into descfile_name
    bool                     briefinfo;         ///< only the best match per feature (not -v)
    float                    max_avg_desc_dist; ///< CTest pass/fail gate; < 0 disables the check
    bool                     allow_empty;       ///< two inputs without features are not an error
    std::filesystem::path    pair_directory;    ///< --pair-dir: directory for the CSV files
    float                    pair_radius_sigma;    ///< PairParams::radius_sigma
    float                    pair_radius_min;      ///< PairParams::radius_min
    float                    pair_max_scale_ratio; ///< PairParams::max_scale_ratio
    float                    ori_tolerance_deg;    ///< PairParams::ori_tolerance_deg
    float                    desc_tolerance;       ///< PairParams::desc_tolerance
};

/** Parses the command line into param and calls feat_t::setUseL2Distance().
 * Numeric values are not range-checked.
 * Terminates the process: with EXIT_SUCCESS after printing usage for --help;
 * with EXIT_FAILURE after printing an error and usage to cerr if the options
 * are invalid or not exactly two input files are given.
 * @param[in]  argc argument count
 * @param[in]  argv arguments
 * @param[out] param on return, param.input.size() == 2
 */
void parseargs( int argc, char** argv, Parameters& param );

/** Prints the usage text and the option descriptions to cerr.
 * @param[in] name program name
 * @param[in] all  options to describe
 */
void usage( char* name, const boost::program_options::options_description& all );
