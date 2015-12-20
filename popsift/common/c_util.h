#pragma once

#include <string>

/* Error handler */
#define ERROR_HANDLER(x, y) check_error((x), (y), __FILE__, __LINE__)

/** 
 * @brief Force exit if condition is 0
 *        MACRO -> ERROR_HANDLER(x, y)
 * 
 * @param condition      Condition to be checked
 * @param errorString    Collection of error strings
 * @param file           File to be checked
 * @param line           Line number
 */

void check_error( bool condition,
                  const std::string& errorString,
                  const char *file,
                  const int line );

/** 
 * @brief Extract filename and prefix
 *        Overloads are defined below
 * 
 * @param filename     The filename
 * @param prefix       Prefix (if there is)
 * 
 * @return std::string Filename with prefix extracted
 */
std::string extract_filename(const std::string& filename, std::string& prefix);

