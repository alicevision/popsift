#include "c_util.h"

#include <iostream>
#include <stdlib.h>

void check_error( bool condition,
                  const std::string& errorString,
                  const char *file,
                  const int line )
{
    if (!condition) {
        std::cerr << "ERROR: "
                  << errorString << std::endl
                  << "in " << file << " line: " << line << std::endl;
        exit( -1 );
    }
}

/** 
 * @brief Extract filename and prefix
 *        Overloads are defined below
 * 
 * @param filename     The filename
 * @param prefix       Prefix (if there is)
 * 
 * @return std::string Filename with prefix extracted
 */
std::string extract_filename(const std::string& filename, std::string & prefix)
{
    std::string ret = filename;

    /* UNIX path delimiter */
    size_t delimiter_pos = ret.find_last_of('/');
    ret.erase(0, delimiter_pos + 1);

    /* Strip the extension */
    size_t extension_pos = ret.find_last_of('.');

    /* Files with no extension */
    if (extension_pos == std::string::npos) {
        prefix = "";
        return ret;
    }

    prefix = ret.substr(extension_pos);
    ret.erase(ret.begin() + extension_pos, ret.end());
    return ret;    
}

