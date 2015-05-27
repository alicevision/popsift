#ifndef __C_UTIL_HPP__
#define __C_UTIL_HPP__

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cstdlib>

// #define INF (1<<29)
// #define NINF (-INF)
// #define EPS 1e-15

/* Solve min max problem */
// #ifndef max
// #define max(a, b) (((a) > (b)) ? (a) : (b))
// #endif
// #ifndef min
// #define min(a, b) (((a) < (b)) ? (a) : (b))
// #endif

/* M_PI */
// #ifdef M_PI
// #undef M_PI
// #endif
// #define M_PI  3.1415926535897932384626433832f
// #define M_PI2 2.0f * M_PI

/* std::vector types */
typedef std::vector <int>        array_i;
typedef std::vector <array_i>    matrix_i;
typedef std::vector <float>      array_f;
typedef std::vector <array_f>    matrix_f;
typedef std::vector <double>     array_d;
typedef std::vector <array_d>    matrix_d;
typedef unsigned int             uint;


/* Error handler */
#define ERROR_HANDLER(x, y) __check_error((x), (y), __FILE__, __LINE__)

/** 
 * @brief Force exit if condition is 0
 *        MACRO -> ERROR_HANDLER(x, y)
 * 
 * @param condition      Condition to be checked
 * @param errorString    Collection of error strings
 * @param file           File to be checked
 * @param line           Line number
 */

inline void __check_error(bool condition, const std::string errorString,
                          const char *file, const int line)
{
    if (!condition) {
        std::cerr << "ERROR: "
                  << errorString << std::endl
                  << "in " << file << " line: " << line << std::endl;
        exit(1);
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
inline std::string __extract_filename(const std::string filename, std::string & prefix)
{
    std::string ret = filename;

#ifdef _WIN32

    /* Windows path delimiter */
    size_t delimiter_pos = ret.find_last_of('\\');
    ret.erase(0, delimiter_pos + 1);

    /* CYGWIN path delimiter */
    delimiter_pos = ret.find_last_of('/');
    ret.erase(0, delimiter_pos + 1);

#else
    /* UNIX path delimiter */
    size_t delimiter_pos = ret.find_last_of('/');
    ret.erase(0, delimiter_pos + 1);
#endif

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

/**
 * Overloads of __extract_filename
 * 
 */

inline std::string extract_filename(const std::string filename,
                                    std::string &   prefix)
{
    return __extract_filename(filename, prefix);
}

inline std::string extract_filename(const std::string filename)
{
    std::string nul;
    return __extract_filename(filename, nul);
}

/** 
 * @name itoa
 * @brief Convert int to string
 * 
 * @param in    Integer to be converted
 * 
 * @return string
 */

inline std::string itoa(int in)
{
    std::stringstream ss;
    ss << in;
    return ss.str();
}

/** 
 * @name rand_float
 @ @brief Generate a float random number
 * 
 * @param low     Bottom range
 * @param high    Top range
 * 
 * @return float  Generated random number
 */

inline float rand_float(float low, float high)
{
    float t = (float) rand() / (float) RAND_MAX;
    return (1.0f - t) * low + t * high;
}

#endif /* __C_UTIL_HPP__ */
