#pragma once

// #include <fstream>
#include <string>
#include <iostream>

#include "c_util.hpp"

/* type definitions */
typedef unsigned char    pixel_uc;
typedef float            pixel_f;
typedef unsigned char    uchar;
typedef unsigned int     uint;

/*************************************/
/* @namespace Comment                */
/* @brief Comment crusher            */
/*        Erase the upcoming comment */
/*************************************/

/** 
 * @brief Comment crusher
 *        Erase the upcoming comments
 * 
 * @param is 
 * @param manip 
 * 
 * @return 
 */
namespace Comment {
    static class _comment {
    } cmnt;
    inline std::istream& operator>>(std::istream& is, _comment & manip) {
        char c;
        char b[1024];
        is >> c;
        if (c != '#')
            return is.putback(c);
        is.getline(b, 1024);
        return is;
    }
}

struct imgStream {
    int width;
    int height;
    pixel_uc *data_r;
    pixel_uc *data_g;
    pixel_uc *data_b;
};

/** 
 * @param filename 
 * 
 * @return bool Return TRUE if file found, FALSE if not found
 */

bool find_file(std::string & filename);

/** 
 * @brief Take an image input, PGM or PPM format, and then
 *        save it into the imgStream buffer
 * 
 * @param filename 
 * @param buffer 
 * @param isPGM 
 */

void read_pgpm(std::string & filename, imgStream & buffer, int &isPGM);
void read_gray(std::string & filename, imgStream & buffer);

/** 
 * @brief Output image in PGM or PPM format
 * 
 * @param filename 
 * @param buffer 
 * @param isPGM 
 */

void out_pgpm(const std::string & filename, imgStream & buffer, int &isPGM);

