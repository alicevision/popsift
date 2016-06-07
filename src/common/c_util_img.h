#pragma once

#include <string>
#include <iostream>

#include "c_util.h"

/* type definitions */
typedef unsigned char    pixel_uc;
typedef float            pixel_f;
typedef unsigned char    uchar;
typedef unsigned int     uint;

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

