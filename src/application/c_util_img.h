/**
 * Copyright (c) 2012, Fixstars Corp.
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are 
 * met:
 *     * Redistributions of source code must retain the above copyright and 
 *       patent notices, this list of conditions and the following 
 *       disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in 
 *       the documentation and/or other materials provided with the 
 *       distribution.
 *     * Neither the name of Fixstars Corp. nor the names of its 
 *       contributors may be used to endorse or promote products derived 
 *       from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS 
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED 
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
 * HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 */
#pragma once

#include <string>
#include <iostream>

/* type definitions */
typedef unsigned char    pixel_uc;
typedef float            pixel_f;
typedef unsigned char    uchar;
typedef unsigned int     uint;

struct imgStream
{
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

