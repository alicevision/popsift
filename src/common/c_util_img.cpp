#include "c_util_img.h"

#include <fstream>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

/* GRAYSCALE */
#define RGB2GRAY_IN_INT

#ifdef RGB2GRAY_IN_INT
// #define RATE_SHIFT 24
// #define R_RATE (uint32_t)(0.298912F * (float)(1<<RATE_SHIFT))
// #define G_RATE (uint32_t)(0.586611F * (float)(1<<RATE_SHIFT))
// #define B_RATE (uint32_t)(0.114478F * (float)(1<<RATE_SHIFT))
// the following are OpenCV's numbers
#define RATE_SHIFT 14
#define R_RATE (uint32_t)4899
#define G_RATE (uint32_t)9617
#define B_RATE (uint32_t)1868
#else
#define R_RATE 0.298912f
#define G_RATE 0.586611f
#define B_RATE 0.114478f
#endif

using namespace std;

/*************************************/
/* @namespace Comment                */
/* @brief Comment crusher            */
/*************************************/
namespace Comment {
    static class _comment {
    } cmnt;
    inline std::istream& operator>>(std::istream& is, _comment & manip) {
        char c;
        char b[1024];
        while( is.good() )
        {
            is >> c;
            if (c != '#')
                return is.putback(c);
            is.getline(b, 1024);
        }
        return is;
    }
}

/** 
 * BEMAP Utility for finding an input file
 * Finding process iterates on the given folder:

 "./",
 "common/data/",
 "../common/data/",
 "../../common/data/",
 "../../../common/data/",
 "../../../../common/data/"

 * 
 * @param filename 
 * 
 * @return bool Return TRUE if file found, FALSE if not found
 */

bool find_file(std::string & filename)
{
    /* try to open the given file first */
    std::fstream tryin(filename.c_str(), std::ios_base::in | std::ios_base::binary);
    if (tryin.good()) {
        return true;
    }

    const char* search_path[] = {
        "./",
        "common/data/",
        "../common/data/",
        "../../common/data/",
        "../../../common/data/",
        "../../../../common/data/"
    };

    /* extract file name first */
    std::string prefix = "";
    std::string realname = "";

    realname = extract_filename(filename, prefix);
    realname += prefix;

    std::cerr << "File not found: " << realname << std::endl;
    std::cerr << "find_file is trying to search " << realname << " in another path..\n";

    if (filename != "") {
        for (int i = 0; i < sizeof(search_path)/sizeof(char *); i++) {
            std::string path(search_path[i]);
            path.append(realname);

            std::fstream trying(path.c_str(), std::ios_base::in | std::ios_base::binary);
            if (trying.good()) {
                /* file found, return this */
                std::cerr << path << " found. Proceed to execute with this.\n";
                filename = path;
                return true;
            }
        }
    }

    return false;
}

/** 
 * @brief Take an image input, PGM or PPM format, and then
 *        save it into the imgStream buffer
 * 
 * @param filename 
 * @param buffer 
 * @param isPGM 
 */

void read_pgpm(std::string & filename, imgStream & buffer, int &isPGM)
{
    if (!find_file(filename)) {
        std::cerr << "ERROR: " << "Cannot open file: " << filename << std::endl;
        exit(1);
    }

    std::fstream in(filename.c_str(),
                    std::ios_base::in | std::ios_base::binary);

    pixel_uc *ptr_r = NULL;
    pixel_uc *ptr_g = NULL;
    pixel_uc *ptr_b = NULL;
    int width, height, maxval;
    char c;
    bool isAscii = false;

    in >> c;
    if (c != 'P') {
        std::cerr << "ERROR: " << "The file is not PGM/PPM format" << std::endl;
        exit(1);
    }

    in >> c;
    switch (c) {
    case '2':
        isAscii = true;
    case '5':
        isPGM = true;
        break;
    case '6':
        isPGM = false;
        break;
    default:
        std::cerr << "ERROR: " << "The file is not PGM/PPM format" << std::endl;
        exit(1);
    }

    in >> Comment::cmnt
       >> width >> Comment::cmnt >> height >> Comment::cmnt >> maxval;

    /* get trash */
    {
        char trash;
        in.get(trash);
    }

    if (maxval > 255) {
        std::cerr << "ERROR: " << "Only 8bit color channel is supported" << std::endl;
        exit(1);
    }
    if (!in.good()) {
        std::cerr << "ERROR: " << "PGM header parsing error" << std::endl;
        exit(1);
    }

    if (isPGM && isAscii) {     // PGM P2 image
        ptr_r = new pixel_uc[width * height];
        pixel_uc *start = ptr_r;
        pixel_uc *end;
        end = start + width * height;

        while (start != end) {
            int i;
            in >> i;
            if (!in.good() || i > maxval) {
                std::cerr << "ERROR: " << "PGM parsing error: " << filename << std::endl
                          << "at pixel: " << start - ptr_r << std::endl;
                exit(1);
            }
            *start++ = pixel_uc(i);
        }
    }

    else if (isPGM && !isAscii) {       // PGM P5 image
        ptr_r = new pixel_uc[width * height];
        pixel_uc *start = ptr_r;
        pixel_uc *end = start + width * height;

        std::streampos beg = in.tellg();
        char *buffer = new char[width * height];
        in.read(buffer, width * height);
        if (!in.good()) {
            std::cerr << "ERROR:"  << "PGM parsing error: " << filename << std::endl
                      << "at pixel: " << start - ptr_r << std::endl;
            exit(1);
        }

        while (start != end)
            *start++ = (pixel_uc) (*buffer++);
    } else if (!isPGM && !isAscii) {    // PPM P6 image
        ptr_r = new pixel_uc[width * height];
        ptr_g = new pixel_uc[width * height];
        ptr_b = new pixel_uc[width * height];
        pixel_uc *start_r = ptr_r;
        pixel_uc *start_g = ptr_g;
        pixel_uc *start_b = ptr_b;
        pixel_uc *end = start_r + width * height;

        std::streampos beg = in.tellg();
        char *buffer = new char[width * height * 3];
        in.read(buffer, width * height * 3);
        if (!in.good()) {
            std::cerr << "ERROR:" << "PGM parsing error: " << filename << std::endl
                      << "at pixel: " << start_r - ptr_r << std::endl;
            exit(1);
        }

        pixel_uc *src = reinterpret_cast < pixel_uc * >(buffer);
        while (start_r != end) {
            *start_r++ = *src++;
            *start_g++ = *src++;
            *start_b++ = *src++;
        }
    } else {
        std::cerr << "ERROR: Unknown parsing mode\n" << std::endl;
        exit(1);
    }

    buffer.width = width;
    buffer.height = height;
    buffer.data_r = ptr_r;
    buffer.data_g = (!isPGM) ? ptr_g : NULL;
    buffer.data_b = (!isPGM) ? ptr_b : NULL;
}

void read_gray(std::string & filename, imgStream & buffer)
{
    // cerr << "Entering " << __FUNCTION__ << endl;
    // cerr << "   filename " << filename << endl;

    bool isPGM;

    if (!find_file(filename)) {
        std::cerr << "ERROR: " << "Cannot open file: " << filename << std::endl;
        exit(1);
    }

    std::fstream in(filename.c_str(),
                    std::ios_base::in | std::ios_base::binary);

    pixel_uc *ptr_r = NULL;
    int width, height, maxval;
    char c;
    bool isAscii = false;

    in >> c;
    if (c != 'P') {
        std::cerr << "ERROR: " << "The file is not PGM/PPM format" << std::endl;
        exit(1);
    }

    in >> c;
    switch (c) {
    case '2':
        isAscii = true;
    case '5':
        isPGM = true;
        break;
    case '6':
        // cerr << "    File is in PPM format" << endl;
        isPGM = false;
        break;
    default:
        std::cerr << "ERROR: " << "The file is not PGM/PPM format" << std::endl;
        exit(1);
    }

    in >> Comment::cmnt
       >> width >> Comment::cmnt >> height >> Comment::cmnt >> maxval;

    /* get trash */
    {
        char trash;
        in.get(trash);
    }

    if (maxval > 255) {
        std::cerr << "ERROR: " << "Only 8bit color channel is supported" << std::endl;
        exit(1);
    }
    if (!in.good()) {
        std::cerr << "ERROR: " << "PGM header parsing error" << std::endl;
        exit(1);
    }

    if( isPGM ) {
      if( isAscii ) { // PGM P2 image
        ptr_r = new pixel_uc[width * height];
        pixel_uc *start = ptr_r;
        pixel_uc *end;
        end = start + width * height;

        while (start != end) {
            int i;
            in >> i;
            if (!in.good() || i > maxval) {
                std::cerr << "ERROR: " << "PGM parsing error: " << filename << std::endl
                          << "at pixel: " << start - ptr_r << std::endl;
                exit(1);
            }
            *start++ = pixel_uc(i);
        }
      } else {
        ptr_r = new pixel_uc[width * height];
        pixel_uc *start = ptr_r;
        pixel_uc *end = start + width * height;

        std::streampos beg = in.tellg();
        char *buffer = new char[width * height];
        in.read(buffer, width * height);
        if (!in.good()) {
            std::cerr << "ERROR:"  << "PGM parsing error: " << filename << std::endl
                      << "at pixel: " << start - ptr_r << std::endl;
            exit(1);
        }

        while (start != end)
            *start++ = (pixel_uc) (*buffer++);
      }
    } else {
      // cerr << "    NOT PGM case" << endl;
      if( not isAscii ) {    // PPM P6 image
        ptr_r = new pixel_uc[width * height];

        std::streampos beg = in.tellg();
        char *buffer = new char[width * height * 3];
        in.read(buffer, width * height * 3);
        if (!in.good()) {
            std::cerr << "ERROR:" << "PGM parsing error: " << filename << std::endl;
            exit(1);
        }

        pixel_uc *src = reinterpret_cast < pixel_uc * >(buffer);
        for( int i=0; i<width * height; i++ ) {
#ifdef RGB2GRAY_IN_INT
            uint32_t r = *src; src++;
            uint32_t g = *src; src++;
            uint32_t b = *src; src++;
            uint32_t res = ( ( R_RATE*r+G_RATE*g+B_RATE*b ) >> RATE_SHIFT );
            ptr_r[i] = (unsigned char)res;
#else // RGB2GRAY_IN_INT
            float r =(*src; src++;
            float g = *src; src++;
            float b = *src; src++;
            ptr_r[i] = (unsigned char)( R_RATE*r+G_RATE*g+B_RATE*b );
#endif // RGB2GRAY_IN_INT
    // cerr << "(" << i%width << "," << i/width << ") -> R " << r << " G " << g << " B " << b << " -> " << (int)ptr_r[i] << endl;
// if( i == 19*width+3 ) {
    // cerr << "(3,19) -> R " << r << " G " << g << " B " << b << " -> " << (int)ptr_r[i] << endl;
// }
        }
      } else {
        std::cerr << "ERROR: Unknown parsing mode\n" << std::endl;
        exit(1);
      }
    }

    buffer.width  = width;
    buffer.height = height;
    buffer.data_r = ptr_r;
    buffer.data_g = 0;
    buffer.data_b = 0;

    // cerr << "Leaving " << __FUNCTION__ << endl;
}

/** 
 * @brief Output image in PGM or PPM format
 * 
 * @param filename 
 * @param buffer 
 * @param isPGM 
 */

void out_pgpm(const std::string & filename, imgStream & buffer, int &isPGM)
{
    int x, y;
    pixel_uc r, g, b;

    int width = buffer.width;
    int height = buffer.height;
    pixel_uc *ptr_r = buffer.data_r;
    pixel_uc *ptr_g = buffer.data_g;
    pixel_uc *ptr_b = buffer.data_b;

    std::fstream out(filename.c_str(),
                     std::ios_base::out | std::ios_base::binary);
    if (!out.good()) {
        std::cerr << "ERROR: " << "Cannot open file: " << filename << std::endl;
        exit(1);
    }

    if (isPGM) {
        out << "P5" << "\n"
            << width << " " << height << "\n" << "255" << "\n";

        for (y = 0; y < height; ++y) {
            for (x = 0; x < width; ++x) {
                r = (pixel_uc) * ptr_r++;
                out << r;
            }
        }
    } else {
        out << "P6" << "\n"
            << width << " " << height << "\n" << "255" << "\n";

        for (y = 0; y < height; ++y) {
            for (x = 0; x < width; ++x) {
                r = (pixel_uc) * ptr_r++;
                g = (pixel_uc) * ptr_g++;
                b = (pixel_uc) * ptr_b++;
                out << r << g << b;
            }
        }
    }
}

