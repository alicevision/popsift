/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "pgmread.h"

#include <boost/algorithm/string/trim.hpp>
#include <boost/filesystem.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>

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

unsigned char* readPGMfile( const string& filename, int& w, int& h )
{
    boost::filesystem::path input_file( filename );

    if( ! boost::filesystem::exists( input_file ) ) {
        cerr << "File " << input_file << " does not exist" << endl;
        return nullptr;
    }

    ifstream pgmfile( filename.c_str(), ios::binary );
    if( ! pgmfile.is_open() ) {
        cerr << "File " << input_file << " could not be opened for reading" << endl;
        return nullptr;
    }

    string pgmtype;
    do {
        getline( pgmfile, pgmtype ); // this is the string version of getline()
        if( pgmfile.fail() ) {
            cerr << "File " << input_file << " is too short" << endl;
            return nullptr;
        }
        boost::algorithm::trim_left( pgmtype ); // nice because of trim
    } while( pgmtype.at(0) == '#' );

    int type;
    if( pgmtype.substr(0,2) == "P2" ) type = 2;
    else if( pgmtype.substr(0,2) == "P3" ) type = 3;
    else if( pgmtype.substr(0,2) == "P5" ) type = 5;
    else if( pgmtype.substr(0,2) == "P6" ) type = 6;
    else {
        cerr << "File " << input_file << " can only contain P2, P3, P5 or P6 PGM images" << endl;
        return nullptr;
    }

    const int maxLineSize{1000};
    char  line[maxLineSize];
    char* parse{nullptr};
    int   maxval{};

    do {
        pgmfile.getline( line, maxLineSize );

        if( pgmfile.fail() ) {
            cerr << "File " << input_file << " is too short" << endl;
            return nullptr;
        }
        int num = pgmfile.gcount();
        parse = line;
        while( (num-- > 0) && isspace( *parse ) ) {
            parse++;
        }
        if( *parse == '#' ) continue;
        const int ct = sscanf( parse, "%d %d", &w, &h );
        if( ct != 2 ) {
            cerr << "Error in " << __FILE__ << ":" << __LINE__ << endl
                 << "File " << input_file << " PGM type header (" << type << ") must be followed by comments and WxH info" << endl
                 << "but line contains " << parse << endl;
            return nullptr;
        }
    } while( *parse == '#' );

    if( w <= 0 || h <= 0 ) {
        cerr << "File " << input_file << " has meaningless image size" << endl;
        return nullptr;
    }

    do {
        pgmfile.getline( line, maxLineSize );
        if( pgmfile.fail() ) {
            cerr << "File " << input_file << " is too short" << endl;
            return nullptr;
        }
        int num = pgmfile.gcount();
        parse = line;
        while( (num-- > 0) && isspace( *parse ) ) {
            parse++;
        }
        if( *parse == '#' ) continue;
        const int ct = sscanf( parse, "%d", &maxval );
        if( ct != 1 ) {
            cerr << "File " << input_file << " PGM dimensions must be followed by comments and max value info" << endl;
            return nullptr;
        }
    } while( *parse == '#' );

    auto input_data = new unsigned char[ w * h ];

    switch( type )
    {
    case 2 :
        for( int i=0; i<w*h; i++ ) {
            int input;
            pgmfile >> input;
            if( maxval == 255 ) {
                input_data[i] = input;
            } else {
                input_data[i] = (unsigned char)(input * 255.0 / maxval );
            }
            if( pgmfile.fail() ) {
                cerr << "File " << input_file << " file too short" << endl;
                delete [] input_data;
                return nullptr;
            }
        }
        break;
    case 3 :
        {
            auto i2 = new unsigned char[ w * h * 3 ];
            unsigned char* src = i2;
            for( int i=0; i<w*h*3; i++ ) {
                int input;
                pgmfile >> input;
                if( maxval == 255 ) {
                    i2[i] = input;
                } else {
                    i2[i] = (unsigned char)(input * 255.0 / maxval );
                }
                if( pgmfile.fail() ) {
                    cerr << "File " << input_file << " file too short" << endl;
                    delete [] i2;
                    delete [] input_data;
                    return nullptr;
                }
            }
            for( int i=0; i<w*h; i++ ) {
#ifdef RGB2GRAY_IN_INT
                const unsigned int r = *src; src++;
                const unsigned int g = *src; src++;
                const unsigned int b = *src; src++;
                const unsigned int res = ( ( R_RATE*r+G_RATE*g+B_RATE*b ) >> RATE_SHIFT );
                input_data[i] = (unsigned char)res;
#else // RGB2GRAY_IN_INT
                const float r = *src; src++;
                const float g = *src; src++;
                const float b = *src; src++;
                input_data[i] = (unsigned char)( R_RATE*r+G_RATE*g+B_RATE*b );
#endif // RGB2GRAY_IN_INT
            }
            delete [] i2;
        }
        break;
    case 5 :
        if( maxval < 256 ) {
            pgmfile.read( (char*)input_data, w*h );
        } else {
            auto i2 = new unsigned short[ w * h ];
            pgmfile.read( (char*)i2, w*h*2 );
            if( pgmfile.fail() ) {
                cerr << "File " << input_file << " file too short" << endl;
                delete [] i2;
                delete [] input_data;
                return nullptr;
            }
            for( int i=0; i<w*h; i++ ) {
                input_data[i] = (unsigned char)(i2[i] * 255.0 / maxval );
            }
            delete [] i2;
        }
        break;
    case 6 :
        if( maxval < 256 ) {
            auto i2 = new unsigned char[ w * h * 3 ];
            unsigned char* src = i2;
            pgmfile.read( (char*)i2, w*h*3 );
            if( pgmfile.fail() ) {
                cerr << "File " << input_file << " file too short" << endl;
                delete [] i2;
                delete [] input_data;
                return nullptr;
            }
            for( int i=0; i<w*h; i++ ) {
#ifdef RGB2GRAY_IN_INT
                unsigned int r = *src; src++;
                unsigned int g = *src; src++;
                unsigned int b = *src; src++;
                unsigned int res = ( ( R_RATE*r+G_RATE*g+B_RATE*b ) >> RATE_SHIFT );
                input_data[i] = (unsigned char)res;
#else // RGB2GRAY_IN_INT
                float r = *src; src++;
                float g = *src; src++;
                float b = *src; src++;
                input_data[i] = (unsigned char)( R_RATE*r+G_RATE*g+B_RATE*b );
#endif // RGB2GRAY_IN_INT
            }
            delete [] i2;
        } else {
            auto i2 = new unsigned short[ w * h * 2 * 3 ];
            unsigned short* src = i2;
            pgmfile.read( (char*)i2, w*h*2*3 );
            if( pgmfile.fail() ) {
                cerr << "File " << input_file << " file too short" << endl;
                delete [] i2;
                delete [] input_data;
                return 0;
            }
            for( int i=0; i<w*h; i++ ) {
#ifdef RGB2GRAY_IN_INT
                unsigned int r = *src; src++;
                unsigned int g = *src; src++;
                unsigned int b = *src; src++;
                unsigned int res = ( ( R_RATE*r+G_RATE*g+B_RATE*b ) >> RATE_SHIFT );
                input_data[i] = (unsigned char)res;
#else // RGB2GRAY_IN_INT
                float r = *src; src++;
                float g = *src; src++;
                float b = *src; src++;
                input_data[i] = (unsigned char)( R_RATE*r+G_RATE*g+B_RATE*b );
#endif // RGB2GRAY_IN_INT
            }
            delete [] i2;
        }
        break;

    default:
        throw std::runtime_error("unsupported type " + std::to_string(type));
    }

    return input_data;
}

