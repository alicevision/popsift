/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>

#define POP_FATAL(s)                                                                                                   \
    {                                                                                                                  \
        std::stringstream ss;                                                                                          \
        ss << __FILE__ << ":" << __LINE__ << std::endl << "    " << s;                                                 \
        throw std::runtime_error{ss.str()};                                                                            \
    }

#define POP_FATAL_FL(s, file, line)                                                                                    \
    {                                                                                                                  \
        std::stringstream ss;                                                                                          \
        ss << file << ":" << line << std::endl << "    " << s << std::endl;                                            \
        throw std::runtime_error{ss.str()};                                                                            \
    }

#define POP_CHECK_NON_NULL(ptr,s) if( ptr == 0 ) { POP_FATAL_FL(s,__FILE__,__LINE__); }

#define POP_CHECK_NON_NULL_FL(ptr,s,file,line) if( ptr == 0 ) { POP_FATAL_FL(s,file,line); }

#define POP_INFO(s)
// #define POP_INFO(s) cerr << __FILE__ << ":" << __LINE__ << std::endl << "    " << s << endl

#define POP_INFO2(silent,s) \
    if (! silent) { \
        std::stringstream ss; \
        std::string       filename; \
        ss << __FILE__; \
        size_t pos = ss.str().find_last_of("/"); \
        if( pos == std::string::npos ) \
            filename = ss.str(); \
        else \
            filename = ss.str().substr( pos+1 ); \
        std::cerr << filename << ":" << __LINE__ << ": " << s << std::endl; \
    }

#define POP_WARN(s) { \
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl; \
        std::cerr << "    WARNING: " << s << std::endl; \
    }

