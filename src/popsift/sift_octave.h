/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "s_image.h"
#include "sift_conf.h"
#include "sift_constants.h"
#include "sift_extremum.h"
#include "common/plane_2d.h"

#include <iostream>
#include <vector>

namespace popsift {

class Octave
{
    int   _w{};
    int   _h{};
    int   _levels{};

    float _w_grid_divider{};
    float _h_grid_divider{};
    int   _debug_octave_id{};

    Plane2D_float _data;
    Plane2D_float _intm;
    Plane2D_float _dog_3d;

public:
    Octave( );
    ~Octave( ) { this->free(); }

    inline void debugSetOctave( uint32_t o ) { _debug_octave_id = o; }

    inline int getLevels() const { return _levels; }
    inline int getWidth()  const { return _w; }
    inline int getHeight() const { return _h; }

    inline float getWGridDivider() const  {
        return _w_grid_divider;
    }
    inline float getHGridDivider() const {
        return _h_grid_divider;
    }

     /**
      * @brief Allocates all GPU memories for one octave.
      * @param conf
      * @param width in floats
      * @param height
      * @param levels
      * @param gauss_group
      */
    void alloc( const Config& conf,
                int           width,
                int           height,
                int           levels,
                int           gauss_group );
    void free();

    void resetDimensions( const Config& conf, int w, int h );

    /**
     * debug:
     * download a level and write to disk
     */
    void download_and_save_array( const char* basename, int octave );
};

} // namespace popsift
