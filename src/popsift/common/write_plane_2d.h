/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <vector>

#include "simd_types.h"
#include "plane_2d.h"

namespace popsift {

void write_plane2D( const char* filename, bool onDevice, Plane2D_float& f );
void write_plane2D( const char* filename, Plane2D_float& f );

void write_plane2Dunscaled( const char* filename, bool onDevice, Plane2D_float& f, int offset=0 );
void write_plane2Dunscaled( const char* filename, Plane2D_float& f, int offset=0 );

/* Write a PPM file instead of a PGM file. All pixel are greyscale pixels as usual, but for this
 * coordinates lists in the list red, only red=255 is written, thus giving is a read color.
 */
void write_plane2Dppm( const char* filename, Plane2D_float& f, const std::vector<int2>& red );

void dump_plane2Dfloat( const char* filename, bool onDevice, Plane2D_float& f );
void dump_plane2Dfloat( const char* filename, Plane2D_float& f );

void write_plane2D( const char* filename, Plane2D_uint8& f );

} // namespace popsift

