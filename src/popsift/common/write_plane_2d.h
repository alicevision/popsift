/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "plane_2d.h"

namespace popsift {

void write_plane2D( const char* filename, bool onDevice, Plane2D_float& f );
void write_plane2D( const char* filename, Plane2D_float& f );

void write_plane2Dunscaled( const char* filename, bool onDevice, Plane2D_float& f, int offset=0 );
void write_plane2Dunscaled( const char* filename, Plane2D_float& f, int offset=0 );

void dump_plane2Dfloat( const char* filename, bool onDevice, Plane2D_float& f );
void dump_plane2Dfloat( const char* filename, Plane2D_float& f );
} // namespace popsift

