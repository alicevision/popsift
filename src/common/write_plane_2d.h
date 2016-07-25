#pragma once

#include "plane_2d.h"

namespace popart {

void write_plane2D( const char* filename, bool onDevice, Plane2D_float& f );
void write_plane2D( const char* filename, Plane2D_float& f );

void write_plane2Dunscaled( const char* filename, bool onDevice, Plane2D_float& f, int offset=0 );
void write_plane2Dunscaled( const char* filename, Plane2D_float& f, int offset=0 );

void dump_plane2Dfloat( const char* filename, bool onDevice, Plane2D_float& f );
void dump_plane2Dfloat( const char* filename, Plane2D_float& f );
} // namespace popart

