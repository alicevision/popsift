/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "write_plane_2d.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>

using namespace std;

namespace popsift {

__host__
void write_plane2D( const char* filename, bool onDevice, Plane2D_float& f )
{
    if( onDevice ) {
        // cerr << __FILE__ << ":" << __LINE__ << ": copying from device" << endl;
        Plane2D_float g;
        g.allocHost( f.getCols(), f.getRows(), CudaAllocated );
        g.memcpyFromDevice( f );
        write_plane2D( filename, g );
        g.freeHost( CudaAllocated );
    } else {
        write_plane2D( filename, f );
    }
}

__host__
void write_plane2Dunscaled( const char* filename, bool onDevice, Plane2D_float& f, int offset )
{
    if( onDevice ) {
        // cerr << __FILE__ << ":" << __LINE__ << ": copying from device" << endl;
        Plane2D_float g;
        g.allocHost( f.getCols(), f.getRows(), CudaAllocated );
        g.memcpyFromDevice( f );
        write_plane2Dunscaled( filename, g, offset );
        g.freeHost( CudaAllocated );
    } else {
        write_plane2Dunscaled( filename, f, offset );
    }
}

__host__
void write_plane2D( const char* filename, Plane2D_float& f )
{
    // cerr << "Enter " << __FUNCTION__ << endl;

    int rows = f.getRows();
    int cols = f.getCols();
    // cerr << "    size: " << cols << "x" << rows << endl;

    unsigned char* c = new unsigned char[rows * cols];
    float minval = std::numeric_limits<float>::max();
    float maxval = std::numeric_limits<float>::min();
    for( int y=0; y<rows; y++ ) {
        for( int x=0; x<cols; x++ ) {
            float v = f.ptr(y)[x];
            // cerr << " " << v;
            minval = min( minval, v );
            maxval = max( maxval, v );
        }
    }
    // cerr << endl;

    // cerr << "    minval: " << minval << endl;
    // cerr << "    maxval: " << maxval << endl;

    float fmaxval = 255.0f / ( maxval - minval );
    for( int y=0; y<rows; y++ ) {
        for( int x=0; x<cols; x++ ) {
            float v = f.ptr(y)[x];
            v = ( v - minval ) * fmaxval;
            c[y*cols+x] = (unsigned char)v;
        }
    }
#if 1
    ofstream of( filename, ios::binary );
    of << "P2" << endl 
       << cols << " " << rows << endl
       << "255" << endl;
    unsigned char* cx = c;
    for( int row=0; row<rows; row++ ) {
        for( int col=0; col<cols; col++ ) {
            int val = *cx;
            cx++;
            of << val << " ";
        }
        of << endl;
    }
    delete [] c;
#else
    ofstream of( filename, ios::binary );
    of << "P5" << endl
       << cols << " " << rows << endl
       << "255" << endl;
    of.write( (char*)c, cols * rows );
    delete [] c;
#endif

    // cerr << "Leave " << __FUNCTION__ << endl;
}

__host__
void write_plane2Dunscaled( const char* filename, Plane2D_float& f, int offset )
{
    int rows = f.getRows();
    int cols = f.getCols();

    int* c = new int[rows * cols];
    for( int y=0; y<rows; y++ ) {
        for( int x=0; x<cols; x++ ) {
            float v = f.ptr(y)[x];
            c[y*cols+x] = v;
        }
    }

    ofstream of( filename, ios::binary );
    of << "P2" << endl
       << cols << " " << rows << endl
       << "255" << endl;
    int* cx = c;
    for( int row=0; row<rows; row++ ) {
        for( int col=0; col<cols; col++ ) {
            int val = *cx;
            cx++;
            of << val+offset << " ";
        }
        of << endl;
    }
    delete [] c;

    // cerr << "Leave " << __FUNCTION__ << endl;
}

__host__
void dump_plane2Dfloat( const char* filename, bool onDevice, Plane2D_float& f )
{
    if( onDevice ) {
        // cerr << __FILE__ << ":" << __LINE__ << ": copying from device" << endl;
        Plane2D_float g;
        g.allocHost( f.getCols(), f.getRows(), CudaAllocated );
        g.memcpyFromDevice( f );
        dump_plane2Dfloat( filename, g );
        g.freeHost( CudaAllocated );
    } else {
        dump_plane2Dfloat( filename, f );
    }
}

__host__
void dump_plane2Dfloat( const char* filename, Plane2D_float& f )
{
    int rows = f.getRows();
    int cols = f.getCols();

    float* c = new float[rows * cols];
    for( int y=0; y<rows; y++ ) {
        for( int x=0; x<cols; x++ ) {
            float v = f.ptr(y)[x];
            c[y*cols+x] = v;
        }
    }

    ofstream of( filename, ios::binary );
    of << "floats" << endl
       << cols << " " << rows << endl;
    of.write( (char*)c, rows*cols*sizeof(float) );
    delete [] c;
}

} // namespace popsift

