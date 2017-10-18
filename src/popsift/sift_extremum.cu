/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "sift_extremum.h"
#include <iomanip>
#include <iostream>
#include <unistd.h>
#ifndef __APPLE__
#include <malloc.h>
#endif
#include <stdlib.h>
#include <errno.h>

using namespace std;

namespace popsift {

Features::Features( )
    : _ext( 0 )
    , _ori( 0 )
    , _num_ext( 0 )
    , _num_ori( 0 )
{ }

Features::Features( int num_ext, int num_ori )
    : _ext( 0 )
    , _ori( 0 )
{
    reset( num_ext, num_ori );
}

Features::~Features( )
{
    free( _ext );
    free( _ori );
}

#ifdef __APPLE__
void* memalign( size_t alignment, size_t size )
{
    void* ret;
    int err = posix_memalign( &ret, alignment, size );
    if( err != 0 ) {
        errno = err;
        ret = 0;
    }
    return ret;
}
#endif

void Features::reset( int num_ext, int num_ori )
{
    if( _ext != 0 ) { free( _ext ); _ext = 0; }
    if( _ori != 0 ) { free( _ori ); _ori = 0; }

    _ext = (Feature*)memalign( sysconf(_SC_PAGESIZE), num_ext * sizeof(Feature) );
    if( _ext == 0 ) {
        cerr << __FILE__ << ":" << __LINE__ << " Runtime error:" << endl
             << "    Failed to (re)allocate memory for downloading " << num_ext << " features" << endl;
        if( errno == EINVAL ) cerr << "    Alignment is not a power of two." << endl;
        if( errno == ENOMEM ) cerr << "    Not enough memory." << endl;
        exit( -1 );
    }
    _ori = (Descriptor*)memalign( sysconf(_SC_PAGESIZE), num_ori * sizeof(Descriptor) );
    if( _ori == 0 ) {
        cerr << __FILE__ << ":" << __LINE__ << " Runtime error:" << endl
             << "    Failed to (re)allocate memory for downloading " << num_ori << " descriptors" << endl;
        if( errno == EINVAL ) cerr << "    Alignment is not a power of two." << endl;
        if( errno == ENOMEM ) cerr << "    Not enough memory." << endl;
        exit( -1 );
    }

    _num_ext = num_ext;
    _num_ori = num_ori;
}

void Features::pin( )
{
    cudaError_t err;
    err = cudaHostRegister( _ext, _num_ext * sizeof(Feature), 0 );
    if( err != cudaSuccess ) {
        cerr << __FILE__ << ":" << __LINE__ << " Runtime warning:" << endl
             << "    Failed to register feature memory in CUDA." << endl
             << "    " << cudaGetErrorString(err) << endl;
    }
    err = cudaHostRegister( _ori, _num_ori * sizeof(Descriptor), 0 );
    if( err != cudaSuccess ) {
        cerr << __FILE__ << ":" << __LINE__ << " Runtime warning:" << endl
             << "    Failed to register descriptor memory in CUDA." << endl
             << "    " << cudaGetErrorString(err) << endl;
    }
}

void Features::unpin( )
{
    cudaHostUnregister( _ext );
    cudaHostUnregister( _ori );
}

void Features::print( std::ostream& ostr, bool write_as_uchar ) const
{
    for( int i=0; i<_num_ext; i++ ) {
        _ext[i].print( ostr, write_as_uchar );
    }
}

std::ostream& operator<<( std::ostream& ostr, const Features& feature )
{
    feature.print( ostr, false );
    return ostr;
}

void Feature::print( std::ostream& ostr, bool write_as_uchar ) const
{
    float sigval =  1.0f / ( sigma * sigma );

    for( int ori=0; ori<num_ori; ori++ ) {
        ostr << xpos << " " << ypos << " "
             << sigval << " 0 " << sigval << " ";
        if( write_as_uchar ) {
            for( int i=0; i<128; i++ ) {
                ostr << roundf(desc[ori]->features[i]) << " ";
            }
        } else {
            ostr << std::setprecision(3);
            for( int i=0; i<128; i++ ) {
                ostr << desc[ori]->features[i] << " ";
            }
            ostr << std::setprecision(6);
        }
        ostr << std::endl;
    }
}

std::ostream& operator<<( std::ostream& ostr, const Feature& feature )
{
    feature.print( ostr, false );
    return ostr;
}

} // namespace popsift
