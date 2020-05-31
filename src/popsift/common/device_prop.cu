/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "debug_macros.h"
#include "device_prop.h"
#include <iostream>
#include <sstream>

namespace popsift { namespace cuda {

using namespace std;

device_prop_t::device_prop_t( )
{
    int         currentDevice;
    cudaError_t err;

    err = cudaGetDevice( &currentDevice );
    POP_CUDA_FATAL_TEST( err, "Cannot get the current CUDA device" );

    err = cudaGetDeviceCount( &_num_devices );
    POP_CUDA_FATAL_TEST( err, "Cannot count devices" );

    _properties.resize(_num_devices);

    for( int n=0; n<_num_devices; ++n ) {
        _properties[n] = new cudaDeviceProp;
        err = cudaGetDeviceProperties( _properties[n], n );
        POP_CUDA_FATAL_TEST( err, "Cannot get properties for a device" );
    }

    err = cudaSetDevice( currentDevice );
    POP_CUDA_FATAL_TEST( err, "Cannot set device 0" );
}

void device_prop_t::print( )
{
    // for( auto ptr : _properties ) {
    std::vector<cudaDeviceProp*>::const_iterator p;
    for( p = _properties.begin(); p!=_properties.end(); p++ ) {
        cudaDeviceProp* ptr = *p;
        std::cout << "Device information:" << endl
                  << "    Name: " << ptr->name << endl
                  << "    Compute Capability:    " << ptr->major << "." << ptr->minor << endl
                  << "    Total device mem:      " << ptr->totalGlobalMem << " B "
                  << ptr->totalGlobalMem/1024 << " kB "
                  << ptr->totalGlobalMem/(1024*1024) << " MB " << endl
                  << "    Per-block shared mem:  " << ptr->sharedMemPerBlock << endl
                  << "    Warp size:             " << ptr->warpSize << endl
                  << "    Max threads per block: " << ptr->maxThreadsPerBlock << endl
                  << "    Max threads per SM(X): " << ptr->maxThreadsPerMultiProcessor << endl
                  << "    Max block sizes:       "
                  << "{" << ptr->maxThreadsDim[0]
                  << "," << ptr->maxThreadsDim[1]
                  << "," << ptr->maxThreadsDim[2] << "}" << endl
                  << "    Max grid sizes:        "
                  << "{" << ptr->maxGridSize[0]
                  << "," << ptr->maxGridSize[1]
                  << "," << ptr->maxGridSize[2] << "}" << endl
                  << "    Number of SM(x)s:      " << ptr->multiProcessorCount << endl
                  << "    Concurrent kernels:    " << (ptr->concurrentKernels?"yes":"no") << endl
                  << "    Mapping host memory:   " << (ptr->canMapHostMemory?"yes":"no") << endl
                  << "    Unified addressing:    " << (ptr->unifiedAddressing?"yes":"no") << endl
                  << endl;
    }
}

void device_prop_t::set( int n, bool print_choice )
{
    cudaError_t err;
    err = cudaSetDevice( n );
    ostringstream ostr;
    ostr << "Cannot set device " << n;
    POP_CUDA_FATAL_TEST( err, ostr.str() );
    if( print_choice ) {
        std::cout << "Choosing device " << n << ": " << _properties[n]->name << std::endl;
    }
}

device_prop_t::~device_prop_t( )
{
    // for( auto ptr : _properties ) {
    std::vector<cudaDeviceProp*>::const_iterator p;
    for( p = _properties.begin(); p!=_properties.end(); p++ ) {
        cudaDeviceProp* ptr = *p;
        delete ptr;
    }
}

bool device_prop_t::checkLimit_2DtexLinear( int& width, int& height, bool printWarn ) const
{
    bool        returnSuccess = true;
    int         currentDevice;
    cudaError_t err;

    err = cudaGetDevice( &currentDevice );
    if( err != cudaSuccess )
    {
        POP_CUDA_WARN( err, "Cannot get current CUDA device" );
        return true;
    }

    if( currentDevice >= _properties.size() )
    {
        POP_WARN( "CUDA device was not registered at program start" );
        return true;
    }

    const cudaDeviceProp* ptr = _properties[currentDevice];
    if( width > ptr->maxTexture2DLayered[0] )
    {
        if( printWarn )
        {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": CUDA device " << currentDevice << std::endl
                      << "    does not support 2D linear textures " << width
                      << " pixels wide." << endl;
        }
        width = ptr->maxTexture2DLayered[0];
        returnSuccess = false;
    }
    if( height > ptr->maxTexture2DLayered[1] )
    {
        if( returnSuccess && printWarn )
        {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": CUDA device " << currentDevice << std::endl
                      << "    does not support 2D linear textures " << height
                      << " pixels high." << endl;
        }
        height = ptr->maxTexture2DLayered[1];
        returnSuccess = false;
    }

    return returnSuccess;
}

bool device_prop_t::checkLimit_2DtexArray( int& width, int& height, bool printWarn ) const
{
    bool        returnSuccess = true;
    int         currentDevice;
    cudaError_t err;

    err = cudaGetDevice( &currentDevice );
    if( err != cudaSuccess )
    {
        POP_CUDA_WARN( err, "Cannot get current CUDA device" );
        return true;
    }

    if( currentDevice >= _properties.size() )
    {
        POP_WARN( "CUDA device was not registered at program start" );
        return true;
    }

    const cudaDeviceProp* ptr = _properties[currentDevice];
    if( width > ptr->maxTexture2D[0] )
    {
        if( printWarn )
        {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": CUDA device " << currentDevice << std::endl
                      << "    does not support 2D array textures " << width
                      << " pixels wide." << endl;
        }
        width = ptr->maxTexture2D[0];
        returnSuccess = false;
    }
    if( height > ptr->maxTexture2D[1] )
    {
        if( returnSuccess && printWarn )
        {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": CUDA device " << currentDevice << std::endl
                      << "    does not support 2D array textures " << height
                      << " pixels high." << endl;
        }
        height = ptr->maxTexture2D[1];
        returnSuccess = false;
    }

    return returnSuccess;
}

bool device_prop_t::checkLimit_2DtexLayered( int& width, int& height, int& layers, bool printWarn ) const
{
    bool        returnSuccess = true;
    int         currentDevice;
    cudaError_t err;

    err = cudaGetDevice( &currentDevice );
    if( err != cudaSuccess )
    {
        POP_CUDA_WARN( err, "Cannot get current CUDA device" );
        return true;
    }

    if( currentDevice >= _properties.size() )
    {
        POP_WARN( "CUDA device was not registered at program start" );
        return true;
    }

    const cudaDeviceProp* ptr = _properties[currentDevice];
    if( width > ptr->maxTexture2DLayered[0] )
    {
        if( printWarn )
        {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": CUDA device " << currentDevice << std::endl
                      << "    does not support 2D array textures " << width
                      << " pixels wide." << endl;
        }
        width = ptr->maxTexture2DLayered[0];
        returnSuccess = false;
    }
    if( height > ptr->maxTexture2DLayered[1] )
    {
        if( returnSuccess && printWarn )
        {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": CUDA device " << currentDevice << std::endl
                      << "    does not support 2D array textures " << height
                      << " pixels high." << endl;
        }
        height = ptr->maxTexture2DLayered[1];
        returnSuccess = false;
    }
    if( layers > ptr->maxTexture2DLayered[2] )
    {
        if( returnSuccess && printWarn )
        {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": CUDA device " << currentDevice << std::endl
                      << "    does not support 2D array textures " << layers
                      << " pixels deep." << endl;
        }
        layers = ptr->maxTexture2DLayered[2];
        returnSuccess = false;
    }

    return returnSuccess;
}

bool device_prop_t::checkLimit_2DsurfLayered( int& width, int& height, int& layers, bool printWarn ) const
{
    bool        returnSuccess = true;
    int         currentDevice;
    cudaError_t err;

    err = cudaGetDevice( &currentDevice );
    if( err != cudaSuccess )
    {
        POP_CUDA_WARN( err, "Cannot get current CUDA device" );
        return true;
    }

    if( currentDevice >= _properties.size() )
    {
        POP_WARN( "CUDA device was not registered at program start" );
        return true;
    }

    const cudaDeviceProp* ptr = _properties[currentDevice];
    if( width > ptr->maxSurface2DLayered[0] )
    {
        if( printWarn )
        {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": CUDA device " << currentDevice << std::endl
                      << "    does not support layered 2D surfaces " << width
                      << " bytes wide." << endl;
        }
        width = ptr->maxSurface2DLayered[0];
        returnSuccess = false;
    }
    if( height > ptr->maxSurface2DLayered[1] )
    {
        if( returnSuccess && printWarn )
        {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": CUDA device " << currentDevice << std::endl
                      << "    does not support layered 2D surfaces " << height
                      << " pixels high." << endl;
        }
        height = ptr->maxSurface2DLayered[1];
        returnSuccess = false;
    }
    if( layers > ptr->maxSurface2DLayered[2] )
    {
        if( returnSuccess && printWarn )
        {
            std::cerr << __FILE__ << ":" << __LINE__
                      << ": CUDA device " << currentDevice << std::endl
                      << "    does not support layered 2D surfaces " << layers
                      << " pixels deep." << endl;
        }
        layers = ptr->maxSurface2DLayered[2];
        returnSuccess = false;
    }

    return returnSuccess;
}

}}

