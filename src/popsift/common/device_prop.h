/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cuda_runtime.h>
#include <vector>

namespace popsift {
namespace cuda {

/**
 * @brief A class to recover, query and print the information about the cuda device.
 */
class device_prop_t
{
    int _num_devices;
    std::vector<cudaDeviceProp*> _properties;

public:
    enum {
        do_warn = true,
        dont_warn = false
    };

public:
    device_prop_t( );
    ~device_prop_t( );

    /**
     * @brief Print the information about the device.
     */
    void print( );

    /**
     * @brief Set the device to use.
     * @param[in] n The index of the device to use.
     * @param[in] print_choice Whether to print information about the chosen device.
     */
    void set( int n, bool print_choice = false );

    /**
     * @brief Check if a request exceeds the current CUDA device's limit in
     *  texture2Dlinear dimensions. texture2Dlinear is based on CUDA memory that
     *  can be accessed directly (i.e. no CudaArray).
     * @param[in,out] width  Desired width of the texture.
     * @param[in,out] height Desired height of the texture.
     * @param[in]     printWarn if true, print warnings to cerr if desired width
     *                          or height exceeds limits.
     * @return   \p true if the desired width and height are possible.
     *           \p false if one or both of the desired width and height are impossible.
     *           The desired width or height (or both) are replaced by the limit.
     */
    bool checkLimit_2DtexLinear( int& width, int& height, bool printWarn ) const;

    /**
     * @brief Check if a request exceeds the current CUDA device's limit in
     *  texture2D dimensions. texture2D is based on CUDA Arrays, which have
     *  invisible layout and can only be filled with cudaMemcpy.
     * @param[in,out] width  Desired width of the texture.
     * @param[in,out] height Desired height of the texture.
     * @param[in]     printWarn if true, print warnings to cerr if desired width
     *                          or height exceeds limits.
     * @return   \p true if the desired width and height are possible.
     *           \p false if one or both of the desired width and height are impossible.
     *           The desired width or height (or both) are replaced by the limit.
     */
    bool checkLimit_2DtexArray( int& width, int& height, bool printWarn ) const;

    /**
     * @brief Check if a request exceeds the current CUDA device's limit in
     *  texture2DLayered dimensions. texture2DLayered refers to a 3D structure, where
     *  interpolation happens only in 3D, effectively creating layers.
     * @param[in,out] width  Desired width of the texture.
     * @param[in,out] height Desired height of the texture.
     * @param[in,out] layers Desired depth of the texture.
     * @param[in]     printWarn if true, print warnings to cerr if desired width
     *                          or height exceeds limits.
     * @return   \p true if the desired width, height and depth are possible.
     *           \p false if one or both of the desired width and height are impossible.
     *           The desired width, height and layers are replaced by the limit
     *           if they exceed it.
     */
    bool checkLimit_2DtexLayered( int& width, int& height, int& layers,
                                  bool printWarn ) const;

    /**
     * @brief Check if a request exceeds the current CUDA device's limit in
     *  surface2DLayered dimensions. surface2DLayered is the writable equivalent
     *  to texture2DLayered, but the width must be given in bytes, not elements.
     *  Since we use float, images cannot be as wide as expected.
     * @param[in,out] width  Desired width of the texture.
     * @param[in,out] height Desired height of the texture.
     * @param[in,out] layers Desired depth of the texture.
     * @param[in]     printWarn if true, print warnings to cerr if desired width
     *                          or height exceeds limits.
     * @return   \p true if the desired width, height and depth are possible.
     *           \p false if one or both of the desired width and height are impossible.
     *           The desired width, height and layers are replaced by the limit
     *           if they exceed it.
     */
    bool checkLimit_2DsurfLayered( int& width, int& height, int& layers,
                                   bool printWarn ) const;
};

}}

