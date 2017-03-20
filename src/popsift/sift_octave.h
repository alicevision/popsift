/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <iostream>
#include <vector>

#include "s_image.h"
#include "sift_conf.h"
#include "sift_extremum.h"
#include "sift_constants.h"

namespace popsift {

class Octave
{
        int _w;
        int _h;
        int _max_w;
        int _max_h;
        int _debug_octave_id;
        int _levels;
        int _gauss_group;

        cudaArray_t           _data;
        cudaChannelFormatDesc _data_desc;
        cudaExtent            _data_ext;
        cudaSurfaceObject_t   _data_surf;
        cudaTextureObject_t   _data_tex_point;
        cudaTextureObject_t   _data_tex_linear;

        cudaArray_t           _interm_array;
        cudaChannelFormatDesc _interm_desc;
        cudaSurfaceObject_t   _interm_surf;
        cudaTextureObject_t   _interm_data_tex_point;
        cudaTextureObject_t   _interm_data_tex_linear;

        cudaArray_t           _dog_3d;
        cudaChannelFormatDesc _dog_3d_desc;
        cudaExtent            _dog_3d_ext;
        cudaSurfaceObject_t   _dog_3d_surf;
        cudaTextureObject_t   _dog_3d_tex_point;
        cudaTextureObject_t   _dog_3d_tex_linear;

        // one CUDA stream per level
        // consider whether some of them can be removed
        cudaStream_t _stream;
        cudaEvent_t  _scale_done;
        cudaEvent_t  _extrema_done;
        cudaEvent_t  _ori_done;
        cudaEvent_t  _desc_done;

    public:
        Octave( );
        ~Octave( ) { this->free(); }

        void resetDimensions( int w, int h );

        inline void debugSetOctave( uint32_t o ) { _debug_octave_id = o; }

        inline int getLevels() const { return _levels; }
        inline int getWidth() const  {
            return _w;
        }
        inline int getHeight() const {
            return _h;
        }

        inline cudaStream_t getStream( ) {
            return _stream;
        }
        inline cudaEvent_t getEventScaleDone( ) {
            return _scale_done;
        }
        inline cudaEvent_t getEventExtremaDone( ) {
            return _extrema_done;
        }
        inline cudaEvent_t getEventOriDone( ) {
            return _ori_done;
        }
        inline cudaEvent_t getEventDescDone( ) {
            return _desc_done;
        }

        inline cudaTextureObject_t getIntermDataTexLinear( ) {
            return _interm_data_tex_linear;
        }
        inline cudaTextureObject_t getIntermDataTexPoint( ) {
            return _interm_data_tex_point;
        }
        inline cudaTextureObject_t getDataTexLinear( ) {
            return _data_tex_linear;
        }
        inline cudaTextureObject_t getDataTexPoint( ) {
            return _data_tex_point;
        }
        inline cudaSurfaceObject_t getDataSurface( ) {
            return _data_surf;
        }
        inline cudaSurfaceObject_t getIntermediateSurface( ) {
            return _interm_surf;
        }
        
        inline cudaSurfaceObject_t& getDogSurface( ) {
            return _dog_3d_surf;
        }
        inline cudaTextureObject_t& getDogTexturePoint( ) {
            return _dog_3d_tex_point;
        }
        inline cudaTextureObject_t& getDogTextureLinear( ) {
            return _dog_3d_tex_linear;
        }

        /**
         * alloc() - allocates all GPU memories for one octave
         * @param width in floats, not bytes!!!
         */
        void alloc( int width,
                    int height,
                    int levels,
                    int gauss_group );
        void free();

        /**
         * debug:
         * download a level and write to disk
         */
         void download_and_save_array( const char* basename, int octave );

private:
    void alloc_data_planes( );
    void alloc_data_tex( );
    void alloc_interm_array( );
    void alloc_interm_tex( );
    void alloc_dog_array( );
    void alloc_dog_tex( );
    void alloc_streams( );
    void alloc_events( );

    void free_events( );
    void free_streams( );
    void free_dog_tex( );
    void free_dog_array( );
    void free_interm_tex( );
    void free_interm_array( );
    void free_data_tex( );
    void free_data_planes( );
};

} // namespace popsift
