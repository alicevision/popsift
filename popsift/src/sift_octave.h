#pragma once

#include <iostream>

#include "s_image.h"
#include "sift_conf.h"
#include "sift_extremum.h"
#include "sift_extrema_mgmt.h"

#define PREALLOC_DESC
#define USE_DYNAMIC_PARALLELISM

namespace popart {

class Octave
{
        int _w;
        int _h;
        int _debug_octave_id;
        int _levels;

        Plane2D_float* _data;
        Plane2D_float  _intermediate_data;

        cudaArray_t           _dog_3d;
        cudaChannelFormatDesc _dog_3d_desc;
        cudaExtent            _dog_3d_ext;

        cudaSurfaceObject_t   _dog_3d_surf;

        cudaTextureObject_t   _dog_3d_tex;

        // one CUDA stream per level
        // cosnider later whether some of them can be removed
        cudaStream_t* _streams;
        cudaEvent_t*  _gauss_done;

    public:
        cudaTextureObject_t* _data_tex;
        cudaTextureObject_t  _interm_data_tex;

    private:
        /* It seems strange strange to collect extrema globally only.
         * Because of the global cut-off, features from the later
         * octave have a smaller chance of being accepted.
         * Besides, the management of computing gradiants and atans
         * must be handled per scale (level) of an octave.
         * There: one set of extrema per octave and level.
         */
        ExtremaMgmt*         _h_extrema_mgmt; // host side info
        ExtremaMgmt*         _d_extrema_mgmt; // device side info
        Extremum**  _h_extrema;
        Extremum**  _d_extrema;
#if defined(PREALLOC_DESC) && defined(USE_DYNAMIC_PARALLELISM)
        int                  _max_desc_pre;
        Descriptor**         _d_desc_pre;
        Descriptor**         _h_desc_pre;
#else
        Descriptor**         _d_desc;
        Descriptor**         _h_desc;
#endif

    public:
        Octave( );
        ~Octave( ) { this->free(); }

        inline void debugSetOctave( uint32_t o ) { _debug_octave_id = o; }

        inline int getLevels() const { return _levels; }
        inline int getWidth() const  { return _data[0].getWidth(); }
        inline int getHeight() const { return _data[0].getHeight(); }

        inline cudaStream_t getStream( uint32_t level ) {
            return _streams[level];
        }
        inline cudaEvent_t getEventGaussDone( uint32_t level ) {
            return _gauss_done[level];
        }

        inline Plane2D_float& getData( uint32_t level ) {
            return _data[level];
        }
        inline Plane2D_float& getIntermediateData( ) {
            return _intermediate_data;
        }
        
        inline cudaSurfaceObject_t& getDogSurface( ) {
            return _dog_3d_surf;
        }
        inline cudaTextureObject_t& getDogTexture( ) {
            return _dog_3d_tex;
        }

        inline uint32_t getFloatSizeData() const {
            return _data[0].getByteSize() / sizeof(float);
        }
        inline uint32_t getByteSizeData() const {
            return _data[0].getByteSize();
        }
        inline uint32_t getByteSizePitch() const {
            return _data[0].getPitch();
        }

        inline ExtremaMgmt* getExtremaMgmtH( uint32_t level ) {
            return &_h_extrema_mgmt[level];
        }

        inline ExtremaMgmt* getExtremaMgmtD( ) {
            return _d_extrema_mgmt;
        }

        inline Extremum* getExtrema( uint32_t level ) {
            return _d_extrema[level];
        }

        // void resetExtremaCount( );
        void readExtremaCount( );
        int getExtremaCount( ) const;
        int getExtremaCount( uint32_t level ) const;

#if defined(PREALLOC_DESC) && defined(USE_DYNAMIC_PARALLELISM)
        int getMaxDescriptors() const { return _max_desc_pre; }
#else
        void        allocDescriptors( );
#endif
        Descriptor* getDescriptors( uint32_t level );
        void        downloadDescriptor( );
        void        writeDescriptor( std::ostream& ostr, float downsampling_factor );

        /**
         * alloc() - allocates all GPU memories for one octave
         * @param width in floats, not bytes!!!
         */
        void alloc( int width, int height, int levels, int layer_max_extrema );
        void free();

        /**
         * debug:
         * download a level and write to disk
         */
        void download_and_save_array( const char* basename, uint32_t octave, uint32_t level );
};

} // namespace popart
