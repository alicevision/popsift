#pragma once

#include <iostream>

#include "s_image.h"
#include "sift_conf.h"
#include "sift_extremum.h"
#include "sift_extrema_mgmt.h"
#include "sift_octave.h"

#ifndef INF
#define INF               (1<<29)
#endif
#ifndef NINF
#define NINF              (-INF)
#endif
#ifndef M_PI
#define M_PI  3.1415926535897932384626433832F
#endif
#ifndef M_PI2
#define M_PI2 (2.0F * M_PI)
#endif

#define PREALLOC_DESC
#define USE_DYNAMIC_PARALLELISM

namespace popart {

class Pyramid
{
    uint32_t     _num_octaves;
    uint32_t     _levels;
    Octave*      _octaves;

    Config::ScalingMode _scaling_mode;

public:
    Pyramid( Image* base,
             int    octaves,
             int    levels,
             int    w,
             int    h,
             Config::ScalingMode scaling_mode );
    ~Pyramid( );

    void build( Image* base );

    void find_extrema( float edgeLimit, float threshold );

    void download_and_save_array( const char* basename, uint32_t octave, uint32_t level );

    void download_descriptors( uint32_t octave );
    void save_descriptors( const char* basename, uint32_t octave, int downscale_factor );


    inline int getNumOctaves() const { return _num_octaves; }
    inline int getNumLevels()  const { return _levels; }

private:
    void build_v11 ( Image* base );

    inline void horiz_from_upscaled_orig_tex( cudaTextureObject_t src_data, int octave );
    inline void downscale_from_prev_octave( int octave, int level );
    inline void downscale_from_prev_octave_and_horiz_blur( int octave, int level );
    inline void horiz_from_prev_level( int octave, int level );
    inline void vert_from_interm( int octave, int level );
    inline void dog_from_blurred( int octave, int level );

    void find_extrema_v6( float edgeLimit, float threshold );

    template<int HEIGHT>
    void find_extrema_v6_sub( float edgeLimit, float threshold );


    void orientation_v1( );

    void descriptors_v1( );

    void test_last_error( int line );
    void debug_out_floats  ( float* data, uint32_t pitch, uint32_t height );
    void debug_out_floats_t( float* data, uint32_t pitch, uint32_t height );
};

} // namespace popart
