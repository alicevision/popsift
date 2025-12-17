#pragma once

// Detect if we're using CUDA or HIP runtime
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
    // Use native CUDA/HIP vector types
    #ifdef __CUDACC__
        // CUDA backend
        #include <vector_types.h>
        #include <vector_functions.hpp>
    #else
        // HIP backend (AdaptiveCpp on AMD)
        #include <hip/hip_vector_types.h>
    #endif
#else
    // Neither CUDA nor HIP: define our own vector types

    struct int2
    {
        int x {};
        int y {};

        int2( )       = default;
        int2( const int2& ) = default;
        int2( int x_, int y_ ) : x(x_) , y(y_) { }
    };

    static inline
    int2 make_int2( int x, int y)
    {
        return int2( x, y);
    }

    struct int3
    {
        int x {};
        int y {};
        int z {};

        int3( )       = default;
        int3( const int3& ) = default;
        int3( int x_, int y_, int z_ ) : x(x_) , y(y_) , z(z_) { }
    };

    static inline
    int3 make_int3( int x, int y, int z )
    {
        return int3( x, y, z );
    }

    struct int4
    {
        int x {};
        int y {};
        int z {};
        int w {};

        int4( )       = default;
        int4( const int4& ) = default;
        int4( int x_, int y_, int z_, int w_ ) : x(x_) , y(y_) , z(z_) , w(w_) { }
    };

    struct float2
    {
        float x {};
        float y {};

        float2( )       = default;
        float2( const float2& ) = default;
        float2( float x_, float y_ ) : x(x_) , y(y_) { }
    };

    static inline
    float2 make_float2( float x, float y )
    {
        return float2( x, y );
    }

    struct float3
    {
        float x {};
        float y {};
        float z {};

        float3( )       = default;
        float3( const float3& ) = default;
        float3( float x_, float y_, float z_ ) : x(x_) , y(y_) , z(z_) { }
    };

    struct float4
    {
        float x {};
        float y {};
        float z {};
        float w {};

        float4( )       = default;
        float4( const float4& ) = default;
        float4( float x_, float y_, float z_, float w_ ) : x(x_) , y(y_) , z(z_) , w(w_) { }
    };

#endif // __CUDACC__ || __HIPCC__ || HIP detection
