#pragma once

struct int2
{
    int x {};
    int y {};

    int2( )       = default;

    int2( const int2& ) = default;

    int2( int x_, int y_ )
        : x(x_) , y(y_)
    { }
};

struct int3
{
    int x {};
    int y {};
    int z {};

    int3( )       = default;

    int3( const int3& ) = default;

    int3( int x_, int y_, int z_ )
        : x(x_) , y(y_) , z(z_)
    { }
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

    int4( int x_, int y_, int z_, int w_ )
        : x(x_) , y(y_) , z(z_) , w(w_)
    { }
};

struct float2
{
    float x {};
    float y {};

    float2( )       = default;

    float2( const float2& ) = default;

    float2( float x_, float y_ )
        : x(x_) , y(y_)
    { }
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

    float3( float x_, float y_, float z_ )
        : x(x_) , y(y_) , z(z_)
    { }
};

struct float4
{
    float x {};
    float y {};
    float z {};
    float w {};

    float4( )       = default;

    float4( const float4& ) = default;

    float4( float x_, float y_, float z_, float w_ )
        : x(x_) , y(y_) , z(z_) , w(w_)
    { }
};

