#pragma once

struct int3
{
    int x {};
    int y {};
    int z {};

    int3( )       = default;

    int3( int3& ) = default;

    int3( int x_, int y_, int z_ )
        : x(x_) , y(y_) , z(z_)
    { }
};

struct int4
{
    int x {};
    int y {};
    int z {};
    int w {};

    int4( )       = default;

    int4( int4& ) = default;

    int4( int x_, int y_, int z_, int w_ )
        : x(x_) , y(y_) , z(z_) , w(w_)
    { }
};

struct float4
{
    float x {};
    float y {};
    float z {};
    float w {};

    float4( )       = default;

    float4( float4& ) = default;

    float4( float x_, float y_, float z_, float w_ )
        : x(x_) , y(y_) , z(z_) , w(w_)
    { }
};

