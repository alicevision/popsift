#pragma once

#include "simd_types.h"

struct Grid
{
    int3    gridDim;    /// the 3 outer looks
    int3    blockDim;   /// the 3 inner loops
    int3    blockIdx;   /// current iteration of the outer loop
    int3    threadIdx;  /// current iteration of the inner loop

    Grid()              = default;
    Grid( const Grid& ) = default;

    void setGridDim( int x_, int y_=1, int z_=1 )
    {
        gridDim.x = x_;
        gridDim.y = y_;
        gridDim.z = z_;
    }

    void setBlockDim( int x_, int y_=1, int z_=1 )
    {
        blockDim.x = x_;
        blockDim.y = y_;
        blockDim.z = z_;
    }

    void resetThread( )
    {
        threadIdx.x = threadIdx.y = threadIdx.z = 0;
    }

    void reset( )
    {
        resetBlock();
        resetThread();
    }

    void resetBlock( )
    {
        blockIdx.x = blockIdx.y = blockIdx.z = 0;
    }

    inline bool nextBlock( ) { return nextBlockXYZ(); }

    bool nextBlockXYZ( )
    {
        if( blockIdx.x < gridDim.x )
        {
            blockIdx.x++;
            if( blockIdx.x != gridDim.x ) return true;

            blockIdx.x = 0;
            return nextBlockYZ();
        }
        return false;
    }

    bool nextBlockYZ( )
    {
        if( blockIdx.y < gridDim.y )
        {
            blockIdx.y++;
            if( blockIdx.y != gridDim.y ) return true;

            blockIdx.y = 0;
            return nextBlockZ();
        }
        return false;
    }

    bool nextBlockZ( )
    {
        if( blockIdx.z < gridDim.z )
        {
            blockIdx.z++;
            if( blockIdx.z != gridDim.z ) return true;

            blockIdx.z = 0;
        }
        return false;
    }

    void resetThreadXYZ( )
    {
        threadIdx.x = threadIdx.y = threadIdx.z = 0;
    }

    void resetThreadYZ( )
    {
        threadIdx.y = threadIdx.z = 0;
    }

    void resetThreadXY( )
    {
        threadIdx.x = threadIdx.y = 0;
    }

    void resetThreadX( )
    {
        threadIdx.x = 0;
    }

    bool nextXYZ( )
    {
        if( threadIdx.x < blockDim.x )
        {
            threadIdx.x++;
            if( threadIdx.x != blockDim.x ) return true;

            threadIdx.x = 0;
            return nextYZ();
        }
        return false;
    }

    bool nextYZ( )
    {
        if( threadIdx.y < blockDim.y )
        {
            threadIdx.y++;
            if( threadIdx.y != blockDim.y ) return true;

            threadIdx.y = 0;
            return nextZ();
        }
        return false;
    }

    bool nextZ( )
    {
        if( threadIdx.z < blockDim.z )
        {
            threadIdx.z++;
            if( threadIdx.z != blockDim.z ) return true;

            threadIdx.z = 0;
        }
        return false;
    }

    bool nextY( )
    {
        if( threadIdx.y >= blockDim.y ) return false;

        threadIdx.y++;
        if( threadIdx.y != blockDim.y ) return true;

        threadIdx.y = 0;
        return false;
    }

    bool nextX( )
    {
        if( threadIdx.x >= blockDim.x ) return false;

        threadIdx.x++;
        if( threadIdx.x != blockDim.x ) return true;

        threadIdx.x = 0;
        return false;
    }

    bool next( )
    {
        if( nextXYZ() ) return true;

        if( nextBlockXYZ() )
            return nextXYZ();
        else
            return false;
    }
};

