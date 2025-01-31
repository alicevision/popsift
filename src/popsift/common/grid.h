#pragma once

struct int3
{
    int x = 0;
    int y = 0;
    int z = 0;

    int3( )       = default;

    int3( int3& ) = default;

    int3( int x_, int y_, int z_ )
        : x(x_)
        : y(y_)
        : z(z_)
    { }
};

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

    bool nextBlock( )
    {
        if( blockIdx.x < gridDim.x )
        {
            blockIdx.x++;
            if( blockIdx.x != gridDim.x ) return true;

            blockIdx.x = 0;
            if( blockIdx.y < gridDim.y )
            {
                blockIdx.y++;
                if( blockIdx.y != gridDim.y ) return true;

                blockIdx.y = 0;
                if( blockIdx.z < gridDim.z )
                {
                    blockIdx.z++;
                    if( blockIdx.z != gridDim.z ) return true;

                    blockIdx.z = 0;
                }
            }
        }
        return false;
    }

    void resetThreadYZ( )
    {
        threadIdx.y = threadIdx.z = 0;
    }

    bool nextYZ( )
    {
        if( threadIdx.y < blockIdx.y )
        {
            threadIdx.y++;
            if( threadIdx.y != blockDim.y ) return true;

            threadIdx.y = 0;
            if( threadIdx.z < blockDim.z )
            {
                threadIdx.z++;
                if( threadIdx.z != blockDim.z ) return true;

                threadIdx.z = 0;
            }
        }
        return false;
    }

    void resetThreadX( )
    {
        threadIdx.x = 0;
    }

    bool nextX( )
    {
        if( threadIdx.x < blockDim.x )
        {
            threadIdx.x++;
            if( threadIdx.x != blockDim.x ) return true;

            threadIdx.x = 0;
        }
        return false;
    }
};

