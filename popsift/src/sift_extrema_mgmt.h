#pragma once

#include <inttypes.h>
#include <iostream>

namespace popart {

extern int h_max_extrema;
extern int h_max_orientations;
extern __device__ __constant__ int d_max_extrema;
extern __device__ __constant__ int d_max_orientations;

class ExtremaMgmt
{
public:
    static void init( int max_extrema );

    void reset( );

    __device__
    inline void clampCounter1() {
        _counter = _counter < d_max_extrema ? _counter : d_max_extrema;
    }

    __host__ __device__
    inline void resetCounter() {
        _counter  = 0;
    }

    __host__ __device__
    inline int getCounter() const {
        return _counter;
    }

    __device__
    int atomicAddCounter( int ct );

private:
    int _counter;
};

} // namespace popart
