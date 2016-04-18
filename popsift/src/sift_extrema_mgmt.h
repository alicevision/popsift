#pragma once

#include <inttypes.h>
#include <iostream>

namespace popart {

class ExtremaMgmt
{
public:
    ExtremaMgmt( );
    ExtremaMgmt( uint32_t m1 );

    void init( uint32_t m1 );

    __device__
    inline void clampCounter1() {
        _counter = _counter < _max1 ? _counter : _max1;
    }

    __host__ __device__
    inline void resetCounter() {
        _counter  = 0;
    }

    __host__ __device__
    inline int getCounter() const {
        return _counter;
    }

    __host__ __device__
    inline int getExtremaMax() const {
        return _max1;
    }

    __host__ __device__
    inline int getOrientationMax() const {
        return _max2;
    }

    __device__
    int atomicAddCounter( int ct );

private:
    uint32_t _counter;
    uint32_t _max1;    // initial max
    uint32_t _max2;    // max after finding alternative angles
                       // Lowe says it happens to 15%, I reserve floor(25%)
};

} // namespace popart
