#include "sift_extrema_mgmt.h"

namespace popart
{

void ExtremaMgmt::init( uint32_t m1 )
{
    _counter = 0;
    _max1    = m1;
    _max2    = m1 + m1/4;
}

ExtremaMgmt::ExtremaMgmt( )
{ }

ExtremaMgmt::ExtremaMgmt( uint32_t m1 )
{
    _counter = 0;
    _max1    = m1;
    _max2    = m1 + m1/4;
}

__device__
int ExtremaMgmt::atomicAddCounter( int ct )
{
    int idx = atomicAdd( &_counter, ct );
    return idx;
}


} // namespace iopart

