#include "keep_time.hpp"
#include <stdio.h>
#include "debug_macros.hpp"

// #include <assert.h>

using namespace std;

namespace popart {

KeepTime::KeepTime( cudaStream_t s )
    : _stream( s )
{
    cudaEventCreate( &_start );
    cudaEventCreate( &_stop );
}

KeepTime::~KeepTime( )
{
    cudaEvent_t ev;
    cudaError_t err;
    while( !_other_events.empty() ) {
        ev = _other_events.front();
        _other_events.pop_front();
        err = cudaEventSynchronize( ev );
        POP_CUDA_FATAL_TEST( err, "Couldn't wait for other event in ~KeepTime: " );
        err = cudaEventDestroy( ev );
        POP_CUDA_FATAL_TEST( err, "Couldn't destroy other event in ~KeepTime: " );
    }

    cudaEventDestroy( _start );
    cudaEventDestroy( _stop );
}

void KeepTime::start()
{
    cudaEventRecord( _start, _stream );
}

void KeepTime::stop( )
{
    cudaEventRecord( _stop, _stream );
}

void KeepTime::report( const char* msg )
{
    cudaEventSynchronize( _stop );
    float diff;
    cudaEventElapsedTime( &diff, _start, _stop );
    printf("%s %f ms\n", msg, diff );
}

void KeepTime::waitFor( cudaStream_t otherStream )
{
    cudaEvent_t ev;
    cudaError_t err = cudaEventCreate( &ev );
    POP_CUDA_FATAL_TEST( err, "Couldn't create sync event in KeepTime: " );

    _other_events.push_back( ev );

    err = cudaEventRecord( ev, otherStream );
    POP_CUDA_FATAL_TEST( err, "Couldn't insert event into other stream in KeepTime: " );

    err = cudaStreamWaitEvent( _stream, ev, 0 );
    POP_CUDA_FATAL_TEST( err, "Couldn't synchronize on event from other stream in KeepTime: " );

}

} // namespace popart

