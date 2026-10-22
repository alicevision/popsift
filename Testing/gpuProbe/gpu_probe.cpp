/*
 * Tells the tests whether a GPU is usable. Prints the devices and exits
 * with 0 if there is one, and exits with 77, the SKIP_RETURN_CODE of the
 * GPU tests, if there is none. PopSift's device_prop_t is not used because
 * it terminates the program when there is no device.
 */
#include <popsift/cuda_to_hip.h>

#include <cstdio>

int main( )
{
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount( &count );
    if( err != cudaSuccess || count == 0 )
    {
        printf( "No usable GPU: %s\n",
                err != cudaSuccess ? cudaGetErrorString( err ) : "no devices found" );
        return 77;
    }
    for( int i = 0; i < count; i++ )
    {
        cudaDeviceProp prop;
        if( cudaGetDeviceProperties( &prop, i ) == cudaSuccess )
        {
            printf( "GPU %d: %s, compute capability %d.%d\n",
                    i, prop.name, prop.major, prop.minor );
        }
    }
    return 0;
}
