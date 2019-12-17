#include <iostream>
#include <vector>
#include <algorithm>
#include "popsift/regression/test_radix_sort.h"
#include "popsift/common/device_prop.h"

std::vector<int> the_list(64);
int buffer[64];

int main()
{
    std::cout << "To test a specific NVIDIA card in your system:" << std::endl
              << "export NVIDIA_VISIBLE_DEVICES=1" << std::endl
              << "export CUDA_VISIBLE_DEVICES=<int>" << std::endl
              << std::endl;

    popsift::cuda::device_prop_t deviceInfo;
    deviceInfo.set( 0, true );
    deviceInfo.print( );

    for( int i=0; i<64; i++ ) the_list[i] = 100-i;

    for( int i=0; i<500; i++ )
        std::next_permutation( the_list.begin(), the_list.end() );
    std::reverse( the_list.begin(), the_list.end() );

    for( int i=0; i<64; i++ )
    {
        buffer[i] = the_list[i];
        std::cout << buffer[i] << " ";
    }
    std::cout << std::endl;

    TestRadix::push( buffer );

    TestRadix::callSort();

    TestRadix::pull( buffer );

    for( int i=0; i<64; i++ )
    {
        std::cout << buffer[i] << " ";
    }
    std::cout << std::endl;
}

