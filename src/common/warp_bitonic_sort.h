#pragma once

#include <cuda_runtime.h>

namespace popart {
namespace BitonicSort {

template<class T>
class Warp32
{
    T*  _array;
public:
    __device__ inline
    int sort32( T* array, int my_index )
    {
        for( int outer=0; outer<5; outer++ ) {
            for( int inner=outer; inner>=0; inner-- ) {
                my_index = shiftit( array, my_index, inner, outer+1, false );
            }
        }
        return my_index;
    }

    __device__ inline
    void sort64( T* array, int2& my_indeces )
    {
        for( int outer=0; outer<5; outer++ ) {
            for( int inner=outer; inner>=0; inner-- ) {
                my_indeces.x = shiftit( array, my_indeces.x, inner, outer+1, false );
                my_indeces.y = shiftit( array, my_indeces.y, inner, outer+1, true );
            }
        }

        if( array[my_indeces.x] < array[my_indeces.y] ) swap( my_indeces.x, my_indeces.y );

        for( int outer=0; outer<5; outer++ ) {
            for( int inner=outer; inner>=0; inner-- ) {
                my_indeces.x = shiftit( array, my_indeces.x, inner, outer+1, false );
                my_indeces.y = shiftit( array, my_indeces.y, inner, outer+1, false );
            }
        }
    }

private:
    __device__ inline
    int shiftit( T* array, const int my_index, const int shift, const int direction, const bool increasing )
    {
        const T    my_val      = _array[my_index];
        const T    other_val   = __shfl_xor( my_val, 1 << shift );
        const bool my_less     = ( my_val < other_val );
        const bool reverse     = ( threadIdx.x & ( 1 << direction ) );
        const bool id_less     = ( ( threadIdx.x & ( 1 << shift ) ) == 0 );
        const bool must_swap   = not ( my_less ^ id_less ^ reverse ^ increasing );

        return ( must_swap ? __shfl_xor( my_index, 1 << shift )
                           : my_index );
    }

    __device__ inline
    void swap( int& l, int& r )
    {
        int m = r;
        r = l;
        l = m;
    }
};
} // namespace popart
} // namespace BitonicSort

