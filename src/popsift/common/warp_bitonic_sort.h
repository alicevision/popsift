/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "assist.h"

#include <cuda_runtime.h>

namespace popsift {
namespace BitonicSort {

template<class T>
class Warp32
{
    T*  _array;
public:
    __device__ inline
    Warp32( T* array ) : _array( array ) { }

    __device__ inline
    int sort32( int my_index )
    {
        for( int outer=0; outer<5; outer++ ) {
            for( int inner=outer; inner>=0; inner-- ) {
                my_index = shiftit( my_index, inner, outer+1, false );
            }
        }
        return my_index;
    }

    __device__ inline
    void sort64( int2& my_indeces )
    {
        for( int outer=0; outer<5; outer++ ) {
            for( int inner=outer; inner>=0; inner-- ) {
                my_indeces.x = shiftit( my_indeces.x, inner, outer+1, false );
                my_indeces.y = shiftit( my_indeces.y, inner, outer+1, true );
            }
        }

        if( _array[my_indeces.x] < _array[my_indeces.y] ) swap( my_indeces.x, my_indeces.y );

        for( int outer=0; outer<5; outer++ ) {
            for( int inner=outer; inner>=0; inner-- ) {
                my_indeces.x = shiftit( my_indeces.x, inner, outer+1, false );
                my_indeces.y = shiftit( my_indeces.y, inner, outer+1, false );
            }
        }
    }

private:
    __device__ inline
    int shiftit( const int my_index, const int shift, const int direction, const bool increasing )
    {
        const T    my_val      = _array[my_index];
        const T    other_val   = popsift::shuffle_xor( my_val, 1 << shift );
        const bool reverse     = ( threadIdx.x & ( 1 << direction ) );
        const bool id_less     = ( ( threadIdx.x & ( 1 << shift ) ) == 0 );
        const bool my_more     = id_less ? ( my_val > other_val )
                                         : ( my_val < other_val );
        const bool must_swap   = ! ( my_more ^ reverse ^ increasing );

        int lane = must_swap ? ( 1 << shift ) : 0;
        return popsift::shuffle_xor( my_index, lane );
    }

    __device__ inline
    void swap( int& l, int& r )
    {
        int m = r;
        r = l;
        l = m;
    }
};
} // namespace popsift
} // namespace BitonicSort

