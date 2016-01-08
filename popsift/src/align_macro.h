#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdio.h> // for printf

#undef NOISY

#if 0
inline size_t align_padding( uint32_t requested, size_t alignment )
{
    // assert( alignment > 0 );
    assert( __builtin_popcount(alignment) == 1 ); // alignment must be a power of 2

    size_t intermediate = (size_t)requested & ( alignment - 1 ); // all set bits below alignment bit

    uint32_t padding = ( alignment - intermediate ) & ~alignment; // inverse of intermediate
    return (size_t)padding;
}
#endif

namespace popart {

inline void align( size_t& requested, size_t alignment )
{
    assert( __builtin_popcount(alignment) == 1 ); // alignment must be a power of 2

    size_t intermediate = requested & ~(alignment-1); // remove all bits below alignment

    if( intermediate == requested ) return; // same means it is already aligned

    requested = intermediate + alignment; // next higher aligned value
}

inline void align( uint32_t& requested, uint32_t alignment )
{
#ifdef NOISY
    printf("align: called with value %u, alignment %u\n", requested, alignment);
#endif

    assert( __builtin_popcount(alignment) == 1 ); // alignment must be a power of 2

    uint32_t intermediate = requested & ~(alignment-1); // remove all bits below alignment

    if( intermediate == requested ) {
#ifdef NOISY
        printf("align: value is already aligned\n");
#endif
        return; // same means it is already aligned
    }

    requested = intermediate + alignment; // next higher aligned value
#ifdef NOISY
    printf("align: aligned value is %u\n", requested);
#endif
}

} // namespace popart
