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
#include <typeinfo>

namespace ExclusivePrefixSum
{
class IgnoreTotal
{
    __device__ inline
    void set( int , int ) { }
};

class IgnoreWriteMapping
{
public:
    __device__ inline
    void set( int , int , int ) { }
};

template<class Reader,
         class Writer,
         class Total = IgnoreTotal,
         class WriteMapping = IgnoreWriteMapping>
class Block
{
    const Reader& _reader;
    Writer&       _writer;
    Total&        _total_writer;
    WriteMapping& _mapping_writer;
    const int     _num;
public:
    /* Instantiate an object of this type.
     * This ExclusivePrefixSum works correctly exclusively in a configuration with
     * block=(32,32,1) and grid=(1,1,1).
     * The parameter num can be any number, but only small numbers up to a few
     * thousand make sense. The template is not intended for large sets.
     * The template classes Reader, Writer and Total must provide ()-operators.
     *   inline __device__ is strongly recommended.
     * Reader must provide operator()(int n) that returns the input int at pos n.
     * Writer must provide operator()(int n) that returns int& for writing at pos n.
     * Total  must provide operator()() that returns int& for writing the total sum.
     */
    __device__
    Block( int num, const Reader& reader, Writer& writer, Total& total_writer, WriteMapping& mapping_writer )
        : _num( num )
        , _reader( reader )
        , _writer( writer )
        , _total_writer( total_writer )
        , _mapping_writer( mapping_writer )
    {
        sum( );
    }

private:
    /* This function computes the actual exclusive prefix summation
     */
    __device__
    void sum( )
    {
        __shared__ int sum[32];
        __shared__ int loop_total;

        if( threadIdx.x == 0 && threadIdx.y == 0 ) {
            loop_total = 0;
        }
        __syncthreads();

        const int start = threadIdx.y * blockDim.x + threadIdx.x;
        const int wrap  = blockDim.x * blockDim.y;
        const int end   = ( _num & (wrap-1) )
                        ? ( _num & ~(wrap-1) ) + wrap
                        : _num;

        for( int x=start; x<end; x+=wrap ) {
            __syncthreads();

            const bool valid = ( x < _num );
            const int  cell  = min( x, _num-1 );

            int ews = 0; // exclusive warp prefix sum
            int self = (valid) ? _reader.get(cell) : 0;

            // This loop is an exclusive prefix sum for one warp
            for( int s=0; s<5; s++ ) {
                const int add = popsift::shuffle_up( ews+self, 1<<s );
                ews += threadIdx.x < (1<<s) ? 0 : add;
            }

            if( threadIdx.x == 31 ) {
                // store inclusive warp prefix sum in shared mem
                // to be summed up in next phase
                sum[threadIdx.y] = ews + self;
            }
            __syncthreads();

            int ibs; // inclusive block prefix sum
            if( threadIdx.y == 0 ) {
                int ebs = 0; // exclusive block prefix sum
                int self = sum[threadIdx.x];

                for( int s=0; s<5; s++ ) {
                    const int add = popsift::shuffle_up( ebs+self, 1<<s );
                    ebs += threadIdx.x < (1<<s) ? 0 : add;
                }

                sum[threadIdx.x] = ebs;
                ibs = ebs + self;
            }
            __syncthreads();

            if( valid ) {
                const int ebs = loop_total + sum[threadIdx.y] + ews;

                /* Conceptually: at index cell of the _writer,
                 * store the exclusive prefix sum ebs.
                 */
                _writer.set( cell, ebs );

                /* Conceptually: at index ebs of the _mapping_writer,
                 * and the self-1 indices after it, store the position
                 * cell within the original array, _reader.
                 */
                _mapping_writer.set( ebs, self, cell );
            }
            __syncthreads();

            if( threadIdx.y == 0 && threadIdx.x == 31 ) {
                loop_total += ibs;
            }
            __syncthreads();
        }

        _total_writer.set( loop_total );
    }
};

} // namespace ExclusivePrefixSum

