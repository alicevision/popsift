#pragma once

#include <cuda_runtime.h>

template<class Reader, class Writer, class Total>
class ExclusivePrefixSum
{
    const Reader& _reader;
    Writer&       _writer;
    Total&        _total_writer;
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
    ExclusivePrefixSum( int num, const Reader& reader, Writer& writer, Total& total_writer )
        : _num( num )
        , _reader( reader )
        , _writer( writer )
        , _total_writer( total_writer )
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
        __syncthreads();

        for( int x=start; x<end; x+=wrap ) {
            const int cell = min( x, _num-1 );

            int ews = 0; // exclusive warp prefix sum
            int self = (x<_num) ? _reader(cell) : 0;

            for( int s=0; s<5; s++ ) {
                const int add = __shfl_up( ews+self, 1<<s );
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
                    const int add = __shfl_up( ebs+self, 1<<s );
                    ebs += threadIdx.x < (1<<s) ? 0 : add;
                }

                sum[threadIdx.x] = ebs;
                ibs = ebs + self;
            }
            __syncthreads();

            if( x<_num ) {
                _writer(cell) = loop_total + sum[threadIdx.y] + ews;
            }

            if( threadIdx.y == 0 && threadIdx.x == 31 ) {
                loop_total += ibs;
            }
            __syncthreads();
        }

        if( threadIdx.y == 0 && threadIdx.x == 31 ) {
            _total_writer() = loop_total;
        }
    }
};

