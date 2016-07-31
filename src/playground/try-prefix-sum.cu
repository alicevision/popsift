#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

using namespace std;

// #define COUNT 205
// #define COUNT 2050
// #define COUNT 20
#define COUNT 3*1024

__device__ int hist[COUNT];
__device__ int psum[COUNT];
__device__ int total;

struct ReadHist
{
    inline __device__
    int operator()( int x ) const {
        return hist[x];
    }
};

struct WriteSum
{
    inline __device__
    int& operator()( int x ) {
        return psum[x];
    }
};

struct WriteTotal
{
    inline __device__
    int& operator()( ) {
        return total;
    }
};

template<class Reader, class Writer, class Total>
class ExclusivePrefixSum
{
    const Reader& _reader;
    Writer&       _writer;
    Total&        _total_writer;
    const int     _num;
public:
    __device__
    ExclusivePrefixSum( int num, const Reader& reader, Writer& writer, Total& total_writer )
        : _num( num )
        , _reader( reader )
        , _writer( writer )
        , _total_writer( total_writer )
    { }

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
                printf("%d %d : %d %d\n", threadIdx.y, threadIdx.x, ibs, loop_total );
            }
            __syncthreads();
        }

        if( threadIdx.y == 0 && threadIdx.x == 31 ) {
            _total_writer() = loop_total;
        }
    }
};

__global__
void init( int num )
{
    for( int i=threadIdx.x; i<num; i+=blockDim.x ) {
        hist[i] = 1;
        psum[i] = 0;
    }
}

__global__
void step1( int num )
{
    ReadHist reader;
    WriteSum writer;
    WriteTotal total;
    ExclusivePrefixSum<ReadHist,WriteSum,WriteTotal> e( num, reader, writer, total );
    e.sum( );
}

__global__
void printme( int num )
{
    printf("Output\n");
    for( int i=0; i<num; i++ ) {
        printf("%d ", psum[i]);
    }
    printf("\n");
    printf("Total: %d\n", total);
    __syncthreads();
}

int main()
{
    dim3 block;
    dim3 grid(1);

    block.x = 1024;
    init<<<grid,block>>>( COUNT );
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if( err != cudaSuccess ) {
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                  << "    called from " << __FILE__ << ":" << __LINE__ << std::endl
                  << "    cudaGetLastError failed: " << cudaGetErrorString(err) << std::endl;
    }

    block.x = 32;
    block.y = 32;
    step1<<<grid,block>>>( COUNT );
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if( err != cudaSuccess ) {
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                  << "    called from " << __FILE__ << ":" << __LINE__ << std::endl
                  << "    cudaGetLastError failed: " << cudaGetErrorString(err) << std::endl;
    }

    block.x = 1;
    block.y = 1;
    printme<<<grid,block>>>( COUNT );
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if( err != cudaSuccess ) {
        std::cerr << __FILE__ << ":" << __LINE__ << std::endl
                  << "    called from " << __FILE__ << ":" << __LINE__ << std::endl
                  << "    cudaGetLastError failed: " << cudaGetErrorString(err) << std::endl;
    }
}

