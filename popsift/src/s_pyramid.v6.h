/*************************************************************
 * V6: device side
 *************************************************************/

#define V6_WIDTH    128
#define V6_RANGE    4 // RANGES from 1 to 12 are possible
#define V6_GAUSS_BASE   ( GAUSS_ONE_SIDE_RANGE - V6_RANGE )
#define V6_FILTERSIZE   ( V6_RANGE + 1        + V6_RANGE )
#define V6_READ_RANGE   ( V6_RANGE + V6_WIDTH + V6_RANGE )
#define V6_LEVELS       _levels

__global__
void filter_gauss_horiz_v6( float* src_data, uint32_t src_w, uint32_t src_h,
                            float* dst_data, uint32_t dst_w, uint32_t dst_h )
{
    int32_t block_x = blockIdx.x * V6_WIDTH; // blockDim.x; <- wrong, it's 136
    int32_t block_y = blockIdx.y;            // blockDim.y; <- it's 1, trivial

    __shared__ float px[V6_READ_RANGE];

    int32_t idx     = threadIdx.x - V6_RANGE;
    int32_t src_idx = clamp( block_x + idx, src_w );
    px[threadIdx.x] = src_data[ block_y * src_w + src_idx ];
    __syncthreads();

    if( threadIdx.x >= V6_WIDTH ) return;

    float out = 0;
    #pragma unroll
    for( int i=0; i<V6_FILTERSIZE; i++ ) {
        out += px[threadIdx.x+i] * popart::d_gauss_filter[V6_GAUSS_BASE+i];
    }

    int dst_row   = block_x + threadIdx.x;
    int dst_col   = block_y;
    if( dst_col < 0 || dst_col >= dst_w ) return;
    if( dst_row < 0 || dst_row >= dst_h ) return;

    dst_data[dst_row * dst_w + dst_col ] = out;
}

__global__
void filter_gauss_horiz_v6_and_dog( float* src_data, uint32_t src_w, uint32_t src_h,
                                    float* dst_data, uint32_t dst_w, uint32_t dst_h,
                                    float* higher_level_data,
                                    float* dog_data )
{
    int32_t block_x = blockIdx.x * V6_WIDTH;
    int32_t block_y = blockIdx.y;

    __shared__ float px[V6_READ_RANGE];

    int32_t idx     = threadIdx.x - V6_RANGE;
    int32_t src_idx = clamp( block_x + idx, src_w );
    px[threadIdx.x] = src_data[ block_y * src_w + src_idx ];
    __syncthreads();

    if( threadIdx.x >= V6_WIDTH ) return;

    float out = 0;
    #pragma unroll
    for( int i=0; i<V6_FILTERSIZE; i++ ) {
        out += px[threadIdx.x+i] * popart::d_gauss_filter[V6_GAUSS_BASE+i];
    }

    int dst_row   = block_x + threadIdx.x;
    int dst_col   = block_y;
    if( dst_col < 0 || dst_col >= dst_w ) return;
    if( dst_row < 0 || dst_row >= dst_h ) return;

    dst_data[dst_row * dst_w + dst_col] = out;

    float cmp;
    cmp = higher_level_data[dst_row * dst_w + dst_col];
    out -= cmp;
    out = fabs(out);
    dog_data[dst_row * dst_w + dst_col] = out;
}

__global__
void filter_gauss_horiz_v6_by_2( float* src_data, uint32_t src_w, uint32_t src_h,
                                 float* dst_data, uint32_t dst_w, uint32_t dst_h )
{
    if( threadIdx.x >= V6_READ_RANGE ) return;
    int32_t block_x = blockIdx.x * V6_WIDTH;
    int32_t block_y = blockIdx.y;

    __shared__ float px[V6_READ_RANGE];

    int32_t idx     = threadIdx.x - V6_RANGE;
    int32_t src_idx = clamp( 2*(block_x + idx), src_w );
    int32_t src_y   = clamp( 2*block_y, src_h );
    float value     = src_data[ src_y * src_w + src_idx ];
    px[threadIdx.x] = value;
    __syncthreads();

    if( threadIdx.x >= V6_WIDTH ) return;

    float out = 0;
    #pragma unroll
    for( int i=0; i<V6_FILTERSIZE; i++ ) {
        out += px[threadIdx.x+i] * popart::d_gauss_filter[V6_GAUSS_BASE+i];
    }

    int dst_row   = block_x + threadIdx.x;
    int dst_col   = block_y;
    if( dst_col < 0 || dst_col >= dst_w ) return;
    if( dst_row < 0 || dst_row >= dst_h ) return;

    dst_data[dst_row * dst_w + dst_col ] = out;
}

/*************************************************************
 * V6: host side
 *************************************************************/
__host__
void Pyramid::build_v6( Image* base )
{
#if 0
    cerr << "Entering " << __FUNCTION__ << " with base image "  << endl
         << "    type size         : " << base->type_size << endl
         << "    aligned byte size : " << base->a_width << "x" << base->a_height << endl
         << "    pitch size        : " << base->pitch << "x" << base->a_height << endl
         << "    original byte size: " << base->u_width << "x" << base->u_height << endl
         << "    aligned pix size  : " << base->a_width/base->type_size << "x" << base->a_height << endl
         << "    original pix size : " << base->u_width/base->type_size << "x" << base->u_height << endl;

#else
    cerr << "Entering " << __FUNCTION__ << " with base image "  << endl;
#endif

    _keep_time_pyramid_v6.start();

    dim3 block;
    block.x = V6_READ_RANGE;

    for( int octave=0; octave<_num_octaves; octave++ ) {
        dim3 grid_t;
        dim3 grid;
        grid_t.x  = _octaves[octave].getPitch() / V6_WIDTH;
        grid_t.y  = _octaves[octave].getHeight();
        grid.x    = _octaves[octave].getTransposedPitch() / V6_WIDTH;
        grid.y    = _octaves[octave].getTransposedHeight();

#if 0
        cerr << "Configuration for octave " << octave << endl
             << "  Normal-to-transposed: layer size: "
             << _octaves[octave].getWidth() << "x" << _octaves[octave].getHeight() << endl
             << "                        grid: "
             << "(" << grid_t.x << "," << grid_t.y << "," << grid_t.z << ")"
             << " block: "
             << "(" << block.x << "," << block.y << "," << block.z << ")" << endl
             << "  Transposed-to-normal: layer size: "
             << _octaves[octave].getTransposedPitch() << "x" << _octaves[octave].getTransposedHeight() << endl
             << "                        grid: "
             << "(" << grid.x << "," << grid.y << "," << grid.z << ")"
             << " block: "
             << "(" << block.x << "," << block.y << "," << block.z << ")" << endl;
#endif

        for( int level=0; level<V6_LEVELS; level++ ) {

            if( level == 0 ) {
                if( octave == 0 ) {
                    filter_gauss_horiz_v6
                        <<<grid_t,block,0,_stream>>>
                        ( base->array.data,
                          base->array.step / sizeof(float),
                          base->array.getRows(),
                          _octaves[octave].getTransposedData(),
                          _octaves[octave].getTransposedPitch(),
                          _octaves[octave].getTransposedHeight() );
                } else {
                    filter_gauss_horiz_v6_by_2
                        <<<grid_t,block,0,_stream>>>
                        ( _octaves[octave-1].getData( V6_LEVELS-1 ),
                          _octaves[octave-1].getPitch(),
                          _octaves[octave-1].getHeight(),
                          _octaves[octave].getTransposedData(),
                          _octaves[octave].getTransposedPitch(),
                          _octaves[octave].getTransposedHeight() );
                }
            } else {
                filter_gauss_horiz_v6
                    <<<grid_t,block,0,_stream>>>
                    ( _octaves[octave].getData( level-1 ),
                      _octaves[octave].getPitch(),
                      _octaves[octave].getHeight(),
                      _octaves[octave].getTransposedData( level ),
                      _octaves[octave].getTransposedPitch(),
                      _octaves[octave].getTransposedHeight() );
            }
            cudaError_t err = cudaGetLastError();
            POP_CUDA_FATAL_TEST( err, "filter_gauss_horiz_v6 failed: " );

            if( level == 0 ) {
                filter_gauss_horiz_v6
                    <<<grid,block,0,_stream>>>
                    ( _octaves[octave].getTransposedData( level ),
                      _octaves[octave].getTransposedPitch(),
                      _octaves[octave].getTransposedHeight(),
                      _octaves[octave].getData( level ),
                      _octaves[octave].getPitch(),
                      _octaves[octave].getHeight() );
            } else {
                assert( _octaves[octave].getDogData() );
                filter_gauss_horiz_v6_and_dog
                    <<<grid,block,0,_stream>>>
                    ( _octaves[octave].getTransposedData( level ),
                      _octaves[octave].getTransposedPitch(),
                      _octaves[octave].getTransposedHeight(),
                      _octaves[octave].getData( level ),
                      _octaves[octave].getPitch(),
                      _octaves[octave].getHeight(),
                      _octaves[octave].getData( level-1 ),
                      _octaves[octave].getDogData( level-1 ) );
            }
            err = cudaGetLastError();
            POP_CUDA_FATAL_TEST( err, "filter_gauss_horiz_v6 failed: " );
        }
    }
    _keep_time_pyramid_v6.stop();
}

// #undef V6_WIDTH
#undef V6_RANGE
#undef V6_GAUSS_BASE
#undef V6_FILTERSIZE
#undef V6_READ_RANGE
#undef V6_LEVELS

