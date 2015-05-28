/*************************************************************
 * V7: device side
 *************************************************************/
#define V7_WIDTH    32
#define V7_RANGE    4 // RANGES from 1 to 12 are possible
#define V7_GAUSS_BASE   ( GAUSS_ONE_SIDE_RANGE - V7_RANGE )
#define V7_FILTERSIZE   ( V7_RANGE + 1        + V7_RANGE )
#define V7_READ_RANGE   ( V7_RANGE + V7_WIDTH + V7_RANGE )
#define V7_LEVELS       _levels

__device__ uint32_t non_null_dog = 0;

__global__
void filter_gauss_horiz_v7( float* src_data,
                            float* dst_data,
                            uint32_t width,
                            uint32_t pitch,
                            uint32_t height,
                            uint32_t t_pitch )
{
    int block_x = blockIdx.x * V7_WIDTH;
    int block_y = blockIdx.y;
    int idx;

    float g;
    float val;
    float out = 0;

    for( int offset = V7_RANGE; offset>0; offset-- ) {
        g  = d_gauss_filter[GAUSS_ONE_SIDE_RANGE - offset];

        idx = clamp( block_x + threadIdx.x - offset, width );
        val = src_data[ block_y * pitch + idx ];
        out += ( val * g );

        idx = clamp( block_x + threadIdx.x + offset, width );
        val = src_data[ block_y * pitch + idx ];
        out += ( val * g );
    }

    g  = d_gauss_filter[GAUSS_ONE_SIDE_RANGE];
    idx = clamp( block_x + threadIdx.x, width );
    val = src_data[ block_y * pitch + idx ];
    out += ( val * g );

    if( block_y >= t_pitch ) return;
    if( idx     >= pitch ) return;

    bool nix = ( block_x + threadIdx.x >= width ) || ( block_y >= height );
    dst_data[ block_y * pitch + idx ] = nix ? 0 : out;
}

__device__
void filter_gauss_vert_v7_sub( float*   src_data,
                               float*   dst_data,
                               uint32_t width,
                               uint32_t pitch,
                               uint32_t height )
{
    const int block_x = blockIdx.x * V7_WIDTH;
    const int block_y = blockIdx.y;
    const int idx     = block_x + threadIdx.x;
    int idy;

    if( idx >= pitch ) return;

    float g;
    float val;
    float out = 0;

    for( int offset = V7_RANGE; offset>0; offset-- ) {
        g  = d_gauss_filter[GAUSS_ONE_SIDE_RANGE - offset];

        idy = clamp( block_y - offset, height );
        val = src_data[ idy * pitch + idx ];
        out += ( val * g );

        idy = clamp( block_y + offset, height );
        val = src_data[ idy * pitch + idx ];
        out += ( val * g );
    }

    g  = d_gauss_filter[GAUSS_ONE_SIDE_RANGE];
    idy = clamp( block_y, height );
    val = src_data[ idy * pitch + idx ];
    out += ( val * g );

    if( idy >= height ) return;
    if( idx >= pitch  ) return;

    bool nix = ( idx >= width );
    dst_data[ idy * pitch + idx ] = nix ? 0 : out;
}

__global__
void filter_gauss_vert_v7( float*   src_data,
                           float*   dst_data,
                           uint32_t width,
                           uint32_t pitch,
                           uint32_t height )
{
    filter_gauss_vert_v7_sub( src_data, dst_data, width, pitch, height );
}

__global__
void filter_gauss_vert_v7_and_dog( float*   src_data,
                                   float*   dst_data,
                                   uint32_t width,
                                   uint32_t pitch,
                                   uint32_t height,
                                   float*   higher_level_data,
                                   float*   dog_data )
{
    filter_gauss_vert_v7_sub( src_data, dst_data, width, pitch, height );

    const int idx = blockIdx.x * V7_WIDTH + threadIdx.x;
    const int idy = blockIdx.y;

    if( idx >= pitch ) return;
    if( idy >= height ) return;

    bool nix = ( idx >= width );

    const int offset = idy * pitch + idx;
    float a, b;
    a = dst_data[ offset ];
    b = higher_level_data[ offset ];
    a = fabs( a - b );
    dog_data[ offset ] = nix ? 0 : a;

    if( nix == false && a > 0 ) {
        atomicAdd( &non_null_dog, 1 );
    }
}

__global__
void filter_gauss_horiz_v7_by_2( float*   src_data,
                                 float*   dst_data,
                                 uint32_t dst_width,
                                 uint32_t dst_pitch,
                                 uint32_t dst_height,
                                 uint32_t src_pitch )
{
    int block_x = blockIdx.x * V7_WIDTH;
    int block_y = blockIdx.y;
    int idx;

    float g;
    float val;
    float out = 0;

    for( int offset = V7_RANGE; offset>0; offset-- ) {
        g  = d_gauss_filter[GAUSS_ONE_SIDE_RANGE - offset];

        idx = clamp( 2 * ( block_x + threadIdx.x - offset ), src_pitch );
        val = src_data[ 2 * block_y * src_pitch + idx ];
        out += ( val * g );

        idx = clamp( 2 * ( block_x + threadIdx.x + offset ), src_pitch );
        val = src_data[ 2 * block_y * src_pitch + idx ];
        out += ( val * g );
    }

    g  = d_gauss_filter[GAUSS_ONE_SIDE_RANGE];
    idx = clamp( 2 * ( block_x + threadIdx.x ), src_pitch );
    val = src_data[ 2 * block_y * src_pitch + idx ];
    out += ( val * g );

    idx = block_x + threadIdx.x;
    if( block_y >= dst_height ) return;
    if( idx     >= dst_pitch  ) return;

    bool nix = ( idx >= dst_width );
    dst_data[ block_y * dst_pitch + idx ] = nix ? 0 : out;
}

/*************************************************************
 * V7: host side
 *************************************************************/
__host__
void Pyramid::build_v7( Image* base )
{
#if (PYRAMID_PRINT_DEBUG==1)
    cerr << "Entering " << __FUNCTION__ << " with base image "  << endl
         << "    type size         : " << base->type_size << endl
         << "    aligned byte size : " << base->a_width << "x" << base->a_height << endl
         << "    pitch size        : " << base->pitch << "x" << base->a_height << endl
         << "    original byte size: " << base->u_width << "x" << base->u_height << endl
         << "    aligned pix size  : " << base->a_width/base->type_size << "x" << base->a_height << endl
         << "    original pix size : " << base->u_width/base->type_size << "x" << base->u_height << endl;
#else
    cerr << "Entering " << __FUNCTION__ << " with base image "  << endl;
#endif // (PYRAMID_PRINT_DEBUG==1)

    cudaDeviceSynchronize();
    uint32_t value = 0;
    cudaMemcpyToSymbol( non_null_dog, &value, sizeof(uint32_t), 0, cudaMemcpyHostToDevice );
    cudaDeviceSynchronize();

#if 0
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        POP_CUDA_FATAL_TEST( err, "entering Pyramid::build_v7: " );
#endif

    _keep_time_pyramid_v7.start();

    dim3 block;
    block.x = V7_WIDTH;

    for( int octave=0; octave<_octaves; octave++ ) {
        dim3 grid_t;
        grid_t.x  = _layers[octave].getPitch()  / V7_WIDTH;
        grid_t.y  = _layers[octave].getTransposedPitch();
        // dim3 grid;
        // grid.x    = _layers[octave].getTransposedPitch() / V7_WIDTH;
        // grid.y    = _layers[octave].getTransposedHeight();

#if 0
        cerr << "Configuration for octave " << octave << endl
             << "  Normal-to-transposed: layer size: "
             << _layers[octave].getWidth() << "x" << _layers[octave].getHeight() << endl
             << "                        grid: "
             << "(" << grid_t.x << "," << grid_t.y << "," << grid_t.z << ")"
             << " block: "
             << "(" << block.x << "," << block.y << "," << block.z << ")" << endl
             << "  Transposed-to-normal: layer size: "
             << _layers[octave].getTransposedPitch() << "x" << _layers[octave].getTransposedHeight() << endl
             << "                        grid: "
             << "(" << grid.x << "," << grid.y << "," << grid.z << ")"
             << " block: "
             << "(" << block.x << "," << block.y << "," << block.z << ")" << endl;
#endif

        for( int level=0; level<V7_LEVELS; level++ ) {

            if( level == 0 ) {
                if( octave == 0 ) {
                    filter_gauss_horiz_v7
                        <<<grid_t,block,0,_stream>>>
                        ( (float*)base->array,
                          _layers[octave].getData2(),
                          base->u_width,
                          _layers[octave].getPitch(),
                          base->u_height,
                          _layers[octave].getTransposedPitch() );
                } else {
                    filter_gauss_horiz_v7_by_2
                        <<<grid_t,block,0,_stream>>>
                        ( _layers[octave-1].getData( V7_LEVELS-3 ),
                          _layers[octave].getData2(),
                          _layers[octave].getWidth(),
                          _layers[octave].getPitch(),
                          _layers[octave].getHeight(),
                          _layers[octave-1].getPitch() );
                }
            } else {
                filter_gauss_horiz_v7
                    <<<grid_t,block,0,_stream>>>
                    ( _layers[octave].getData( level-1 ),
                      _layers[octave].getData2( level ),
                      _layers[octave].getWidth(),
                      _layers[octave].getPitch(),
                      _layers[octave].getHeight(),
                      _layers[octave].getTransposedPitch() );
            }
            cudaError_t err = cudaGetLastError();
            POP_CUDA_FATAL_TEST( err, "filter_gauss_horiz_v7 failed: " );

            if( level == 0 ) {
                filter_gauss_vert_v7
                    <<<grid_t,block,0,_stream>>>
                    ( _layers[octave].getData2( level ),
                      _layers[octave].getData( level ),
                      _layers[octave].getWidth(),
                      _layers[octave].getPitch(),
                      _layers[octave].getHeight() );
            } else {
                assert( _layers[octave].getDogData() );
                filter_gauss_vert_v7_and_dog
                    <<<grid_t,block,0,_stream>>>
                    ( _layers[octave].getData2( level ),
                      _layers[octave].getData( level ),
                      _layers[octave].getWidth(),
                      _layers[octave].getPitch(),
                      _layers[octave].getHeight(),
                      _layers[octave].getData( level-1 ),
                      _layers[octave].getDogData( level-1 ) );
            }
            err = cudaGetLastError();
            POP_CUDA_FATAL_TEST( err, "filter_gauss_horiz_v7 failed: " );
        }
    }

    _keep_time_pyramid_v7.stop();

    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol( &value, non_null_dog, sizeof(uint32_t), 0, cudaMemcpyDeviceToHost );
    cerr << "The total of dog symbols written is " << value << endl;
}

// #undef V7_WIDTH
#undef V7_RANGE
#undef V7_GAUSS_BASE
#undef V7_FILTERSIZE
#undef V7_READ_RANGE
#undef V7_LEVELS

