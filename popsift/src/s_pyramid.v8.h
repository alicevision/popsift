/*************************************************************
 * V8: device side
 *************************************************************/
#define V8_FILTERSIZE   ( V8_RANGE + 1        + V8_RANGE )
#define V8_LEVELS       _levels

void V8_checkerr( NppStatus err, const char* file, uint32_t line )
{
    if( err != 0 ) {
        cerr << file << ":" << line << ": ";
        switch( err ) {
        case NPP_NOT_SUPPORTED_MODE_ERROR :
            cerr << "NPP_NOT_SUPPORTED_MODE_ERROR";
            break;
        case NPP_MASK_SIZE_ERROR :
            cerr << "NPP_MASK_SIZE_ERROR";
            break;
        case NPP_STEP_ERROR :
            cerr << "NPP_STEP_ERROR";
            break;
        default :
            cerr << "Something went wrong - " << err;
            break;
        }
        cerr << endl;
        exit( -1 );
    }
}

#define V8_CHECKERR( err ) V8_checkerr( err, __FILE__, __LINE__ )


/*************************************************************
 * V8: host side
 *************************************************************/
inline NppiSize mksz( uint32_t w, uint32_t h )
{
    NppiSize n;
    n.width = w;
    n.height = h;
    return n;
}

inline NppiPoint mkpt( uint32_t x, uint32_t y )
{
    NppiPoint n;
    n.x = x;
    n.y = y;
    return n;
}

__host__
void Pyramid::build_v8( Image* base )
{
    cerr << "Entering " << __FUNCTION__ << " with base image "  << endl
         << "    type size         : " << base->type_size << endl
         << "    aligned byte size : " << base->a_width << "x" << base->a_height << endl
         << "    pitch size        : " << base->pitch << "x" << base->a_height << endl
         << "    original byte size: " << base->u_width << "x" << base->u_height << endl
         << "    aligned pix size  : " << base->a_width/base->type_size << "x" << base->a_height << endl
         << "    original pix size : " << base->u_width/base->type_size << "x" << base->u_height << endl;

    KeepTime keep( _stream );
    keep.start();

    NppStatus status;
    Npp32s    srcStep;
    Npp32s    dstStep;
    NppiSize  dstSize;

    for( int octave=0; octave<_num_octaves; octave++ ) {
        for( int level=0; level<V8_LEVELS; level++ ) {
            if( level == 0 ) {
                if( octave == 0 ) {
                    cerr << __LINE__ << " call nppiFilterGaussBorder_32f_C1R" << endl;
                    dstStep       = _octaves[octave].getByteSizePitch( );
                    dstSize.width = base->pitch / sizeof(float);
                    dstSize.height = base->a_height;

                    NppiMaskSize mns = NPP_MASK_SIZE_9_X_9;
                    // NppiMaskSize mns = NPP_MASK_SIZE_3_X_3;

                    status = nppiFilterGaussBorder_32f_C1R(
                                (const Npp32f*)base->array,
                                (Npp32s)base->pitch,
                                mksz( base->u_width/sizeof(float), base->u_height/sizeof(float) ),
                                mkpt( 0, 0 ),
                                (Npp32f*)_octaves[octave].getData( level ),
                                dstStep,
                                dstSize,
                                mns,
                                NPP_BORDER_REPLICATE );
                    V8_CHECKERR( status );
                } else {
                    NppiSize srcSize;
                    NppiRect srcROI;
                    NppiRect dstROI;
                    srcSize.width  = _octaves[octave-1].getWidth();
                    srcSize.height = _octaves[octave-1].getHeight();
                    srcROI.x       = 0;
                    srcROI.y       = 0;
                    srcROI.width   = srcSize.width;
                    srcROI.height  = srcSize.height;
                    srcStep        = _octaves[octave-1].getByteSizePitch();
                    dstROI.x       = 0;
                    dstROI.y       = 0;
                    dstROI.width   = _octaves[octave].getWidth();
                    dstROI.height  = _octaves[octave].getHeight();
                    dstSize.width  = _octaves[octave].getWidth();
                    dstSize.height = _octaves[octave].getHeight();
                    dstStep        = _octaves[octave].getByteSizePitch();
                    cerr << __LINE__ << " call nppiResizeSqrPixel_32f_C1R" << endl;
                    status = nppiResizeSqrPixel_32f_C1R(
                                (const Npp32f*)_octaves[octave-1].getData( V8_LEVELS-1 ),
                                srcSize,
                                srcStep,
                                srcROI,
                                (Npp32f*)_octaves[octave].getData2(),
                                dstStep,
                                dstROI,
                                0.5, 0.5,
                                0.0, 0.0,
                                NPPI_INTER_LINEAR );
                    V8_CHECKERR( status );

                    NppiMaskSize mns = NPP_MASK_SIZE_9_X_9;
                    // NppiMaskSize mns = NPP_MASK_SIZE_3_X_3;
                    cerr << __LINE__ << " call nppiFilterGaussBorder_32f_C1R" << endl;
                    status = nppiFilterGaussBorder_32f_C1R(
                                (const Npp32f*)_octaves[octave].getData2(),
                                (Npp32s)_octaves[octave].getByteSizePitch(),
                                mksz( _octaves[octave].getWidth(),
                                      _octaves[octave].getHeight() ),
                                mkpt( 0, 0 ),
                                (Npp32f*)_octaves[octave].getData(),
                                (Npp32s)_octaves[octave].getByteSizePitch(),
                                dstSize,
                                mns,
                                NPP_BORDER_REPLICATE );
                    V8_CHECKERR( status );
                }
            } else {
                dstSize.width  = _octaves[octave].getWidth();
                dstSize.height = _octaves[octave].getHeight();
                dstStep        = _octaves[octave].getByteSizePitch();

                NppiMaskSize mns = NPP_MASK_SIZE_9_X_9;
                // NppiMaskSize mns = NPP_MASK_SIZE_3_X_3;
                cerr << __LINE__ << " call nppiFilterGaussBorder_32f_C1R" << endl;
                status = nppiFilterGaussBorder_32f_C1R(
                            (const Npp32f*)_octaves[octave].getData( level - 1 ),
                            (Npp32s)_octaves[octave].getByteSizePitch(),
                            mksz( _octaves[octave].getWidth(),
                                  _octaves[octave].getHeight() ),
                            mkpt( 0, 0 ),
                            (Npp32f*)_octaves[octave].getData( level ),
                            dstStep,
                            dstSize,
                            mns,
                            NPP_BORDER_REPLICATE );
                V8_CHECKERR( status );
            }

            if( level > 0 ) {
                dstSize.width  = _octaves[octave].getWidth();
                dstSize.height = _octaves[octave].getHeight();
                dstStep        = _octaves[octave].getByteSizePitch();
                cerr << __LINE__ << " call nppiAbsDiff_32f_C1R" << endl;
                status = nppiAbsDiff_32f_C1R( (const Npp32f*)_octaves[octave].getData( level ),
                                              dstStep,
                                              (const Npp32f*)_octaves[octave].getData( level-1 ),
                                              dstStep,
                                              (Npp32f*)_octaves[octave].getDogData( level-1 ),
                                              dstStep,
                                              dstSize );
                V8_CHECKERR( status );
            }
        }
    }
    keep.stop("    Time for V8 after all octaves, 3 levels pyramid: ");
}

#undef V8_FILTERSIZE
#undef V8_LEVELS

