/*
 * Copyright 2020, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <nppdefs.h>
#include <nppi.h>

#include "s_image.h"
#include "common/write_plane_2d.h"
#include "common/nppi_error_strings.h"

// #ifdef USE_NVTX
// #include <nvToolsExtCuda.h>
// #else
// #define nvtxRangePushA(a)
// #define nvtxRangePop()
// #endif

using namespace std;

namespace popsift {

static void chkCudaState( const char* file, int line )
{
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if( err != cudaSuccess )
    {
        std::cerr << file << ":" << line << ": CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit( -1 );
    }
}

static void chkSuccess( const char* file, int line, const char* ctx, NppStatus code )
{
    if( code < 0 )
    {
        std::cerr << file << ":" << line << " error calling " << ctx << ": " << code << std::endl;
        cudaError_t err = cudaGetLastError();
        std::cerr << "          CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
    else if( code > 0 )
    {
        std::cerr << file << ":" << line << " warning calling " << ctx << ": " << code << std::endl;
    }
}

/*************************************************************
 * ImageBase::wallisFilter
 *************************************************************/

static void boxFilter( const float* in, size_t in_pitch, float* out, size_t out_pitch, const NppiSize& roi, const NppiSize& mask, const NppiPoint& anchor )
{
    NppStatus err;

    std::cerr << __FILE__ << ":" <<  __LINE__ << ": input for nppiFilterBox_32f_C1R: " << std::endl
              << "    input pitch: " << in_pitch << std::endl
              << "    output pitch: " << out_pitch << std::endl
              << "    ROI: " << roi.width << "x" << roi.height << std::endl
              << "    mask: " << mask.width << "x" << mask.height << std::endl
              << "    anchor: " << anchor.x << "x" << anchor.y << std::endl
              << std::endl;
    err = nppiFilterBox_32f_C1R( in, in_pitch,
                                 out, out_pitch,
                                 roi,
                                 mask,
                                 anchor );
    chkCudaState( __FILE__, __LINE__ );
    chkSuccess( __FILE__, __LINE__, "nppiFilterBox_32f_C1R", err );
    chkCudaState( __FILE__, __LINE__ );
}

    // Taken from here: https://se.mathworks.com/matlabcentral/answers/287847-what-is-wallis-filter-i-have-an-essay-on-it-and-i-cannot-understand-of-find-info-on-it

    // function WallisFilter(obj, Md, Dd, Amax, p, W)
    // Md and Dd are mean and contrast to match,
    // Amax and p constrain the change in individual pixels,
void ImageBase::wallisFilter( Plane2D<float>& D, Plane2D<float>& input, int filterWidth, size_t pitch )
{
    std::cerr << __FILE__ << ":" <<  __LINE__ << ": entering wallisFilter, pitch=" << pitch
              << " filterwidth=" << filterWidth << std::endl;

    chkCudaState( __FILE__, __LINE__ );
    write_plane2D( "wallis-step-input.pgm", input );
    chkCudaState( __FILE__, __LINE__ );

    NppStatus err;
    const NppiSize  COMPLETE{_w,_h}; // = { .width = _w, .height = _h };
    const NppiPoint NOOFFSET{0, 0}; // = { .x = 0, .y = 0 };
    const float     Md   = 32767.0f; // desired average mean
    const float     Dd   = Md/2.0f;  // desired average standard deviation
    const float     Amax =     2.5f; // maximum gain factor to prevent extreme values
    const float     p    =     0.8f; // mean proportionality filter controlling image flatness [0:1]
 
    // int w = filterWidth >> 1; // floor(W/2)
    if( filterWidth %2 == 0 ) filterWidth++;

    const NppiSize FILTERSIZE{filterWidth,filterWidth}; //  = { .height = filterWidth, .width = filterWidth };

    const NppiSize edgeSize{ filterWidth/2, filterWidth/2 };
    const NppiSize protectedSize{ _w+filterWidth-1,_h+filterWidth-1 };

    Plane2D<float> I;
    I.allocDev( protectedSize.width, protectedSize.height,  popsift::ManagedMem, pitch );

    std::cerr << __FILE__ << ":" <<  __LINE__ << ": I allocated"
              << ", width=" << protectedSize.width
              << ", elemsize=" << I.elemSize()
              << ", align=" << pitch
              << ", pitch=" << I.getPitchInBytes() << std::endl;

    err = nppiCopyReplicateBorder_32f_C1R( input.data,
                                           input.getPitchInBytes(),
                                           COMPLETE,
                                           I.data,
                                           I.getPitchInBytes(),
                                           protectedSize,
                                           edgeSize.height,
                                           edgeSize.width );
    chkCudaState( __FILE__, __LINE__ );
    chkSuccess( __FILE__, __LINE__, "nppiCopyReplicateBorder_32f_C1R", err );
    chkCudaState( __FILE__, __LINE__ );
    write_plane2D( "wallis-step-I.pgm", I );
    chkCudaState( __FILE__, __LINE__ );

    Plane2D<float> M;
    M.allocDev( _w, _h, popsift::ManagedMem, pitch );
    chkCudaState( __FILE__, __LINE__ );

    std::cerr << __FILE__ << ":" <<  __LINE__ << ": M allocated"
              << ", width=" << _w
              << ", elemsize=" << M.elemSize()
              << ", align=" << pitch
              << ", pitch=" << M.getPitchInBytes() << std::endl;

#if 1
    boxFilter( I.data, // src ptr
               I.getPitchInBytes(),  // src step
               M.data,              // dst ptr
               M.getPitchInBytes(), // dst step
               COMPLETE,            // region
               FILTERSIZE,          // filtersize
               NOOFFSET );          // shift
#else
    std::cerr << __FILE__ << ":" <<  __LINE__ << ": input for nppiFilterBox_32f_C1R: "
              << " input pitch: " << I.getPitchInBytes()
              << " output pitch: " << M.getPitchInBytes()
              << " ROI: " << COMPLETE.width << "x" << COMPLETE.height << std::endl;
    err = nppiFilterBox_32f_C1R( I.data, // src ptr
                                 I.getPitchInBytes(),  // src step
                                 M.data,              // dst ptr
                                 M.getPitchInBytes(), // dst step
                                 COMPLETE,            // region
                                 FILTERSIZE,          // filtersize
                                 startPoint ); // NOOFFSET );          // shift
    chkCudaState( __FILE__, __LINE__ );
    chkSuccess( __FILE__, __LINE__, "nppiFilterBox_32f_C1R", err );
    chkCudaState( __FILE__, __LINE__ );
#endif
    write_plane2D( "wallis-step-M.pgm", M );
    chkCudaState( __FILE__, __LINE__ );

    // Plane2D<float> ipsum( _w, _h );
    // compute the inclusive prefix sum on all horizontals
    // after that compute the inclusive prefix sum on all verticals
    // that creates the basis for a box filter
    // ipsum = initBoxFilter( _input_image_d );
    // compute box filter ( pix(x+filterWidth/2,y+filterWidth/2) - pix(x-filterWidth/2,y-filterWidth/2) ) / filerWidth^2
    // M = runBoxFilter( ipsum, w );

    Plane2D<float> FminusM;
    FminusM.allocDev( _w, _h, popsift::ManagedMem, pitch );
    // FminusM = _input_image_d - M; // element-wise substract 
    err = nppiSub_32f_C1R( input.data,
                           input.getPitchInBytes(),
                           M.data,
                           M.getPitchInBytes(),
                           FminusM.data,
                           FminusM.getPitchInBytes(),
                           COMPLETE );
    chkSuccess( __FILE__, __LINE__, "nppiSub_32f_C1R", err );
    chkCudaState( __FILE__, __LINE__ );
    write_plane2D( "wallis-step-FminusM.pgm", FminusM );
    chkCudaState( __FILE__, __LINE__ );

    Plane2D<float> Dprep;
    Dprep.allocDev( _w, _h, popsift::ManagedMem, pitch );
    // Plane2D<float> D; - the output parameter
    // D.allocDev( _w, _h, popsift::ManagedMem, pitch );
    // compute element-wise: ( _input_image_d[pos] - M[pos] )^2
    // D = FminusM;
    // D.square(); // element-wise square
    err = nppiSqr_32f_C1R( FminusM.data,
                           FminusM.getPitchInBytes(),
                           Dprep.data,
                           Dprep.getPitchInBytes(),
                           COMPLETE );
    chkSuccess( __FILE__, __LINE__, "nppiSqr_32f_C1R", err );
    chkCudaState( __FILE__, __LINE__ );
    write_plane2D( "wallis-step-Dprep.pgm", Dprep );
    chkCudaState( __FILE__, __LINE__ );

    // ipsum = initBoxFilter( D );
    // D = runBoxFilter( ipsum, w );
    // D.divide( filterWidth^2 );
    err = nppiFilterBox_32f_C1R( Dprep.data,
                                 Dprep.getPitchInBytes(),
                                 D.data,
                                 D.getPitchInBytes(),
                                 COMPLETE,
                                 FILTERSIZE,
                                 NOOFFSET );
    chkSuccess( __FILE__, __LINE__, "nppiFilterBox_32f_C1R", err );
    chkCudaState( __FILE__, __LINE__ );
    write_plane2D( "wallis-step-D-1.pgm", D );
    chkCudaState( __FILE__, __LINE__ );

    // D.sqrt();
    err = nppiSqrt_32f_C1IR( D.data,
                             D.getPitchInBytes(),
                             COMPLETE );
    chkSuccess( __FILE__, __LINE__, "nppiSqrt_32f_C1IR", err );
    chkCudaState( __FILE__, __LINE__ );
    // D.multiply( Amax );
    err = nppiMulC_32f_C1IR( Amax,
                             D.data,
                             D.getPitchInBytes(),
                             COMPLETE );
    chkSuccess( __FILE__, __LINE__, "nppiMulC_32f_C1IR", err );
    chkCudaState( __FILE__, __LINE__ );
    // D.add( Dd );
    err = nppiAddC_32f_C1IR( Dd,
                             D.data,
                             D.getPitchInBytes(),
                             COMPLETE );
    chkSuccess( __FILE__, __LINE__, "nppiAddC_32f_C1IR", err );
    chkCudaState( __FILE__, __LINE__ );
    write_plane2D( "wallis-step-D-2.pgm", D );
    chkCudaState( __FILE__, __LINE__ );

    Plane2D<float> G;
    G.allocDev( _w, _h, popsift::ManagedMem, pitch );
    // G = FminusM;
    // G.multiply( Amax * Dd );
    err = nppiMulC_32f_C1R( D.data,
                            D.getPitchInBytes(),
                            Amax * Dd,
                            G.data,
                            G.getPitchInBytes(),
                            COMPLETE );
    chkSuccess( __FILE__, __LINE__, "nppiMulC_32f_C1R", err );
    chkCudaState( __FILE__, __LINE__ );
    write_plane2D( "wallis-step-G.pgm", G );
    chkCudaState( __FILE__, __LINE__ );

    // D = G / D; // element-wise division
    err = nppiDiv_32f_C1IR( G.data,
                            G.getPitchInBytes(),
                            D.data,
                            D.getPitchInBytes(),
                            COMPLETE );
    chkSuccess( __FILE__, __LINE__, "nppiDiv_32f_C1IR", err );
    chkCudaState( __FILE__, __LINE__ );

    // D.add( p * Md );
    err = nppiAddC_32f_C1IR( p * Md,
                             D.data,
                             D.getPitchInBytes(),
                             COMPLETE );
    chkSuccess( __FILE__, __LINE__, "nppiAddC_32f_C1IR", err );
    chkCudaState( __FILE__, __LINE__ );

    // M.multiply( 1-p );
    err = nppiMulC_32f_C1IR( 1.0f-p,
                             M.data,
                             M.getPitchInBytes(),
                             COMPLETE );
    chkSuccess( __FILE__, __LINE__, "nppiMulC_32f_C1IR", err );
    chkCudaState( __FILE__, __LINE__ );

    // D = D + M; // element-wise addition
    err = nppiAdd_32f_C1IR( M.data,
                            M.getPitchInBytes(),
                            D.data,
                            D.getPitchInBytes(),
                            COMPLETE );
    chkSuccess( __FILE__, __LINE__, "nppiAdd_32f_C1IR", err );
    chkCudaState( __FILE__, __LINE__ );
    write_plane2D( "wallis-step-D-3.pgm", D );
    chkCudaState( __FILE__, __LINE__ );
    // D.max(0);
#if 1
    err = nppiThreshold_32f_C1IR( D.data,
                                  D.getPitchInBytes(),
                                  COMPLETE,
                                  0.0f,
                                  NPP_CMP_LESS );
    chkSuccess( __FILE__, __LINE__, "nppiThreshold_32f_C1IR", err );
#else
    err = nppiThreshold_LTVal_32f_C1IR( D.data,
                                        D.getPitchInBytes(),
                                        COMPLETE,
                                        0.0f,   // if less-than this
                                        0.0f ); // set to this value
    chkSuccess( __FILE__, __LINE__, "nppiThreshold_LTVal_32f_C1IR", err );
#endif
    // D.min(65534)
    err = nppiThreshold_GTVal_32f_C1IR( D.data,
                                        D.getPitchInBytes(),
                                        COMPLETE,
                                        65534.0f,   // if greater-than this
                                        65534.0f ); // set to this value
    chkSuccess( __FILE__, __LINE__, "nppiThreshold_GTVal_32f_C1IR", err );
    write_plane2D( "wallis-step-D-4.pgm", D );
}

/*************************************************************
 * Image
 *************************************************************/

void Image::wallis( int filterWidth, size_t pitch )
{
    NppStatus err;
    const NppiSize  COMPLETE{_w,_h}; // = { .width = _w, .height = _h };
    Plane2D<float> a;
    Plane2D<float> b;
    a.allocDev( _w, _h, popsift::ManagedMem, pitch );
    b.allocDev( _w, _h, popsift::ManagedMem, pitch );
    chkCudaState( __FILE__, __LINE__ );

    std::cerr << __FILE__ << ":" << __LINE__ << " writing input image of size " << _w << "x" << _h << " for Wallis" << std::endl;

    chkCudaState( __FILE__, __LINE__ );
    write_plane2D( "wallis-step-byte-0-input.pgm", true, _input_image_d );
    chkCudaState( __FILE__, __LINE__ );

    std::cerr << __FILE__ << ":" << __LINE__ << ": input  pitch in bytes: " << _input_image_d.getPitchInBytes() << std::endl
              << "    output pitch in bytes: " << a.getPitchInBytes() << std::endl;

    chkCudaState( __FILE__, __LINE__ );
    err = nppiConvert_8u32f_C1R( _input_image_d.data,
                                 _input_image_d.getPitchInBytes(),
                                 a.data,
                                 a.getPitchInBytes(),
                                 COMPLETE );
    chkCudaState( __FILE__, __LINE__ );
    chkSuccess( __FILE__, __LINE__, "nppiConvert_8u32f_C1R", err );
    chkCudaState( __FILE__, __LINE__ );
    write_plane2D( "wallis-step-byte-1-a.pgm", a );
    chkCudaState( __FILE__, __LINE__ );

    err = nppiMulC_32f_C1R( a.data,
                            a.getPitchInBytes(),
                            256.0f,
                            b.data,
                            b.getPitchInBytes(),
                            COMPLETE );
    chkCudaState( __FILE__, __LINE__ );
    chkSuccess( __FILE__, __LINE__, "nppiMulC_32f_C1R", err );
    chkCudaState( __FILE__, __LINE__ );
    write_plane2D( "wallis-step-byte-2-b.pgm", a );
    chkCudaState( __FILE__, __LINE__ );

    wallisFilter( a, b, filterWidth, pitch );
    chkCudaState( __FILE__, __LINE__ );
    write_plane2D( "wallis-step-byte-3-a.pgm", a );
    chkCudaState( __FILE__, __LINE__ );

    err = nppiMulC_32f_C1R( a.data,
                            a.getPitchInBytes(),
                            256.0f,
                            b.data,
                            b.getPitchInBytes(),
                            COMPLETE );
    chkCudaState( __FILE__, __LINE__ );
    chkSuccess( __FILE__, __LINE__, "nppiMulC_32f_C1R", err );
    chkCudaState( __FILE__, __LINE__ );
    write_plane2D( "wallis-step-byte-4-b.pgm", a );
    chkCudaState( __FILE__, __LINE__ );

    err = nppiConvert_32f8u_C1R( b.data,
                                 b.getPitchInBytes(),
                                 _input_image_d.data,
                                 _input_image_d.getPitchInBytes(),
                                 COMPLETE,
                                 NPP_RND_NEAR );
    chkCudaState( __FILE__, __LINE__ );
    chkSuccess( __FILE__, __LINE__, "nppiConvert_32f8u_C1R", err );
    chkCudaState( __FILE__, __LINE__ );
}

/*************************************************************
 * ImageFloat
 *************************************************************/

void ImageFloat::wallis( int filterWidth, size_t pitch )
{
    const NppiSize  COMPLETE{_w,_h}; // = { .width = _w, .height = _h };
    Plane2D<float> D;
    D.allocDev( _w, _h, popsift::ManagedMem, pitch );

    nppiMulC_32f_C1R( _input_image_d.data,
                      _input_image_d.getPitchInBytes(),
                      65534.0f,
                      D.data,
                      D.getPitchInBytes(),
                      COMPLETE );

    wallisFilter( _input_image_d, D, filterWidth, pitch );

    nppiDivC_32f_C1IR( 65534.0f,
                       _input_image_d.data,
                       _input_image_d.getPitchInBytes(),
                       COMPLETE );
}

} // namespace popsift

