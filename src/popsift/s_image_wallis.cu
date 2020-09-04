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

static void boxFilter( Plane2D<float>& orig_input, Plane2D<float>& output, int filterWidth, size_t pitch )
{
    NppStatus err;

    int width     = output.getWidth();
    int height    = output.getHeight();

    const NppiSize  roi           { width, height };
    const NppiSize  mask          { filterWidth, filterWidth };
    const NppiPoint NOOFFSET      { 0, 0 };
    const NppiSize  edgeSize      { filterWidth/2, filterWidth/2 };
    const NppiSize  protectedSize { width + filterWidth - 1, height + filterWidth - 1 };

    Plane2D<float> I;
    I.allocDev( protectedSize.width, protectedSize.height,  popsift::ManagedMem, pitch );

    err = nppiCopyReplicateBorder_32f_C1R( orig_input.data,
                                           orig_input.getPitchInBytes(),
                                           roi,
                                           I.data,
                                           I.getPitchInBytes(),
                                           protectedSize,
                                           edgeSize.height,
                                           edgeSize.width );
    chkSuccess( __FILE__, __LINE__, "nppiCopyReplicateBorder_32f_C1R", err );

    float*       out       = output.data;
    const size_t out_pitch = output.getPitchInBytes();
    const float* in        = I.data;
    const size_t in_pitch  = I.getPitchInBytes();

    err = nppiFilterBox_32f_C1R( in, in_pitch,
                                 out, out_pitch,
                                 roi,
                                 mask,
                                 NOOFFSET );
    chkSuccess( __FILE__, __LINE__, "nppiFilterBox_32f_C1R", err );

    I.free();
}

static void printMinMax( Plane2D<float>& in, float& minVal, float& maxVal )
{
    NppStatus   stat;
    int width     = in.getWidth();
    int height    = in.getHeight();

    const NppiSize  roi { width, height };

    int sz;
    stat = nppiMinMaxGetBufferHostSize_32f_C1R( roi, &sz );
    chkSuccess( __FILE__, __LINE__, "nppiSub_32f_C1R", stat );

    unsigned char* ptr;
    cudaMallocManaged( &ptr, sz );
    chkCudaState( __FILE__, __LINE__ );

    struct MinMax
    {
        float _min;
        float _max;
    };
    struct MinMax* m;

    cudaMallocManaged( &m, sizeof(MinMax) );
    chkCudaState( __FILE__, __LINE__ );

    stat = nppiMinMax_32f_C1R( in.data, in.getPitchInBytes(), roi, &m->_min, &m->_max, ptr );
    chkCudaState( __FILE__, __LINE__ );
    chkSuccess( __FILE__, __LINE__, "nppiSub_32f_C1R", stat );

    minVal = m->_min;
    maxVal = m->_max;

    cudaFree( ptr );
    cudaFree( m );
}

/*************************************************************
 * ImageBase::wallisFilter
 *************************************************************/

    // Taken from here: https://se.mathworks.com/matlabcentral/answers/287847-what-is-wallis-filter-i-have-an-essay-on-it-and-i-cannot-understand-of-find-info-on-it

    // function WallisFilter(obj, Md, Dd, Amax, p, W)
    // Md and Dd are mean and contrast to match,
    // Amax and p constrain the change in individual pixels,
void ImageBase::wallisFilter( Plane2D<float>& input, int filterWidth, const float Md, const float Dd, const float Amax, const float p, Plane2D<float>& D, size_t pitch )
{
    std::cerr << __FILE__ << ":" <<  __LINE__ << ": entering wallisFilter, pitch=" << pitch
              << " filterwidth=" << filterWidth << std::endl;

    NppStatus err;
    const NppiSize  COMPLETE{_w,_h}; // = { .width = _w, .height = _h };
 
    // int w = filterWidth >> 1; // floor(W/2)
    if( filterWidth %2 == 0 ) filterWidth++;

    Plane2D<float> M;
    M.allocDev( _w, _h, popsift::ManagedMem, pitch );

    boxFilter( input,       // src
               M,           // dst
               filterWidth, // filtersize
               pitch );     // pitch alignment for allocations
    write_plane2D( "wallis-step-M.pgm", M );

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
    write_plane2D( "wallis-step-FminusM.pgm", FminusM );

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
    write_plane2D( "wallis-step-Dprep.pgm", Dprep );

    // ipsum = initBoxFilter( D );
    // D = runBoxFilter( ipsum, w );
    // D.divide( filterWidth^2 );
    boxFilter( Dprep,       // src
               D,           // dst
               filterWidth, // filtersize
               pitch );     // pitch alignment for allocations
    chkSuccess( __FILE__, __LINE__, "nppiFilterBox_32f_C1R", err );
    write_plane2D( "wallis-step-D-1.pgm", D );

    // D.sqrt();
    err = nppiSqrt_32f_C1IR( D.data,
                             D.getPitchInBytes(),
                             COMPLETE );
    chkSuccess( __FILE__, __LINE__, "nppiSqrt_32f_C1IR", err );

    // D.multiply( Amax );
    err = nppiMulC_32f_C1IR( Amax,
                             D.data,
                             D.getPitchInBytes(),
                             COMPLETE );
    chkSuccess( __FILE__, __LINE__, "nppiMulC_32f_C1IR", err );

    // D.add( Dd );
    err = nppiAddC_32f_C1IR( Dd,
                             D.data,
                             D.getPitchInBytes(),
                             COMPLETE );
    chkSuccess( __FILE__, __LINE__, "nppiAddC_32f_C1IR", err );
    write_plane2D( "wallis-step-D-2.pgm", D );

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
    write_plane2D( "wallis-step-G.pgm", G );

    // D = G / D; // element-wise division
    err = nppiDiv_32f_C1IR( G.data,
                            G.getPitchInBytes(),
                            D.data,
                            D.getPitchInBytes(),
                            COMPLETE );
    chkSuccess( __FILE__, __LINE__, "nppiDiv_32f_C1IR", err );

    // D.add( p * Md );
    err = nppiAddC_32f_C1IR( p * Md,
                             D.data,
                             D.getPitchInBytes(),
                             COMPLETE );
    chkSuccess( __FILE__, __LINE__, "nppiAddC_32f_C1IR", err );

    // M.multiply( 1-p );
    err = nppiMulC_32f_C1IR( 1.0f-p,
                             M.data,
                             M.getPitchInBytes(),
                             COMPLETE );
    chkSuccess( __FILE__, __LINE__, "nppiMulC_32f_C1IR", err );

    // D = D + M; // element-wise addition
    err = nppiAdd_32f_C1IR( M.data,
                            M.getPitchInBytes(),
                            D.data,
                            D.getPitchInBytes(),
                            COMPLETE );
    chkSuccess( __FILE__, __LINE__, "nppiAdd_32f_C1IR", err );
    write_plane2D( "wallis-step-D-3.pgm", D );

    // D.max(0);
    err = nppiThreshold_LTVal_32f_C1IR( D.data,
                                        D.getPitchInBytes(),
                                        COMPLETE,
                                        0.0f,   // if less-than this
                                        0.0f ); // set to this value
    chkSuccess( __FILE__, __LINE__, "nppiThreshold_LTVal_32f_C1IR", err );

    // D.min(65534)
    err = nppiThreshold_GTVal_32f_C1IR( D.data,
                                        D.getPitchInBytes(),
                                        COMPLETE,
                                        65534.0f,   // if greater-than this
                                        65534.0f ); // set to this value
    chkSuccess( __FILE__, __LINE__, "nppiThreshold_GTVal_32f_C1IR", err );
    write_plane2D( "wallis-step-D-4.pgm", D );

    M.free();
    FminusM.free();
    Dprep.free();
    G.free();
}

/*************************************************************
 * Image
 *************************************************************/

void Image::wallis( int filterWidth, const float Md, const float Dd, const float Amax, const float p, size_t pitch )
{
    float minVal, maxVal;

    NppStatus err;
    const NppiSize  COMPLETE{_w,_h}; // = { .width = _w, .height = _h };
    Plane2D<float> a;
    Plane2D<float> b;
    a.allocDev( _w, _h, popsift::ManagedMem, pitch );
    b.allocDev( _w, _h, popsift::ManagedMem, pitch );

    std::cerr << __FILE__ << ":" << __LINE__ << " writing input image of size " << _w << "x" << _h << " for Wallis" << std::endl;

    write_plane2D( "wallis-step-byte-0-input.pgm", true, _input_image_d );
    chkCudaState( __FILE__, __LINE__ );

    std::cerr << __FILE__ << ":" << __LINE__ << ": input  pitch in bytes: " << _input_image_d.getPitchInBytes() << std::endl
              << "    output pitch in bytes: " << a.getPitchInBytes() << std::endl;

    err = nppiConvert_8u32f_C1R( _input_image_d.data,
                                 _input_image_d.getPitchInBytes(),
                                 a.data,
                                 a.getPitchInBytes(),
                                 COMPLETE );
    chkSuccess( __FILE__, __LINE__, "nppiConvert_8u32f_C1R", err );
    write_plane2D( "wallis-step-byte-1-a.pgm", a );
    printMinMax( a, minVal, maxVal );
    std::cerr << "    " << __LINE__ << ": in a min=" << minVal << ", max=" << maxVal << std::endl;

    err = nppiMulC_32f_C1R( a.data,
                            a.getPitchInBytes(),
                            256.0f,
                            b.data,
                            b.getPitchInBytes(),
                            COMPLETE );
    chkSuccess( __FILE__, __LINE__, "nppiMulC_32f_C1R", err );
    write_plane2D( "wallis-step-byte-2-b.pgm", a );
    printMinMax( b, minVal, maxVal );
    std::cerr << "    " << __LINE__ << ": in b min=" << minVal << ", max=" << maxVal << std::endl;

    wallisFilter( b, filterWidth, Md, Dd, Amax, p, a, pitch );
    write_plane2D( "wallis-step-byte-3-a.pgm", a );
    printMinMax( a, minVal, maxVal );
    std::cerr << "    " << __LINE__ << ": in a min=" << minVal << ", max=" << maxVal << std::endl;

    err = nppiDivC_32f_C1R( a.data,
                            a.getPitchInBytes(),
                            256.0f,
                            b.data,
                            b.getPitchInBytes(),
                            COMPLETE );
    chkSuccess( __FILE__, __LINE__, "nppiMulC_32f_C1R", err );
    write_plane2D( "wallis-step-byte-4-b.pgm", a );
    printMinMax( b, minVal, maxVal );
    std::cerr << "    " << __LINE__ << ": in b min=" << minVal << ", max=" << maxVal << std::endl;

    Plane2D<uint8_t> c;
    c.allocDev( _w, _h, popsift::ManagedMem, pitch );
    err = nppiConvert_32f8u_C1R( b.data,
                                 b.getPitchInBytes(),
                                 c.data,
                                 c.getPitchInBytes(),
                                 COMPLETE,
                                 NPP_RND_NEAR );
    chkSuccess( __FILE__, __LINE__, "nppiConvert_32f8u_C1R", err );
    write_plane2D( "wallis-step-byte-5-b.pgm", c );

    err = nppiConvert_32f8u_C1R( b.data,
                                 b.getPitchInBytes(),
                                 _input_image_d.data,
                                 _input_image_d.getPitchInBytes(),
                                 COMPLETE,
                                 NPP_RND_NEAR );
    chkSuccess( __FILE__, __LINE__, "nppiConvert_32f8u_C1R", err );
}

/*************************************************************
 * ImageFloat
 *************************************************************/

void ImageFloat::wallis( int filterWidth, const float Md, const float Dd, const float Amax, const float p, size_t pitch )
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

    wallisFilter( D, filterWidth, Md, Dd, Amax, p, _input_image_d, pitch );

    nppiDivC_32f_C1IR( 65534.0f,
                       _input_image_d.data,
                       _input_image_d.getPitchInBytes(),
                       COMPLETE );
}

} // namespace popsift

