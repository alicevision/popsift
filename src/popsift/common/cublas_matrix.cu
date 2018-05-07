#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cublas_v2.h>

#include "cublas_matrix.h"

using namespace std;

template<typename T>
void cuda_alloc_dev( T** ptr, size_t size, const char* file, int line );

template<typename T>
void cuda_alloc_hst( T** ptr, size_t size, const char* file, int line );

void cuda_free_dev( void* ptr );
void cuda_free_hst( void* ptr );

/**********************************************************************************/
/* CuFortMatrix functions                                                               */
/**********************************************************************************/

CuFortMatrix::CuFortMatrix( const int rows, const int cols, bool onDevice, bool allocate )
    : _rows( rows )
    , _cols( cols )
    , _onDevice( onDevice )
    , _ownedBuf( allocate )
    , _buf( 0 )
{
    if( _ownedBuf )
    {
        if( _onDevice )
            cuda_alloc_dev( &_buf, _rows * _cols * sizeof(float), __FILE__, __LINE__ );
        else
            cuda_alloc_hst( &_buf, _rows * _cols * sizeof(float), __FILE__, __LINE__ );
    }
}

CuFortMatrix::~CuFortMatrix( )
{
    if( _ownedBuf )
    {
        if( _onDevice )
            cuda_free_dev( _buf );
        else
            cuda_free_hst( _buf );
    }
    _buf = 0;
}

void CuFortMatrix::setExternalBuf( float* buf )
{
    if( not _ownedBuf )
    {
        _buf = (float*)buf;
    }
    else
    {
        cerr << "Programming Error: " << __FILE__ << ":" << __LINE__ << endl
             << "    Trying to set external buf on CuFortMatrix configured for local allocation." << endl;
    }
}

void CuFortMatrix::setNull( )
{
    if( _onDevice )
        cudaMemset( _buf, 0, _rows*_cols*sizeof(float) );
    else
        memset( _buf, 0, _rows*_cols*sizeof(float) );
}

CuFortMatrix& CuFortMatrix::operator=( const CuFortMatrix& other )
{
    if( rows() != other.rows() || cols() != other.cols() )
    {
        cerr << "Cannot assign matrices of different dimensions to each other" << endl;
        return *this;
    }
    if( onDevice() == CuFortMatrix::OnHost && other.onDevice() == CuFortMatrix::OnHost )
    {
        memcpy( data(), other.data(), rows()*cols()*sizeof(float) );
    }
    else if( onDevice() == CuFortMatrix::OnDevice && other.onDevice() == CuFortMatrix::OnDevice )
    {
        cudaMemcpy( data(), other.data(), rows()*cols()*sizeof(float), cudaMemcpyDeviceToDevice );
    }
    else if( onDevice() == CuFortMatrix::OnDevice && other.onDevice() == CuFortMatrix::OnHost )
    {
        cublasStatus_t stat;
        stat = cublasSetMatrix( rows(), cols(), sizeof(float),
                                other.data(), other.leadingDim(),
                                data(), leadingDim() );
        if( stat != CUBLAS_STATUS_SUCCESS ) {
            cerr << "Error in copying matrix from host to device.";
        }
    }
    else
    {
        cublasStatus_t stat;
        stat = cublasGetMatrix( rows(), cols(), sizeof(float),
                                other.data(), other.leadingDim(),
                                data(), leadingDim() );
        if( stat != CUBLAS_STATUS_SUCCESS ) {
            cerr << "Error in copying matrix from device to to.";
        }
    }

    return *this;
}

ostream& operator<<( ostream& ostr, const CuFortMatrix& m )
{
    m.print( ostr );
    return ostr;
}

void CuFortMatrix::print( ostream& ostr ) const
{
    ostr << rows() << "x" << cols() << " =" << endl;
    for( int row=0; row<rows(); row++ ) {
        for( int col=0; col<cols(); col++ ) {
            ostr << setw(4) << get(row,col) << " ";
        }
        ostr << endl;
    }
}

void CuFortMatrix::setMult( cublasHandle_t handle, const CuFortMatrix& left, const CuFortMatrix& right )
{
    if( not onDevice() || not left.onDevice() || not right.onDevice() ) {
        cerr << "ERROR in " << __FILE__ << ":" << __LINE__ << endl
             << "    Supporting only multiplication on the device side so far. Doing nothing." << endl;
        return;
    }
    if( left.rows() != rows() ) {
        cerr << "ERROR in " << __FILE__ << ":" << __LINE__ << endl
             << "    left input and output rows must be identical. Doing nothing." << endl;
        return;
    }
    if( right.cols() != cols() ) {
        cerr << "ERROR in " << __FILE__ << ":" << __LINE__ << endl
             << "    right input and output cols must be identical. Doing nothing." << endl;
        return;
    }
    if( left.cols() != right.rows() ) {
        cerr << "ERROR in " << __FILE__ << ":" << __LINE__ << endl
             << "    left input cols and right input rows must be identical. Doing nothing." << endl;
        return;
    }

    setNull( );

    float alpha = 1.0f;
    float beta  = 1.0f;
    cublasStatus_t stat;
    stat = cublasSgemm( handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        rows(), cols(), left.cols(),
                        &alpha,
                        left.data(), left.leadingDim(),
                        right.data(), right.leadingDim(),
                        &beta,
                        data(), leadingDim() );
    if( stat != CUBLAS_STATUS_SUCCESS ) {
        cerr << "CuFortMatrix multiplication failed" << endl;
    }
}

void CuFortMatrix::setSwapTransMult( cublasHandle_t handle, const CuFortMatrix& left, const CuFortMatrix& right )
{
    if( not onDevice() || not left.onDevice() || not right.onDevice() ) {
        cerr << "ERROR: Supporting only multiplication on the device side so far. Doing nothing." << endl;
        return;
    }
    if( right.cols() != rows() ) {
        cerr << "ERROR: left input and output rows must be identical. Doing nothing." << endl;
        return;
    }
    if( left.rows() != cols() ) {
        cerr << "ERROR: right input and output cols must be identical. Doing nothing." << endl;
        return;
    }
    if( left.cols() != right.rows() ) {
        cerr << "ERROR: left input cols and right input rows must be identical. Doing nothing." << endl;
        return;
    }

    setNull( );

    float alpha = 1.0f;
    float beta  = 1.0f;
    cublasStatus_t stat;
    stat = cublasSgemm( handle,
                        CUBLAS_OP_T, CUBLAS_OP_T,
                        rows(), cols(), left.cols(),
                        &alpha,
                        right.data(), right.leadingDim(),
                        left.data(), left.leadingDim(),
                        &beta,
                        data(), leadingDim() );
    if( stat != CUBLAS_STATUS_SUCCESS ) {
        cerr << "CuFortMatrix multiplication failed" << endl;
    }
}

void CuFortMatrix::setATransMult( cublasHandle_t handle, const CuFortMatrix& left, const CuFortMatrix& right )
{
    if( not onDevice() || not left.onDevice() || not right.onDevice() ) {
        cerr << "ERROR in " << __FILE__ << ":" << __LINE__ << endl
             << "    Supporting only multiplication on the device side so far. Doing nothing." << endl
             << "    A^T MxK = " << left.cols() << "x" << left.rows() << endl
             << "    B   KxN = " << right.rows() << "x" << right.cols() << endl
             << "    C   MxN = " << rows() << "x" << cols() << endl;
        return;
    }
    if( right.cols() != cols() ) {
        cerr << "ERROR in " << __FILE__ << ":" << __LINE__ << endl
             << "    left input and output rows must be identical. Doing nothing." << endl
             << "    A^T MxK = " << left.cols() << "x" << left.rows() << endl
             << "    B   KxN = " << right.rows() << "x" << right.cols() << endl
             << "    C   MxN = " << rows() << "x" << cols() << endl;
        return;
    }
    if( left.cols() != rows() ) {
        cerr << "ERROR in " << __FILE__ << ":" << __LINE__ << endl
             << "    right input and output cols must be identical. Doing nothing." << endl
             << "    A^T MxK = " << left.cols() << "x" << left.rows() << endl
             << "    B   KxN = " << right.rows() << "x" << right.cols() << endl
             << "    C   MxN = " << rows() << "x" << cols() << endl;
        return;
    }
    if( left.rows() != right.rows() ) {
        cerr << "ERROR in " << __FILE__ << ":" << __LINE__ << endl
             << "    left input cols and right input rows must be identical. Doing nothing." << endl
             << "    A^T MxK = " << left.cols() << "x" << left.rows() << endl
             << "    B   KxN = " << right.rows() << "x" << right.cols() << endl
             << "    C   MxN = " << rows() << "x" << cols() << endl;
        return;
    }

    setNull( );

    float alpha = 1.0f;
    float beta  = 1.0f;
    cublasStatus_t stat;
    stat = cublasSgemm( handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        rows(), cols(), right.rows(),
                        &alpha,
                        left.data(), left.leadingDim(),
                        right.data(), right.leadingDim(),
                        &beta,
                        data(), leadingDim() );
    if( stat != CUBLAS_STATUS_SUCCESS ) {
        cerr << "ERROR in " << __FILE__ << ":" << __LINE__ << endl
             << "    CuFortMatrix multiplication failed" << endl;
    }
}

void CuFortMatrix::setBTransMult( cublasHandle_t handle, const CuFortMatrix& left, const CuFortMatrix& right )
{
    if( not onDevice() || not left.onDevice() || not right.onDevice() ) {
        cerr << "ERROR in " << __FILE__ << ":" << __LINE__ << endl
             << "    Supporting only multiplication on the device side so far. Doing nothing." << endl
             << "    A   MxK = " << left.rows() << "x" << left.cols() << endl
             << "    B^T KxN = " << right.cols() << "x" << right.rows() << endl
             << "    C   MxN = " << rows() << "<" << cols() << endl;
        return;
    }
    if( right.rows() != rows() ) {
        cerr << "ERROR in " << __FILE__ << ":" << __LINE__ << endl
             << "    left input and output rows must be identical. Doing nothing." << endl
             << "    A   MxK = " << left.rows() << "x" << left.cols() << endl
             << "    B^T KxN = " << right.cols() << "x" << right.rows() << endl
             << "    C   MxN = " << rows() << "<" << cols() << endl;
        return;
    }
    if( left.rows() != cols() ) {
        cerr << "ERROR in " << __FILE__ << ":" << __LINE__ << endl
             << "    right input and output cols must be identical. Doing nothing." << endl
             << "    A   MxK = " << left.rows() << "x" << left.cols() << endl
             << "    B^T KxN = " << right.cols() << "x" << right.rows() << endl
             << "    C   MxN = " << rows() << "<" << cols() << endl;
        return;
    }
    if( left.cols() != right.cols() ) {
        cerr << "ERROR in " << __FILE__ << ":" << __LINE__ << endl
             << "    left input cols and right input rows must be identical. Doing nothing." << endl
             << "    A   MxK = " << left.rows() << "x" << left.cols() << endl
             << "    B^T KxN = " << right.cols() << "x" << right.rows() << endl
             << "    C   MxN = " << rows() << "<" << cols() << endl;
        return;
    }

    setNull( );

    float alpha = 1.0f;
    float beta  = 1.0f;
    cublasStatus_t stat;
    stat = cublasSgemm( handle,
                        CUBLAS_OP_N, CUBLAS_OP_T,
                        rows(), cols(), left.cols(),
                        &alpha,
                        left.data(), left.leadingDim(),
                        right.data(), right.leadingDim(),
                        &beta,
                        data(), leadingDim() );
    if( stat != CUBLAS_STATUS_SUCCESS ) {
        cerr << "ERROR in " << __FILE__ << ":" << __LINE__ << endl
             << "    CuFortMatrix multiplication failed" << endl;
    }
}

/**********************************************************************************/
/* helper functions                                                               */
/**********************************************************************************/

template<typename T>
void cuda_alloc_dev( T** ptr, size_t size, const char* file, int line )
{
    cudaError_t err;
    err = cudaMalloc( ptr, size );
    if( err != cudaSuccess ) {
        cerr << "Could not alloc " << size << " bytes on device." << endl
             << "    file=" << file << ":" << line << endl;
        exit( -1 );
    }
}

template<typename T>
void cuda_alloc_hst( T** ptr, size_t size, const char* file, int line )
{
    cudaError_t err;
    err = cudaMallocHost( ptr, size );
    if( err != cudaSuccess ) {
        cerr << "Could not alloc " << size << " bytes on host." << endl
             << "    file=" << file << ":" << line << endl;
        exit( -1 );
    }
}

void cuda_free_dev( void* ptr )
{
    cudaFree( ptr );
}

void cuda_free_hst( void* ptr )
{
    cudaFreeHost( ptr );
}

