#pragma once

#include <iostream>
#include <cublas_v2.h>

using namespace std;

class CuFortMatrix
{
    int            _rows;
    int            _cols;
    bool           _onDevice;
    bool           _ownedBuf;
    float*         _buf;

public:
    inline static int idxColmajBase1( int row, int col, int pitch )
    {
        return (col-1)*pitch + row-1;
    }

    inline static int idxColmajBase0( int row, int col, int pitch )
    {
        return col*pitch + row;
    }

    inline static int idxRowmajBase1( int row, int col, int pitch )
    {
        return (row-1)*pitch + col-1;
    }

    inline static int idxRowmajBase0( int row, int col, int pitch )
    {
        return row*pitch + col;
    }

public:
    enum {
        OnHost   = false,
        OnDevice = true
    };

    enum {
        DontAllocate = false,
        Allocate     = true
    };

    /** Construct a matrix structure for use with CuBlas matrix multiplication.
     *  Its memory arrangement complies with Fortran and Matlab (column major),
     *  not for C-like layout (row major).
     *  onDevice: OnHost assumes device on host, OnDevice assumes on a CUDA card.
     *  allocate: Allocate allocate memory in the constructor, DontAllocate
     *            waits for a subsequent call to setExternalBuf.
     */
    CuFortMatrix( const int rows, const int cols, bool onDevice, bool allocate=Allocate );
    ~CuFortMatrix( );

    /** Set the internal buffer to buf.
     *  The row/col information must have been set correctly in the constructor.
     *  It must be allocated with allocate=DontAllocate.
     *  Keep in mind that the internal interpretation is in column-major memory
     *  layout.
     */
    void setExternalBuf( float* buf );

    void setNull( );

    /** set value with base zero indexing */
    inline void set( int row, int col, float val ) {
        _buf[ idxColmajBase0( row, col, leadingDim() ) ] = val;
    }
    /** get value with base zero indexing */
    inline float get( int row, int col ) const {
        return _buf[ idxColmajBase0( row, col, leadingDim() ) ];
    }

    /** set value with base one indexing */
    inline void setBase1( int row, int col, float val ) {
        _buf[ idxColmajBase1( row, col, leadingDim() ) ] = val;
    }
    /** get value with base one indexing */
    inline float getBase1( int row, int col ) const {
        return _buf[ idxColmajBase1( row, col, leadingDim() ) ];
    }

    /** setMult performs C=A*B for two matrices in column major layout,
     *  which is the Matlab and Fortran layout.
     */
    void setMult( cublasHandle_t handle, const CuFortMatrix& A, const CuFortMatrix& B );

    /** setSwapTransMult performs C=B^T*A^T for two matrices in column major order.
     *  However, this is equivalent to C=A*B when A, B and C are all in row major layout,
     *  which is the C/C++ layout.
     */
    void setSwapTransMult( cublasHandle_t handle, const CuFortMatrix& A, const CuFortMatrix& B );

    /** setATransMult perform C = A^T * B
     */
    void setATransMult( cublasHandle_t handle, const CuFortMatrix& A, const CuFortMatrix& B );

    /** setBTransMult perform C = A * B^T
     *  This useful when multiplying 2 natrices that have identical layout, but one should
     *  transposed.
     */
    void setBTransMult( cublasHandle_t handle, const CuFortMatrix& A, const CuFortMatrix& B );

    inline float*       data( )             { return _buf; }
    inline const float* data( ) const       { return _buf; }
    inline int          rows( ) const       { return _rows; }
    inline int          cols( ) const       { return _cols; }
    inline int          leadingDim( ) const { return _rows; }
    inline bool         onDevice( ) const   { return _onDevice; }

    void print( ostream& ostr ) const;

    friend ostream& operator<<( ostream& ostr, const CuFortMatrix& m );

    CuFortMatrix& operator=( const CuFortMatrix& other );
};

