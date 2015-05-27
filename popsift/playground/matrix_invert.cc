#include <iostream>
#include <iomanip>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

using namespace std;

bool inverse_gaussian_elimination( const float B [/*Y=*/3] [/*X=*/3],
                                         float A [/*Y=*/3] [/*X=*/3],
                                         float inputVector [/*X=*/3],
                                         float outputVector [/*X=*/3] )
{
    float b[3];

    for( int j=0; j<3; j++ ) {
        for( int i=0; i<3; i++ )
            A[j][i] = B[j][i];
        b[j] = inputVector[j];
    }

    for( int j = 0 ; j < 3 ; j++ ) {
        // look for leading pivot
        float maxa    = 0;
        float maxabsa = 0;
        int   maxi    = -1;
        for( int i = j ; i < 3 ; i++ ) {
            float a    = A[j][i];
            float absa = fabs( a );
            if ( absa > maxabsa ) {
                maxa    = a;
                maxabsa = absa;
                maxi    = i;
            }
        }

        // singular?
        if( maxabsa < 1e-15 ) {
            b[0] = 0;
            b[1] = 0;
            b[2] = 0;
            return false;
        }

        int i = maxi;

        // swap j-th row with i-th row and
        // normalize j-th row
        for(int jj = j ; jj < 3 ; ++jj) {
            float tmp = A[jj][j];
            A[jj][j]  = A[jj][i];
            A[jj][i]  = tmp;
            A[jj][j] /= maxa;
        }
        float tmp = b[j];
        b[j]  = b[i];
        b[i]  = tmp;
        b[j] /= maxa;

        // elimination
        for(int ii = j+1 ; ii < 3 ; ++ii) {
            float x = A[j][ii];
            for( int jj = j ; jj < 3 ; jj++ ) {
                A[jj][ii] -= x * A[jj][j];
            }
            b[ii] -= x * b[j] ;
        }
    }

    // backward substitution
    for( int i = 2 ; i > 0 ; i-- ) {
        float x = b[i] ;
        for( int ii = i-1 ; ii >= 0 ; ii-- ) {
            b[ii] -= x * A[i][ii];
        }
    }

    outputVector[0] = b[0];
    outputVector[1] = b[1];
    outputVector[2] = b[2];

    return true;
}

float symmetricDeterminant( const float theMatrix [/*Y=*/3] [/*X=*/3] )
{
    float a1 = theMatrix[0][0] * theMatrix[1][1] * theMatrix[2][2];
    float a2 = theMatrix[0][0] * theMatrix[1][2] * theMatrix[1][2];
    float b1 = theMatrix[0][1] * theMatrix[0][1] * theMatrix[2][2];
    float b2 = theMatrix[0][1] * theMatrix[0][2] * theMatrix[1][2];
    float c1 = b2;
    float c2 = theMatrix[0][2] * theMatrix[0][2] * theMatrix[1][1];
    return a1 - a2 - b1 + b2 + c1 - c2;
}

float determinantOfMinor( int          theRowHeightY,
                           int          theColumnWidthX,
                           const float theMatrix [/*Y=*/3] [/*X=*/3] )
{
  int x1 = theColumnWidthX == 0 ? 1 : 0;  /* always either 0 or 1 */
  int x2 = theColumnWidthX == 2 ? 1 : 2;  /* always either 1 or 2 */
  int y1 = theRowHeightY   == 0 ? 1 : 0;  /* always either 0 or 1 */
  int y2 = theRowHeightY   == 2 ? 1 : 2;  /* always either 1 or 2 */

  return ( theMatrix [y1] [x1]  *  theMatrix [y2] [x2] )
      -  ( theMatrix [y1] [x2]  *  theMatrix [y2] [x1] );
}

float determinantOfMinorSymmetric( int         threadId,
                                   int         excludeX,
                                   int         excludeY,
                                   const float matrix[3][3] )
{
    /*
     * we want only the determinants opposing
     * (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
     *   00     01     02     11     12     22
     *  b3_0   b3_1   b3_2   b3_4   b3_5   b3_8
     * or: a subset best expressed at base 3
     */
    uint32_t mask = threadId % 6;
    uint32_t b3_y = ( mask != 3 ) : mask / 3 : 2;
    uint32_t b3_x = ( mask != 3 ) : mask % 3 : 2;
}

void symmetricInverse( const float i [/*Y=*/3] [/*X=*/3],
                             float o [/*Y=*/3] [/*X=*/3] )
{
    float det0b = - i[1][2] * i[1][2];
    float det0a =   i[1][1] * i[2][2];
    float det0 = det0b + det0a;

    float det1b = - i[0][1] * i[2][2];
    float det1a =   i[1][2] * i[0][2];
    float det1 = det1b + det1a;

    float det2b = - i[1][1] * i[0][2];
    float det2a =   i[0][1] * i[1][2];
    float det2 = det2b + det2a;

    float det3b = - i[0][2] * i[0][2];
    float det3a =   i[0][0] * i[2][2];
    float det3 = det3b + det3a;

    float det4b = - i[0][0] * i[1][2];
    float det4a =   i[0][1] * i[0][2];
    float det4 = det4b + det4a;

    float det5b = - i[0][1] * i[0][1];
    float det5a =   i[0][0] * i[1][1];
    float det5 = det5b + det5a;

    float det;
    det  = ( i[0][0] * det0 );
    det += ( i[0][1] * det1 );
    det += ( i[0][2] * det2 );

    float rsd = 1.0 / det;

    o[0][0] = det0 * rsd;
    o[1][0] = det1 * rsd;
    o[2][0] = det2 * rsd;
    o[1][1] = det3 * rsd;
    o[1][2] = det4 * rsd;
    o[2][2] = det5 * rsd;
    o[0][1] = o[1][0];
    o[0][2] = o[2][0];
    o[2][1] = o[1][2];
}

float determinant( const float theMatrix [/*Y=*/3] [/*X=*/3] )
{
  return ( theMatrix [0] [0]  *  determinantOfMinor( 0, 0, theMatrix ) )
      -  ( theMatrix [0] [1]  *  determinantOfMinor( 0, 1, theMatrix ) )
      +  ( theMatrix [0] [2]  *  determinantOfMinor( 0, 2, theMatrix ) );
}

bool inverse( const float theMatrix [/*Y=*/3] [/*X=*/3],
                    float theOutput [/*Y=*/3] [/*X=*/3] )
{
  float det = determinant( theMatrix );

    /* Arbitrary for now.  This should be something nicer... */
  if ( fabs(det) < 1e-2 )
  {
    memset( theOutput, 0, 9*sizeof(float) );
    return false;
  }

  float oneOverDeterminant = 1.0 / det;

  for (   int y = 0;  y < 3;  y ++ )
    for ( int x = 0;  x < 3;  x ++   )
    {
        /* Rule is inverse = 1/det * minor of the TRANSPOSE matrix.  *
         * Note (y,x) becomes (x,y) INTENTIONALLY here!              */
      theOutput [y] [x]
        = determinantOfMinor( x, y, theMatrix ) * oneOverDeterminant;

        /* (y0,x1)  (y1,x0)  (y1,x2)  and (y2,x1)  all need to be negated. */
      if( 1 == ((x + y) % 2) )
        theOutput [y] [x] = - theOutput [y] [x];
    }

  return true;
}

void printVector( const float theVector [/*X=*/3] )
{
    cout << "[  ";
        for ( int x = 0;  x < 3;  x ++   )
            cout << theVector [x] << "  ";
    cout << "]" << endl;
}

void printMatrix( const float theMatrix [/*Y=*/3] [/*X=*/3] )
{
  for ( int y = 0;  y < 3;  y ++ )
  {
    cout << "[  ";
    for ( int x = 0;  x < 3;  x ++   )
      cout << theMatrix [y] [x] << "  ";
    cout << "]" << endl;
  }
  cout << endl;
}

void matrixMultiply(  const float theMatrixA [/*Y=*/3] [/*X=*/3],
                      const float theMatrixB [/*Y=*/3] [/*X=*/3],
                            float theOutput  [/*Y=*/3] [/*X=*/3]  )
{
  for (   int y = 0;  y < 3;  y ++ )
    for ( int x = 0;  x < 3;  x ++   )
    {
      theOutput [y] [x] = 0;
      for ( int i = 0;  i < 3;  i ++ )
        theOutput [y] [x] +=  theMatrixA [y] [i] * theMatrixB [i] [x];
    }
}

void matrixMultiply(  const float theMatrix [/*Y=*/3] [/*X=*/3],
                      const float theInput  [/*X=*/3],
                            float theOutput [/*X=*/3] )
{
    for (   int y = 0;  y < 3;  y ++ ) {
        theOutput [y] = 0;
        for ( int i = 0;  i < 3;  i ++ )
            theOutput [y] +=  theMatrix [y] [i] * theInput [i];
    }
}

#define RANDOM_D(a,b)   float( random() % b )

int main(int argc, char **argv)
{
  if ( argc > 1 )
    srandom( atoi( argv[1] ) );

  float m[3][3];
  float o[3][3];
  float mm[3][3];
  float r[3];
  for( int i=0; i<3; i++ ) {
    for( int j=0; j<3; j++ )
        m[i][j] = ( random() % 10000 - 5000 ) / 1000.0;
    r[i] = ( random() % 10000 - 5000 ) / 1000.0;
  }
  m[0][1] = m[1][0];
  m[0][2] = m[2][0];
  m[1][2] = m[2][1];

  if ( argc <= 2 )
    cout << fixed << setprecision(3);

  cerr << "The input matrix:" << endl;
  printMatrix(m);
  cout << endl << endl;

  // SHOW( determinant(m) );
  // cout << endl << endl;

  // BOUT( inverse(m, o) );
  inverse(m, o);
  cerr << "The input matrix after calling inverse:" << endl;
  printMatrix(m);
  cerr << "The output matrix after calling inverse:" << endl;
  printMatrix(o);
  // cout << endl << endl;

  matrixMultiply( m, o, mm );
  // cerr << "The input matrix after calling multiply:" << endl;
  // printMatrix(m);
  // cerr << "The output matrix after calling multiply:" << endl;
  // printMatrix(o);
  cerr << "The result matrix after calling multiply:" << endl;
  printMatrix(mm);  
  cout << endl << endl;

  cout << "Determinant: " << determinant(m) << endl;
  cout << "Symmetric determinant: " << determinant(m) << endl << endl;
  cout << "Symmetric inverse: " << endl;
  symmetricInverse( m, o );
  printMatrix(o);  

  cerr << "The input vector:" << endl;
  printVector( r );
  float outv[3];
  matrixMultiply( o, r, outv );
  cerr << "The output vector after calling multiply:" << endl;
  printVector( outv );
  matrixMultiply( m, outv, r );
  cerr << "The output vector after calling multiply with the inverse:" << endl;
  printVector( r );
  cout << endl << endl;

  inverse_gaussian_elimination( m, o, r, outv );
  cerr << "The input matrix after calling gaussian inverse:" << endl;
  printMatrix(m);
  cerr << "The output matrix after calling gaussian inverse:" << endl;
  printMatrix(o);
  cout << endl << endl;

  cerr << "The input vector:" << endl;
  printVector( r );
  cerr << "The output vector after calling gauss elim:" << endl;
  printVector( outv );
  cout << endl << endl;
}

