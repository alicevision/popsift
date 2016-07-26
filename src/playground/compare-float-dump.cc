#include <iostream>
#include <math.h>
#include <string>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

using namespace std;

void usage( const char* cmd, string info )
{
    cerr << "Error: " << info << endl
         << endl
         << "Usage: " << cmd << " <infile1> <infile2>" << endl
         << "    take two files containing a structured float dump of an image and print the pixel-by-pixel differences" << endl
         << endl
         << "     : " << cmd << " --show <infile2>" << endl
         << "    dump the float values of a structured float dump of an image" << endl
         << endl
         << "     : " << cmd << " --ext <infile1> <infile2> <infile3>" << endl
         << "    take 3 images and find if a pixel in the middle one is extrema for its neighbourhood" << endl
         << endl
         << "     : " << cmd << " --conv-pgm <infile> <outfile>" << endl
         << "    take an input image and write the int-converted values to an ASCII PGM file" << endl
         << endl;
    exit( -1 );
}

float* read_header( const char* filename, int& cols, int& rows )
{
    char  buf[ 100 ];

    FILE* f = fopen( filename, "r" );

    if( not f ) {
        fprintf( stderr, "Couldn't open file %s\n", filename );
        perror( "Cause: " );
        return 0;
    }

    if( fgets( buf, 100, f ) == 0 ) {
        fprintf( stderr, "Couldn't read first line in %s\n", filename );
        perror( "Cause: " );
        return 0;
    }
    if( strncmp( buf, "floats", 6 ) ) {
        fprintf( stderr, "Couldn't read first line 'floats' in %s\n", filename );
        return 0;
    }
    if( fgets( buf, 100, f ) == 0 ) {
        fprintf( stderr, "Couldn't read second line in %s\n", filename );
        perror( "Cause: " );
        return 0;
    }
    if( sscanf( buf, "%d %d", &cols, &rows ) != 2 ) {
        fprintf( stderr, "Couldn't read width X height in %s\n", filename );
        perror( "Cause: " );
        return 0;
    }

    int    ct   = cols * rows;
    float* data = new float[ ct ];

    if( fread( data, sizeof(float), ct, f ) != ct ) {
        fprintf( stderr, "Couldn't read %d floats from %s\n", ct, filename );
        perror( "Cause: " );
        return 0;
    }

    fclose( f );

    return data;
}

void showfile( const char* filename )
{
    int cols_1, rows_1;
    float* data1 = read_header( filename, cols_1, rows_1 );
    if( not data1 ) exit( -1 );

    for( int i=0; i<rows_1; i++ ) {
        for( int j=0; j<cols_1; j++ ) {
            printf("%.3f ", data1[i*cols_1 + j] );
        }
        printf("\n");
    }
}

void findext( const char* f0, const char* f1, const char* f2 )
{
    int cols[3], rows[3];
    float* data[3];

    data[0] = read_header( f0, cols[0], rows[0] );
    data[1] = read_header( f1, cols[1], rows[1] );
    data[2] = read_header( f2, cols[2], rows[2] );

    if( not data[0] ) exit( -1 );
    if( not data[1] ) exit( -1 );
    if( not data[2] ) exit( -1 );

#define TX(l,i,j) data[l][(i)*cols[0] + (j)]

#define TXSIGN(i,j,SGN) ( \
                ( TX(1,i,j) SGN TX(1,i-1,j-1) ) && \
                ( TX(1,i,j) SGN TX(1,i-1,j  ) ) && \
                ( TX(1,i,j) SGN TX(1,i-1,j+1) ) && \
 \
                ( TX(1,i,j) SGN TX(1,i  ,j-1) ) && \
                ( TX(1,i,j) SGN TX(1,i  ,j+1) ) && \
 \
                ( TX(1,i,j) SGN TX(1,i+1,j-1) ) && \
                ( TX(1,i,j) SGN TX(1,i+1,j  ) ) && \
                ( TX(1,i,j) SGN TX(1,i+1,j+1) ) && \
 \
                ( TX(1,i,j) SGN TX(0,i-1,j-1) ) && \
                ( TX(1,i,j) SGN TX(0,i-1,j  ) ) && \
                ( TX(1,i,j) SGN TX(0,i-1,j+1) ) && \
                ( TX(1,i,j) SGN TX(0,i  ,j-1) ) && \
                ( TX(1,i,j) SGN TX(0,i  ,j  ) ) && \
                ( TX(1,i,j) SGN TX(0,i  ,j+1) ) && \
                ( TX(1,i,j) SGN TX(0,i+1,j-1) ) && \
                ( TX(1,i,j) SGN TX(0,i+1,j  ) ) && \
                ( TX(1,i,j) SGN TX(0,i+1,j+1) ) && \
 \
                ( TX(1,i,j) SGN TX(2,i-1,j-1) ) && \
                ( TX(1,i,j) SGN TX(2,i-1,j  ) ) && \
                ( TX(1,i,j) SGN TX(2,i-1,j+1) ) && \
                ( TX(1,i,j) SGN TX(2,i  ,j-1) ) && \
                ( TX(1,i,j) SGN TX(2,i  ,j  ) ) && \
                ( TX(1,i,j) SGN TX(2,i  ,j+1) ) && \
                ( TX(1,i,j) SGN TX(2,i+1,j-1) ) && \
                ( TX(1,i,j) SGN TX(2,i+1,j  ) ) && \
                ( TX(1,i,j) SGN TX(2,i+1,j+1) ) )

    for( int i=1; i<rows[0]-1; i++ ) {
        for( int j=1; j<cols[0]-1; j++ ) {
            bool is_a_max = TXSIGN(i,j,>);
            bool is_a_min = TXSIGN(i,j,<);

            if( is_a_min ) {
                printf("Extremum (min) at (%d,%d): %f\n", j, i, TX(1,i,j) );
            }
            if( is_a_max ) {
                printf("Extremum (max) at (%d,%d): %f\n", j, i, TX(1,i,j) );
            }
        }
    }
}

void convpgm( const char* in_f, const char* out_f )
{
    int cols, rows;
    float* data;

    data = read_header( in_f, cols, rows );

    FILE* out = fopen( out_f, "w" );
    if( not out ) {
        fprintf( stderr, "Couldn't open %s for writing\n", out_f );
        perror(":");
        exit( -1 );
    }

#if 0
    int minval = 0;
    int maxval = 0;
    for( int row=0; row<rows; row++ ) {
        for( int col=0; col<cols; col++ ) {
            int val = (int)data[row*cols+col];
            minval = min( minval, val );
            maxval = max( maxval, val );
        }
    }
#endif

    fprintf( out, "P2\n"
                  "%d %d\n"
                  "%d\n",
                  rows, cols,
                  255 );
                  // maxval - minval );
    for( int row=0; row<rows; row++ ) {
        for( int col=0; col<cols; col++ ) {
            int val = (int)data[row*cols+col];
            // fprintf( out, "%d ", val - minval );
            fprintf( out, "%d ", val );
        }
        fprintf( out, "\n" );
    }

    fclose( out );
}

int main( int argc, char*argv[] )
{
    if( argc < 3 ) {
        usage( argv[0], "Wrong parameter count" );
    }

    if( string(argv[1]) == "--show" ) {
        if( argc != 3 ) usage( argv[0], "Wrong parameter count" );
        showfile( argv[2] );
        exit( 0 );
    }

    if( string(argv[1]) == "--conv-pgm" ) {
        if( argc != 4 ) usage( argv[0], "Wrong parameter count" );
        convpgm( argv[2], argv[3] );
        exit( 0 );
    }

    if( string(argv[1]) == "--ext" ) {
        if( argc != 5 ) usage( argv[0], "Wrong parameter count" );
        findext( argv[2], argv[3], argv[4] );
        exit( 0 );
    }

    if( argc != 3 ) usage( argv[0], "Wrong parameter count" );

    int cols_1, cols_2, rows_1, rows_2;

    float* data1 = read_header( argv[1], cols_1, rows_1 );
    if( not data1 ) exit( -1 );

    float* data2 = read_header( argv[2], cols_2, rows_2 );
    if( not data2 ) exit( -1 );

    if( cols_1 != cols_2 || rows_1 != rows_2 ) {
        fprintf( stderr, "Image dumps must be for same size\n"
                         "    size %d X %d for %s\n"
                         "    size %d X %d for %s\n",
                         cols_1, rows_1, argv[1],
                         cols_2, rows_2, argv[2] );
        exit( -1 );
    }

    float maxdiff = 0;
    for( int i=0; i<rows_1; i++ ) {
        for( int j=0; j<cols_1; j++ ) {
            float delta = data1[i*cols_1 + j] - data2[i*cols_1 + j];
            printf("%.3f ", delta );
            if( maxdiff < fabsf( delta ) ) maxdiff = fabsf( delta );
        }
        printf("\n");
    }

    printf( "Max diff: %f\n", maxdiff );
}

