// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "../src/device_internal.hh"

#include <string>

using testsweeper::get_wtime;

// -----------------------------------------------------------------------------
void test_enums()
{
    printf( "%s\n", __func__ );

    using str = std::string;

    require( blas::layout2char( blas::Layout::ColMajor ) == 'C' );
    require( blas::layout2char( blas::Layout::RowMajor ) == 'R' );
    require( blas::layout2str( blas::Layout::ColMajor ) == str("col") );
    require( blas::layout2str( blas::Layout::RowMajor ) == str("row") );
    require( blas::char2layout( 'C' ) == blas::Layout::ColMajor );
    require( blas::char2layout( 'c' ) == blas::Layout::ColMajor );
    require( blas::char2layout( 'R' ) == blas::Layout::RowMajor );
    require( blas::char2layout( 'r' ) == blas::Layout::RowMajor );

    require( blas::op2char( blas::Op::NoTrans   ) == 'N' );
    require( blas::op2char( blas::Op::Trans     ) == 'T' );
    require( blas::op2char( blas::Op::ConjTrans ) == 'C' );
    require( blas::op2str( blas::Op::NoTrans   ) == str("notrans") );
    require( blas::op2str( blas::Op::Trans     ) == str("trans") );
    require( blas::op2str( blas::Op::ConjTrans ) == str("conj") );
    require( blas::char2op( 'N' ) == blas::Op::NoTrans   );
    require( blas::char2op( 'n' ) == blas::Op::NoTrans   );
    require( blas::char2op( 'T' ) == blas::Op::Trans     );
    require( blas::char2op( 't' ) == blas::Op::Trans     );
    require( blas::char2op( 'C' ) == blas::Op::ConjTrans );
    require( blas::char2op( 'c' ) == blas::Op::ConjTrans );

    require( blas::uplo2char( blas::Uplo::Upper   ) == 'U' );
    require( blas::uplo2char( blas::Uplo::Lower   ) == 'L' );
    require( blas::uplo2char( blas::Uplo::General ) == 'G' );
    require( blas::uplo2str( blas::Uplo::Upper   ) == str("upper") );
    require( blas::uplo2str( blas::Uplo::Lower   ) == str("lower") );
    require( blas::uplo2str( blas::Uplo::General ) == str("general") );
    require( blas::char2uplo( 'U' ) == blas::Uplo::Upper   );
    require( blas::char2uplo( 'u' ) == blas::Uplo::Upper   );
    require( blas::char2uplo( 'L' ) == blas::Uplo::Lower   );
    require( blas::char2uplo( 'l' ) == blas::Uplo::Lower   );
    require( blas::char2uplo( 'G' ) == blas::Uplo::General );
    require( blas::char2uplo( 'g' ) == blas::Uplo::General );

    require( blas::diag2char( blas::Diag::NonUnit ) == 'N' );
    require( blas::diag2char( blas::Diag::Unit    ) == 'U' );
    require( blas::diag2str( blas::Diag::NonUnit ) == str("nonunit") );
    require( blas::diag2str( blas::Diag::Unit    ) == str("unit") );
    require( blas::char2diag( 'N' ) == blas::Diag::NonUnit );
    require( blas::char2diag( 'n' ) == blas::Diag::NonUnit );
    require( blas::char2diag( 'U' ) == blas::Diag::Unit    );
    require( blas::char2diag( 'u' ) == blas::Diag::Unit    );

    require( blas::side2char( blas::Side::Left  ) == 'L' );
    require( blas::side2char( blas::Side::Right ) == 'R' );
    require( blas::side2str( blas::Side::Left  ) == str("left") );
    require( blas::side2str( blas::Side::Right ) == str("right") );
    require( blas::char2side( 'L' ) == blas::Side::Left  );
    require( blas::char2side( 'l' ) == blas::Side::Left  );
    require( blas::char2side( 'R' ) == blas::Side::Right );
    require( blas::char2side( 'r' ) == blas::Side::Right );
}

// -----------------------------------------------------------------------------
void test_exceptions()
{
    printf( "%s\n", __func__ );

    using str = std::string;

    try {
        throw blas::Error();
    }
    catch (blas::Error const& ex) {
        require( ex.what() == str("") );
    }
    catch (...) {
        require( false );
    }

    //--------------------
    // Error inherits from std::exception
    try {
        throw blas::Error();
    }
    catch (std::exception const& ex) {
        require( ex.what() == str("") );
    }
    catch (...) {
        require( false );
    }

    //--------------------
    // Error can take message.
    try {
        throw blas::Error( "message" );
    }
    catch (std::exception const& ex) {
        require( ex.what() == str("message") );
    }
    catch (...) {
        require( false );
    }

    //--------------------
    // Error can take message and function.
    try {
        throw blas::Error( "message", __func__ );
    }
    catch (std::exception const& ex) {
        require( ex.what() == str("message, in function test_exceptions") );
    }
    catch (...) {
        require( false );
    }
}

// -----------------------------------------------------------------------------
void test_abs1()
{
    printf( "%s\n", __func__ );

    float sx = -3.141592653589793f;
    require( blas::abs1( sx ) == -sx );

    double dx = -3.141592653589793;
    require( blas::abs1( dx ) == -dx );

    float sxi = -2.718281828459045f;
    std::complex<float> cx( sx, sxi );
    require( blas::abs1( cx ) == -sx + -sxi );

    double dxi = -2.718281828459045;
    std::complex<double> zx( dx, dxi );
    require( blas::abs1( zx ) == -dx + -dxi );
}

// -----------------------------------------------------------------------------
void test_is_complex()
{
    printf( "%s\n", __func__ );

    require( ! blas::is_complex< int >::value );
    require( ! blas::is_complex< int64_t >::value );
    require( ! blas::is_complex< float >::value );
    require( ! blas::is_complex< double >::value );
    require( blas::is_complex< std::complex<float> >::value );
    require( blas::is_complex< std::complex<double> >::value );
}

// -----------------------------------------------------------------------------
void test_real_imag_conj()
{
    printf( "%s\n", __func__ );

    using blas::real;  // same as std::real
    using blas::imag;  // same as std::imag
    using blas::conj;  // different than std::conj for real types

    int ix = 1234;
    int ir1 = real( ix );  require( ir1 == ix );
    int ii1 = imag( ix );  require( ii1 == 0  );
    int ic1 = conj( ix );  require( ic1 == ix );
    int ir2 = blas::real( ix );  require( ir2 == ix );
    int ii2 = blas::imag( ix );  require( ii2 == 0  );
    int ic2 = blas::conj( ix );  require( ic2 == ix );
    int ir3 =  std::real( ix );  require( ir3 == ix );
    int ii3 =  std::imag( ix );  require( ii3 == 0  );
    //int ic3 =  std::conj( ix );  require( ic3 == ix );  // compile error

    float sx = 3.141592653589793f;
    float sr1 = real( sx );  require( sr1 == sx );
    float si1 = imag( sx );  require( si1 == 0  );
    float sc1 = conj( sx );  require( sc1 == sx );
    float sr2 = blas::real( sx );  require( sr2 == sx );
    float si2 = blas::imag( sx );  require( si2 == 0  );
    float sc2 = blas::conj( sx );  require( sc2 == sx );
    float sr3 =  std::real( sx );  require( sr3 == sx );
    float si3 =  std::imag( sx );  require( si3 == 0  );
    //float sc3 =  std::conj( sx );  require( sc3 == sx );  // compile error

    // Note: require( conj( dx ) == dx ) doesn't catch compile errors below
    // due to std::conj returning complex for all input types;
    // assigning to dc causes compile error.
    double dx = 3.141592653589793;
    double dr1 = real( dx );        require( dr1 == dx );
    double di1 = imag( dx );        require( di1 == 0  );
    double dc1 = conj( dx );        require( dc1 == dx );
    double dr2 = blas::real( dx );  require( dr2 == dx );
    double di2 = blas::imag( dx );  require( di2 == 0  );
    double dc2 = blas::conj( dx );  require( dc2 == dx );
    double dr3 =  std::real( dx );  require( dr3 == dx );
    double di3 =  std::imag( dx );  require( di3 == 0  );
    //double dc3 = std::conj( dx );  require( dc3 == dx );  // compile error

    float sxi = 2.718281828459045f;
    std::complex<float> cx( sx, sxi );
    std::complex<float> cx_( sx, -sxi );
    float               cr1 = real( cx );  require( cr1 == sx  );
    float               ci1 = imag( cx );  require( ci1 == sxi );
    std::complex<float> cc1 = conj( cx );  require( cc1 == cx_ );
    float               cr2 = blas::real( cx );  require( cr2 == sx  );
    float               ci2 = blas::imag( cx );  require( ci2 == sxi );
    // compile error: static assertion failed
    //std::complex<float> cc2 = blas::conj( cx );  require( cc2 == cx_ );
    float               cr3 =  std::real( cx );  require( cr3 == sx  );
    float               ci3 =  std::imag( cx );  require( ci3 == sxi );
    std::complex<float> cc3 =  std::conj( cx );  require( cc3 == cx_ );

    double dxi = 2.718281828459045;
    std::complex<double> zx( dx, dxi );
    std::complex<double> zx_( dx, -dxi );
    double               zr1 = real( zx );  require( zr1 == dx  );
    double               zi1 = imag( zx );  require( zi1 == dxi );
    std::complex<double> zc1 = conj( zx );  require( zc1 == zx_ );
    double               zr2 = blas::real( zx );  require( zr2 == dx  );
    double               zi2 = blas::imag( zx );  require( zi2 == dxi );
    // compile error: static assertion failed
    //std::complex<double> zc2 = blas::conj( zx );  require( zc2 == zx_ );
    double               zr3 =  std::real( zx );  require( zr3 == dx  );
    double               zi3 =  std::imag( zx );  require( zi3 == dxi );
    std::complex<double> zc3 =  std::conj( zx );  require( zc3 == zx_ );
}

// -----------------------------------------------------------------------------
void test_real_type()
{
    printf( "%s\n", __func__ );

    // Extra parens needed to avoid confusing preprocessor: require( (...) );
    require( (std::is_same< blas::real_type< float  >, float  >::value) );
    require( (std::is_same< blas::real_type< double >, double >::value) );
    require( (std::is_same< blas::real_type< std::complex<float>  >, float  >::value) );
    require( (std::is_same< blas::real_type< std::complex<double> >, double >::value) );

    // pairs
    require( (std::is_same< blas::real_type< float,  float  >, float  >::value) );
    require( (std::is_same< blas::real_type< float,  double >, double >::value) );
    require( (std::is_same< blas::real_type< double, double >, double >::value) );
    require( (std::is_same< blas::real_type< float,  std::complex<float>  >, float  >::value) );
    require( (std::is_same< blas::real_type< double, std::complex<float>  >, double >::value) );
    require( (std::is_same< blas::real_type< float,  std::complex<double> >, double >::value) );
    require( (std::is_same< blas::real_type< double, std::complex<double> >, double >::value) );
    require( (std::is_same< blas::real_type< std::complex<float>,  float  >, float  >::value) );
    require( (std::is_same< blas::real_type< std::complex<double>, float  >, double >::value) );
    require( (std::is_same< blas::real_type< std::complex<double>, double >, double >::value) );
    require( (std::is_same< blas::real_type< std::complex<float>, std::complex<double> >, double >::value) );

    // triples
    require( (std::is_same< blas::real_type< float, float,  float >, float  >::value) );
    require( (std::is_same< blas::real_type< float, double, float >, double >::value) );
    require( (std::is_same< blas::real_type< float, float,  std::complex<float>  >, float  >::value) );
    require( (std::is_same< blas::real_type< float, double, std::complex<double> >, double >::value) );
    require( (std::is_same< blas::real_type< float, double, std::complex<float>  >, double >::value) );
}

// -----------------------------------------------------------------------------
void test_complex_type()
{
    printf( "%s\n", __func__ );

    // Extra parens needed to avoid confusing preprocessor: require( (...) );
    require( (std::is_same< blas::complex_type< float  >, std::complex<float>  >::value) );
    require( (std::is_same< blas::complex_type< double >, std::complex<double> >::value) );
    require( (std::is_same< blas::complex_type< std::complex<float>  >, std::complex<float>  >::value) );
    require( (std::is_same< blas::complex_type< std::complex<double> >, std::complex<double> >::value) );

    // pairs
    require( (std::is_same< blas::complex_type< float,  float  >, std::complex<float>  >::value) );
    require( (std::is_same< blas::complex_type< float,  double >, std::complex<double> >::value) );
    require( (std::is_same< blas::complex_type< double, double >, std::complex<double> >::value) );
    require( (std::is_same< blas::complex_type< float,  std::complex<float>  >, std::complex<float>  >::value) );
    require( (std::is_same< blas::complex_type< double, std::complex<float>  >, std::complex<double> >::value) );
    require( (std::is_same< blas::complex_type< float,  std::complex<double> >, std::complex<double> >::value) );
    require( (std::is_same< blas::complex_type< double, std::complex<double> >, std::complex<double> >::value) );
    require( (std::is_same< blas::complex_type< std::complex<float>,  float  >, std::complex<float>  >::value) );
    require( (std::is_same< blas::complex_type< std::complex<double>, float  >, std::complex<double> >::value) );
    require( (std::is_same< blas::complex_type< std::complex<double>, double >, std::complex<double> >::value) );
    require( (std::is_same< blas::complex_type< std::complex<float>, std::complex<double> >, std::complex<double> >::value) );

    // triples
    require( (std::is_same< blas::complex_type< float, float,  float >, std::complex<float>  >::value) );
    require( (std::is_same< blas::complex_type< float, double, float >, std::complex<double> >::value) );
    require( (std::is_same< blas::complex_type< float, float,  std::complex<float>  >, std::complex<float>  >::value) );
    require( (std::is_same< blas::complex_type< float, double, std::complex<double> >, std::complex<double> >::value) );
    require( (std::is_same< blas::complex_type< float, double, std::complex<float>  >, std::complex<double> >::value) );
}

// -----------------------------------------------------------------------------
void test_scalar_type()
{
    printf( "%s\n", __func__ );

    // Extra parens needed to avoid confusing preprocessor: require( (...) );
    require( (std::is_same< blas::scalar_type< float  >, float  >::value) );
    require( (std::is_same< blas::scalar_type< double >, double >::value) );
    require( (std::is_same< blas::scalar_type< std::complex<float>  >, std::complex<float>  >::value) );
    require( (std::is_same< blas::scalar_type< std::complex<double> >, std::complex<double> >::value) );

    // pairs
    require( (std::is_same< blas::scalar_type< float,  float  >, float  >::value) );
    require( (std::is_same< blas::scalar_type< float,  double >, double >::value) );
    require( (std::is_same< blas::scalar_type< double, double >, double >::value) );
    require( (std::is_same< blas::scalar_type< float,  std::complex<float>  >, std::complex<float>  >::value) );
    require( (std::is_same< blas::scalar_type< double, std::complex<float>  >, std::complex<double> >::value) );
    require( (std::is_same< blas::scalar_type< float,  std::complex<double> >, std::complex<double> >::value) );
    require( (std::is_same< blas::scalar_type< double, std::complex<double> >, std::complex<double> >::value) );
    require( (std::is_same< blas::scalar_type< std::complex<float>,  float  >, std::complex<float>  >::value) );
    require( (std::is_same< blas::scalar_type< std::complex<double>, float  >, std::complex<double> >::value) );
    require( (std::is_same< blas::scalar_type< std::complex<double>, double >, std::complex<double> >::value) );
    require( (std::is_same< blas::scalar_type< std::complex<float>, std::complex<double> >, std::complex<double> >::value) );

    // triples
    require( (std::is_same< blas::scalar_type< float, float,  float >, float  >::value) );
    require( (std::is_same< blas::scalar_type< float, double, float >, double >::value) );
    require( (std::is_same< blas::scalar_type< float, float,  std::complex<float>  >, std::complex<float>  >::value) );
    require( (std::is_same< blas::scalar_type< float, double, std::complex<double> >, std::complex<double> >::value) );
    require( (std::is_same< blas::scalar_type< float, double, std::complex<float>  >, std::complex<double> >::value) );
}

// -----------------------------------------------------------------------------
void test_make_scalar()
{
    printf( "%s\n", __func__ );

    float  sxr = 3.141592653589793f;
    double dxr = 3.141592653589793;
    float  sxi = 2.718281828459045f;
    double dxi = 2.718281828459045;

    auto sx = blas::make_scalar<float> ( sxr, sxi );
    auto dx = blas::make_scalar<double>( dxr, dxi );
    auto cx = blas::make_scalar< std::complex<float>  >( sxr, sxi );
    auto zx = blas::make_scalar< std::complex<double> >( dxr, dxi );

    require( sx == sxr);
    require( dx == dxr );
    require( cx == std::complex<float> ( sxr, sxi ) );
    require( zx == std::complex<double>( dxr, dxi ) );
}

//------------------------------------------------------------------------------
/// Tests low-level wrappers around cuBLAS / rocBLAS functions, and
/// tests SYCL functions.
///
void test_device_routines()
{
    printf( "%s\n", __func__ );

    int repeat = 4;
    double t;
    int device_cnt;

    // Count GPU devices. Usually 1st call is expensive.
    for (int i = 0; i < repeat; ++i) {
        t = get_wtime();
        device_cnt = blas::get_device_count();
        t = get_wtime() - t;
        printf( "get_device_count = %d   %11.3f usec\n", device_cnt, t * 1e6 );
        require( device_cnt >= 0 );
    }
    printf( "\n" );

    const int MAX_DEVICES = 10;
    device_cnt = std::min( device_cnt, MAX_DEVICES );

    //----------------------------------------
    // CUDA and ROCm code. These have a BLAS handle that oneMKL doesn't.
    #if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )

        blas::Queue::stream_t streams[ MAX_DEVICES * repeat ];
        blas::Queue::handle_t handles[ MAX_DEVICES * repeat ];
        blas::Queue::event_t   events[ MAX_DEVICES * repeat ];

        // Set device.
        for (int i = 0; i < repeat; ++i) {
            for (int dev = 0; dev < device_cnt; ++dev) {
                t = get_wtime();
                blas::internal_set_device( dev );
                t = get_wtime() - t;
                printf( "set_device( %d )        %11.3f usec\n", dev, t * 1e6 );
            }
        }
        printf( "\n" );

        // Create streams.
        for (int dev = 0; dev < device_cnt; ++dev) {
            blas::internal_set_device( dev );
            for (int i = 0; i < repeat; ++i) {
                t = get_wtime();
                streams[ dev * repeat + i ] = blas::stream_create();
                t = get_wtime() - t;
                printf( "stream_create( %d )     %11.3f usec\n", dev, t * 1e6 );
            }
        }
        printf( "\n" );

        // Create handles
        for (int dev = 0; dev < device_cnt; ++dev) {
            blas::internal_set_device( dev );
            for (int i = 0; i < repeat; ++i) {
                t = get_wtime();
                handles[ dev * repeat + i ]
                    = blas::handle_create( streams[ dev * repeat + i ] );
                t = get_wtime() - t;
                printf( "handle_create( %d )     %11.3f usec\n", dev, t * 1e6 );
            }
        }
        printf( "\n" );

        // Set stream on handle.
        for (int dev = 0; dev < device_cnt; ++dev) {
            blas::internal_set_device( dev );
            for (int i = 0; i < repeat; ++i) {
                t = get_wtime();
                blas::handle_set_stream( handles[ dev * repeat + i ],
                                         streams[ dev * repeat + i ] );
                t = get_wtime() - t;
                printf( "handle_set_stream( %d ) %11.3f usec\n", dev, t * 1e6 );
            }
        }
        printf( "\n" );

        // Create events.
        for (int dev = 0; dev < device_cnt; ++dev) {
            blas::internal_set_device( dev );
            for (int i = 0; i < repeat; ++i) {
                t = get_wtime();
                events[ dev * repeat + i ] = blas::event_create();
                t = get_wtime() - t;
                printf( "event_create( %d )      %11.3f usec\n", dev, t * 1e6 );
            }
        }
        printf( "\n" );

        // Record events.
        for (int dev = 0; dev < device_cnt; ++dev) {
            blas::internal_set_device( dev );
            for (int i = 0; i < repeat; ++i) {
                t = get_wtime();
                blas::event_record( events[ dev * repeat + i ],
                                    streams[ dev * repeat + i ] );
                t = get_wtime() - t;
                printf( "event_record( %d )      %11.3f usec\n", dev, t * 1e6 );
            }
        }
        printf( "\n" );

        // Sync events.
        for (int dev = 0; dev < device_cnt; ++dev) {
            blas::internal_set_device( dev );
            for (int i = 0; i < repeat; ++i) {
                t = get_wtime();
                blas::stream_wait_event( streams[ dev * repeat + i ],
                                         events[ dev * repeat + i ], 0 );
                t = get_wtime() - t;
                printf( "stream_wait_event( %d ) %11.3f usec\n", dev, t * 1e6 );
            }
        }
        printf( "\n" );

        // Sync events again (already sync'd).
        for (int dev = 0; dev < device_cnt; ++dev) {
            blas::internal_set_device( dev );
            for (int i = 0; i < repeat; ++i) {
                t = get_wtime();
                blas::stream_wait_event( streams[ dev * repeat + i ],
                                         events[ dev * repeat + i ], 0 );
                t = get_wtime() - t;
                printf( "stream_wait_event( %d ) %11.3f usec\n", dev, t * 1e6 );
            }
        }
        printf( "\n" );

        // Destroy events.
        for (int dev = 0; dev < device_cnt; ++dev) {
            blas::internal_set_device( dev );
            for (int i = 0; i < repeat; ++i) {
                t = get_wtime();
                blas::event_destroy( events[ dev * repeat + i ] );
                t = get_wtime() - t;
                printf( "event_destroy( %d )     %11.3f usec\n", dev, t * 1e6 );
            }
        }
        printf( "\n" );

        // Destroy handles.
        for (int dev = 0; dev < device_cnt; ++dev) {
            blas::internal_set_device( dev );
            for (int i = 0; i < repeat; ++i) {
                t = get_wtime();
                blas::handle_destroy( handles[ dev * repeat + i ] );
                t = get_wtime() - t;
                printf( "handle_destroy( %d )    %11.3f usec\n", dev, t * 1e6 );
            }
        }
        printf( "\n" );

        // Destroy streams.
        for (int dev = 0; dev < device_cnt; ++dev) {
            blas::internal_set_device( dev );
            for (int i = 0; i < repeat; ++i) {
                t = get_wtime();
                blas::stream_destroy( streams[ dev * repeat + i ] );
                t = get_wtime() - t;
                printf( "stream_destroy( %d )    %11.3f usec\n", dev, t * 1e6 );
            }
        }
        printf( "\n" );

    //----------------------------------------
    #elif defined( BLAS_HAVE_SYCL )

        // stream_t === sycl::queue
        // Use pointer so we can time destructor.
        blas::Queue::stream_t* stream_ptrs[ MAX_DEVICES * repeat ];

        // Array of sycl::queues as destination for copy.
        // Default constructs the queues.
        t = get_wtime();
        blas::Queue::stream_t stream_copies[ MAX_DEVICES * repeat ];
        t = get_wtime() - t;
        printf( "sycl::queue()           %11.3f usec\n", t * 1e6 );

        // No have set_device.

        // Create streams (sycl::queues).
        for (int dev = 0; dev < device_cnt; ++dev) {
            for (int i = 0; i < repeat; ++i) {
                t = get_wtime();
                stream_ptrs[ dev * repeat + i ]
                    = new sycl::queue( blas::DeviceList::at( dev ) );
                t = get_wtime() - t;
                printf( "sycl::queue( %d )       %11.3f usec\n", dev, t * 1e6 );
            }
        }
        printf( "\n" );

        // SYCL has events, but with different API.
        // No handle_create, handle_set_stream, event_create, event_record,
        // stream_wait_event, event_destroy.

        // Copy streams (sycl::queues).
        for (int dev = 0; dev < device_cnt; ++dev) {
            for (int i = 0; i < repeat; ++i) {
                t = get_wtime();
                stream_copies[ dev * repeat + i ]
                    = *stream_ptrs[ dev * repeat + i ];
                t = get_wtime() - t;
                printf( "sycl::operator = ( %d ) %11.3f usec\n", dev, t * 1e6 );
            }
        }
        printf( "\n" );

        // Destroy streams (sycl::queues).
        for (int dev = 0; dev < device_cnt; ++dev) {
            for (int i = 0; i < repeat; ++i) {
                t = get_wtime();
                delete stream_ptrs[ dev * repeat + i ];
                stream_ptrs[ dev * repeat + i ] = nullptr;
                t = get_wtime() - t;
                printf( "sycl::~queue( %d )      %11.3f usec\n", dev, t * 1e6 );
            }
        }
        printf( "\n" );
    #endif // SYCL
}

//------------------------------------------------------------------------------
/// Tests Queue constructor.
///
void test_queue()
{
    printf( "%s\n", __func__ );

    int repeat = 4;
    int device_cnt = blas::get_device_count();
    double t;

    printf( "sizeof( Queue ) %ld bytes\n", sizeof( blas::Queue ) );

    // Create and destroy several queues on each device.
    std::vector< blas::Queue* > queues( device_cnt * repeat );
    for (int dev = 0; dev < device_cnt; ++dev) {
        for (int i = 0; i < repeat; ++i) {
            t = get_wtime();
            queues[ dev * repeat + i ] = new blas::Queue( dev );
            t = get_wtime() - t;
            printf( "blas::Queue( %d )       %11.3f usec\n", dev, t * 1e6 );
        }
    }
    printf( "\n" );

    // Destroy queues.
    for (int dev = 0; dev < device_cnt; ++dev) {
        for (int i = 0; i < repeat; ++i) {
            t = get_wtime();
            delete queues[ dev * repeat + i ];
            t = get_wtime() - t;
            printf( "delete queues[ %d ]     %11.3f usec\n", dev, t * 1e6 );
        }
    }
    printf( "\n" );
}

// -----------------------------------------------------------------------------
void test_queue_from_stream()
{
    printf( "%s\n", __func__ );

    #if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS ) || defined( BLAS_HAVE_SYCL )
        int repeat = 4;
        int device_cnt = blas::get_device_count();
        double t;

        const int MAX_DEVICES = 10;
        device_cnt = std::min( device_cnt, MAX_DEVICES );
        blas::Queue::stream_t streams[ MAX_DEVICES ];
    #endif

    // CUDA and ROCm code. These have a BLAS HANDLE that SYCL doesn't.
    #if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )

        blas::Queue::handle_t handles[ MAX_DEVICES ];

        // Create streams and handles.
        for (int dev = 0; dev < device_cnt; ++dev) {
            blas::internal_set_device( dev );
            streams[ dev ] = blas::stream_create();
            handles[ dev ] = blas::handle_create( streams[ dev ] );
        }

        std::vector< blas::Queue* > queues( device_cnt * repeat );

        // Create Queues from streams.
        for (int dev = 0; dev < device_cnt; ++dev) {
            for (int i = 0; i < repeat; ++i) {
                t = get_wtime();
                queues[ dev * repeat + i ] = new blas::Queue( dev, streams[ dev ] );
                t = get_wtime() - t;
                printf( "blas::Queue( %d, stream )  %11.3f usec\n",
                        dev, t * 1e6 );
            }
        }
        printf( "\n" );

        for (int dev = 0; dev < device_cnt; ++dev) {
            for (int i = 0; i < repeat; ++i) {
                t = get_wtime();
                delete queues[ dev * repeat + i ];
                t = get_wtime() - t;
                printf( "delete queues[ %d ]        %11.3f usec\n", dev, t * 1e6 );
            }
        }
        printf( "\n" );

        // Create Queues from handles.
        for (int dev = 0; dev < device_cnt; ++dev) {
            for (int i = 0; i < repeat; ++i) {
                t = get_wtime();
                queues[ dev * repeat + i ] = new blas::Queue( dev, handles[ dev ] );
                t = get_wtime() - t;
                printf( "blas::Queue( %d, handle )  %11.3f usec\n",
                        dev, t * 1e6 );
            }
        }
        printf( "\n" );

        for (int dev = 0; dev < device_cnt; ++dev) {
            for (int i = 0; i < repeat; ++i) {
                t = get_wtime();
                delete queues[ dev * repeat + i ];
                t = get_wtime() - t;
                printf( "delete queues[ %d ]        %11.3f usec\n", dev, t * 1e6 );
            }
        }
        printf( "\n" );

        // Destroy streams and handles.
        for (int dev = 0; dev < device_cnt; ++dev) {
            blas::internal_set_device( dev );
            blas::handle_destroy( handles[ dev ] );
            blas::stream_destroy( streams[ dev ] );
        }

    #elif defined( BLAS_HAVE_SYCL )

        // Create streams (sycl::queues)
        for (int dev = 0; dev < device_cnt; ++dev) {
            streams[ dev ] = sycl::queue( blas::DeviceList::at( dev ) );
        }

        std::vector< blas::Queue* > queues( device_cnt * repeat );

        // Create Queues from streams.
        for (int dev = 0; dev < device_cnt; ++dev) {
            for (int i = 0; i < repeat; ++i) {
                t = get_wtime();
                queues[ dev * repeat + i ] = new blas::Queue( dev, streams[ dev ] );
                t = get_wtime() - t;
                printf( "blas::Queue( %d, stream )  %11.3f usec\n",
                        dev, t * 1e6 );
            }
        }
        printf( "\n" );

        for (int dev = 0; dev < device_cnt; ++dev) {
            for (int i = 0; i < repeat; ++i) {
                t = get_wtime();
                delete queues[ dev * repeat + i ];
                t = get_wtime() - t;
                printf( "delete queues[ %d ]        %11.3f usec\n", dev, t * 1e6 );
            }
        }
        printf( "\n" );

        // sycl::queues destroyed implicitly.

    #endif  // BLAS_HAVE_SYCL
}

// -----------------------------------------------------------------------------
void test_queue_fork()
{
    using blas::MaxForkSize;

    printf( "%s\n", __func__ );

    int device_cnt = blas::get_device_count();
    if (device_cnt == 0)
        return;

    int dev = 0;

    float alpha = 3.1415;
    int n = 1000;
    int cnt = 30;

    std::vector<int> num_streams_test { 1, 3, MaxForkSize };
    for (int num_streams : num_streams_test) {
        printf( "queue( dev %d )\n", dev );
        blas::Queue queue( dev );

        float* x = blas::device_malloc<float>( n, queue );

        auto orig_stream = queue.stream();
        #if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )
            auto handle = queue.handle();
        #endif

        std::vector< blas::Queue::stream_t > streams( num_streams );

        for (int iter = 0; iter < 3; ++iter) {
            //-----
            // Fork with num_streams; MaxForkSize is default.
            //printf( "fork( %d )\n", num_streams );
            double time_fork = get_wtime();
            if (num_streams == MaxForkSize)
                queue.fork();
            else
                queue.fork( num_streams );
            time_fork = get_wtime() - time_fork;

            // Get streams after the first fork.
            if (iter == 0) {
                for (int i = 0; i < num_streams; ++i) {
                    streams[ i ] = queue.stream();
                    queue.revolve();
                }
            }

            // Launch kernel on streams that cycle every num_streams times.
            for (int i = 0; i < cnt; ++i) {
                //printf( "i %2d, stream %p\n", i, (void*) queue.stream() );
                require( queue.stream() == streams[ i % num_streams ] );
                #if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )
                    require( queue.handle() == handle );
                    require( blas::handle_get_stream( handle ) == queue.stream() );
                #endif
                blas::scal( n, alpha, x, 1, queue );
                queue.revolve();
            }

            // This join time includes time to execute blas::scal.
            double time_join = get_wtime();
            queue.join();
            time_join = get_wtime() - time_join;

            // Join should restore original stream.
            require( queue.stream() == orig_stream );
            #if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )
                require( blas::handle_get_stream( handle ) == orig_stream );
            #endif

            // Fork again, then immediately join.
            double time_fork2 = get_wtime();
            if (num_streams == MaxForkSize)
                queue.fork();
            else
                queue.fork( num_streams );
            time_fork2 = get_wtime() - time_fork2;

            double time_join2 = get_wtime();
            queue.join();
            time_join2 = get_wtime() - time_join2;

            printf( "fork( %d ) %11.3f usec, join %11.3f usec (calls blas::scal)\n",
                    num_streams, time_fork * 1e6, time_join * 1e6 );
            printf( "fork( %d ) %11.3f usec, join %11.3f usec (no call)\n",
                    num_streams, time_fork2 * 1e6, time_join2 * 1e6 );
        }
        printf( "After first one, fork should be faster.\n" );

        queue.sync();

        blas::device_free( x, queue );
        x = nullptr;

        // queue is destroyed here.
        printf( "    ~queue\n" );
    }
}

// -----------------------------------------------------------------------------
void test_util( Params& params, bool run )
{
    int64_t m = params.dim.m();
    (void)m; // unused; just so `./tester --dim 100 util` works.

    if (! run)
        return;

    static bool first = true;
    if (first) {
        first = false;

        test_enums();
        test_exceptions();
        test_abs1();
        test_is_complex();
        test_real_imag_conj();
        test_real_type();
        test_complex_type();
        test_scalar_type();
        test_scalar_type();
        test_make_scalar();

        // GPU routines
        test_device_routines();
        test_queue();
        test_queue_from_stream();
        test_queue_fork();
        printf( "\n" );
    }

    params.okay() = true;
}
