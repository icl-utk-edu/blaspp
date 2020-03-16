// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"

#include <string>

// -----------------------------------------------------------------------------
void test_enums()
{
    using str = std::string;

    assert( blas::layout2char( blas::Layout::ColMajor ) == 'C' );
    assert( blas::layout2char( blas::Layout::RowMajor ) == 'R' );
    assert( blas::layout2str( blas::Layout::ColMajor ) == str("col") );
    assert( blas::layout2str( blas::Layout::RowMajor ) == str("row") );
    assert( blas::char2layout( 'C' ) == blas::Layout::ColMajor );
    assert( blas::char2layout( 'c' ) == blas::Layout::ColMajor );
    assert( blas::char2layout( 'R' ) == blas::Layout::RowMajor );
    assert( blas::char2layout( 'r' ) == blas::Layout::RowMajor );

    assert( blas::op2char( blas::Op::NoTrans   ) == 'N' );
    assert( blas::op2char( blas::Op::Trans     ) == 'T' );
    assert( blas::op2char( blas::Op::ConjTrans ) == 'C' );
    assert( blas::op2str( blas::Op::NoTrans   ) == str("notrans") );
    assert( blas::op2str( blas::Op::Trans     ) == str("trans") );
    assert( blas::op2str( blas::Op::ConjTrans ) == str("conj") );
    assert( blas::char2op( 'N' ) == blas::Op::NoTrans   );
    assert( blas::char2op( 'n' ) == blas::Op::NoTrans   );
    assert( blas::char2op( 'T' ) == blas::Op::Trans     );
    assert( blas::char2op( 't' ) == blas::Op::Trans     );
    assert( blas::char2op( 'C' ) == blas::Op::ConjTrans );
    assert( blas::char2op( 'c' ) == blas::Op::ConjTrans );

    assert( blas::uplo2char( blas::Uplo::Upper   ) == 'U' );
    assert( blas::uplo2char( blas::Uplo::Lower   ) == 'L' );
    assert( blas::uplo2char( blas::Uplo::General ) == 'G' );
    assert( blas::uplo2str( blas::Uplo::Upper   ) == str("upper") );
    assert( blas::uplo2str( blas::Uplo::Lower   ) == str("lower") );
    assert( blas::uplo2str( blas::Uplo::General ) == str("general") );
    assert( blas::char2uplo( 'U' ) == blas::Uplo::Upper   );
    assert( blas::char2uplo( 'u' ) == blas::Uplo::Upper   );
    assert( blas::char2uplo( 'L' ) == blas::Uplo::Lower   );
    assert( blas::char2uplo( 'l' ) == blas::Uplo::Lower   );
    assert( blas::char2uplo( 'G' ) == blas::Uplo::General );
    assert( blas::char2uplo( 'g' ) == blas::Uplo::General );

    assert( blas::diag2char( blas::Diag::NonUnit ) == 'N' );
    assert( blas::diag2char( blas::Diag::Unit    ) == 'U' );
    assert( blas::diag2str( blas::Diag::NonUnit ) == str("nonunit") );
    assert( blas::diag2str( blas::Diag::Unit    ) == str("unit") );
    assert( blas::char2diag( 'N' ) == blas::Diag::NonUnit );
    assert( blas::char2diag( 'n' ) == blas::Diag::NonUnit );
    assert( blas::char2diag( 'U' ) == blas::Diag::Unit    );
    assert( blas::char2diag( 'u' ) == blas::Diag::Unit    );

    assert( blas::side2char( blas::Side::Left  ) == 'L' );
    assert( blas::side2char( blas::Side::Right ) == 'R' );
    assert( blas::side2str( blas::Side::Left  ) == str("left") );
    assert( blas::side2str( blas::Side::Right ) == str("right") );
    assert( blas::char2side( 'L' ) == blas::Side::Left  );
    assert( blas::char2side( 'l' ) == blas::Side::Left  );
    assert( blas::char2side( 'R' ) == blas::Side::Right );
    assert( blas::char2side( 'r' ) == blas::Side::Right );
}

// -----------------------------------------------------------------------------
void test_exceptions()
{
    using str = std::string;

    try {
        throw blas::Error();
    }
    catch (blas::Error const& ex) {
        assert( ex.what() == str("") );
    }
    catch (...) {
        assert( false );
    }

    //--------------------
    // Error inherits from std::exception
    try {
        throw blas::Error();
    }
    catch (std::exception const& ex) {
        assert( ex.what() == str("") );
    }
    catch (...) {
        assert( false );
    }

    //--------------------
    // Error can take message.
    try {
        throw blas::Error( "message" );
    }
    catch (std::exception const& ex) {
        assert( ex.what() == str("message") );
    }
    catch (...) {
        assert( false );
    }

    //--------------------
    // Error can take message and function.
    try {
        throw blas::Error( "message", __func__ );
    }
    catch (std::exception const& ex) {
        assert( ex.what() == str("message, in function test_exceptions") );
    }
    catch (...) {
        assert( false );
    }
}

// -----------------------------------------------------------------------------
void test_abs1()
{
    float sx = -3.141592653589793f;
    assert( blas::abs1( sx ) == -sx );

    double dx = -3.141592653589793;
    assert( blas::abs1( dx ) == -dx );

    float sxi = -2.718281828459045f;
    std::complex<float> cx( sx, sxi );
    assert( blas::abs1( cx ) == -sx + -sxi );

    double dxi = -2.718281828459045;
    std::complex<double> zx( dx, dxi );
    assert( blas::abs1( zx ) == -dx + -dxi );
}

// -----------------------------------------------------------------------------
void test_is_complex()
{
    assert( ! blas::is_complex< int >::value );
    assert( ! blas::is_complex< int64_t >::value );
    assert( ! blas::is_complex< float >::value );
    assert( ! blas::is_complex< double >::value );
    assert( blas::is_complex< std::complex<float> >::value );
    assert( blas::is_complex< std::complex<double> >::value );
}

// -----------------------------------------------------------------------------
void test_real_imag_conj()
{
    using blas::real;  // same as std::real
    using blas::imag;  // same as std::imag
    using blas::conj;  // different than std::conj for real types

    int ix = 1234;
    int ir1 = real( ix );  assert( ir1 == ix );
    int ii1 = imag( ix );  assert( ii1 == 0  );
    int ic1 = conj( ix );  assert( ic1 == ix );
    int ir2 = blas::real( ix );  assert( ir2 == ix );
    int ii2 = blas::imag( ix );  assert( ii2 == 0  );
    int ic2 = blas::conj( ix );  assert( ic2 == ix );
    int ir3 =  std::real( ix );  assert( ir3 == ix );
    int ii3 =  std::imag( ix );  assert( ii3 == 0  );
    //int ic3 =  std::conj( ix );  assert( ic3 == ix );  // compile error

    float sx = 3.141592653589793f;
    float sr1 = real( sx );  assert( sr1 == sx );
    float si1 = imag( sx );  assert( si1 == 0  );
    float sc1 = conj( sx );  assert( sc1 == sx );
    float sr2 = blas::real( sx );  assert( sr2 == sx );
    float si2 = blas::imag( sx );  assert( si2 == 0  );
    float sc2 = blas::conj( sx );  assert( sc2 == sx );
    float sr3 =  std::real( sx );  assert( sr3 == sx );
    float si3 =  std::imag( sx );  assert( si3 == 0  );
    //float sc3 =  std::conj( sx );  assert( sc3 == sx );  // compile error

    // Note: assert( conj( dx ) == dx ) doesn't catch compile errors below
    // due to std::conj returning complex for all input types;
    // assigning to dc causes compile error.
    double dx = 3.141592653589793;
    double dr1 = real( dx );        assert( dr1 == dx );
    double di1 = imag( dx );        assert( di1 == 0  );
    double dc1 = conj( dx );        assert( dc1 == dx );
    double dr2 = blas::real( dx );  assert( dr2 == dx );
    double di2 = blas::imag( dx );  assert( di2 == 0  );
    double dc2 = blas::conj( dx );  assert( dc2 == dx );
    double dr3 =  std::real( dx );  assert( dr3 == dx );
    double di3 =  std::imag( dx );  assert( di3 == 0  );
    //double dc3 = std::conj( dx );  assert( dc3 == dx );  // compile error

    float sxi = 2.718281828459045f;
    std::complex<float> cx( sx, sxi );
    std::complex<float> cx_( sx, -sxi );
    float               cr1 = real( cx );  assert( cr1 == sx  );
    float               ci1 = imag( cx );  assert( ci1 == sxi );
    std::complex<float> cc1 = conj( cx );  assert( cc1 == cx_ );
    float               cr2 = blas::real( cx );  assert( cr2 == sx  );
    float               ci2 = blas::imag( cx );  assert( ci2 == sxi );
    // compile error: static assertion failed
    //std::complex<float> cc2 = blas::conj( cx );  assert( cc2 == cx_ );
    float               cr3 =  std::real( cx );  assert( cr3 == sx  );
    float               ci3 =  std::imag( cx );  assert( ci3 == sxi );
    std::complex<float> cc3 =  std::conj( cx );  assert( cc3 == cx_ );

    double dxi = 2.718281828459045;
    std::complex<double> zx( dx, dxi );
    std::complex<double> zx_( dx, -dxi );
    double               zr1 = real( zx );  assert( zr1 == dx  );
    double               zi1 = imag( zx );  assert( zi1 == dxi );
    std::complex<double> zc1 = conj( zx );  assert( zc1 == zx_ );
    double               zr2 = blas::real( zx );  assert( zr2 == dx  );
    double               zi2 = blas::imag( zx );  assert( zi2 == dxi );
    // compile error: static assertion failed
    //std::complex<double> zc2 = blas::conj( zx );  assert( zc2 == zx_ );
    double               zr3 =  std::real( zx );  assert( zr3 == dx  );
    double               zi3 =  std::imag( zx );  assert( zi3 == dxi );
    std::complex<double> zc3 =  std::conj( zx );  assert( zc3 == zx_ );
}

// -----------------------------------------------------------------------------
void test_real_type()
{
    // For some reason, assert( std::is_same<...>::value ) won't compile:
    // error: macro "assert" passed 2 arguments, but takes just 1

    bool ok = true;
    ok = ok && std::is_same< blas::real_type< float  >, float  >::value;
    ok = ok && std::is_same< blas::real_type< double >, double >::value;
    ok = ok && std::is_same< blas::real_type< std::complex<float>  >, float  >::value;
    ok = ok && std::is_same< blas::real_type< std::complex<double> >, double >::value;

    // pairs
    ok = ok && std::is_same< blas::real_type< float,  float  >, float  >::value;
    ok = ok && std::is_same< blas::real_type< float,  double >, double >::value;
    ok = ok && std::is_same< blas::real_type< double, double >, double >::value;
    ok = ok && std::is_same< blas::real_type< float,  std::complex<float>  >, float  >::value;
    ok = ok && std::is_same< blas::real_type< double, std::complex<float>  >, double >::value;
    ok = ok && std::is_same< blas::real_type< float,  std::complex<double> >, double >::value;
    ok = ok && std::is_same< blas::real_type< double, std::complex<double> >, double >::value;
    ok = ok && std::is_same< blas::real_type< std::complex<float>,  float  >, float  >::value;
    ok = ok && std::is_same< blas::real_type< std::complex<double>, float  >, double >::value;
    ok = ok && std::is_same< blas::real_type< std::complex<double>, double >, double >::value;
    ok = ok && std::is_same< blas::real_type< std::complex<float>, std::complex<double> >, double >::value;

    // triples
    ok = ok && std::is_same< blas::real_type< float, float,  float >, float  >::value;
    ok = ok && std::is_same< blas::real_type< float, double, float >, double >::value;
    ok = ok && std::is_same< blas::real_type< float, float,  std::complex<float>  >, float  >::value;
    ok = ok && std::is_same< blas::real_type< float, double, std::complex<double> >, double >::value;
    ok = ok && std::is_same< blas::real_type< float, double, std::complex<float>  >, double >::value;

    assert( ok );
}

// -----------------------------------------------------------------------------
void test_complex_type()
{
    // For some reason, assert( std::is_same<...>::value ) won't compile:
    // error: macro "assert" passed 2 arguments, but takes just 1

    bool ok = true;
    ok = ok && std::is_same< blas::complex_type< float  >, std::complex<float>  >::value;
    ok = ok && std::is_same< blas::complex_type< double >, std::complex<double> >::value;
    ok = ok && std::is_same< blas::complex_type< std::complex<float>  >, std::complex<float>  >::value;
    ok = ok && std::is_same< blas::complex_type< std::complex<double> >, std::complex<double> >::value;

    // pairs
    ok = ok && std::is_same< blas::complex_type< float,  float  >, std::complex<float>  >::value;
    ok = ok && std::is_same< blas::complex_type< float,  double >, std::complex<double> >::value;
    ok = ok && std::is_same< blas::complex_type< double, double >, std::complex<double> >::value;
    ok = ok && std::is_same< blas::complex_type< float,  std::complex<float>  >, std::complex<float>  >::value;
    ok = ok && std::is_same< blas::complex_type< double, std::complex<float>  >, std::complex<double> >::value;
    ok = ok && std::is_same< blas::complex_type< float,  std::complex<double> >, std::complex<double> >::value;
    ok = ok && std::is_same< blas::complex_type< double, std::complex<double> >, std::complex<double> >::value;
    ok = ok && std::is_same< blas::complex_type< std::complex<float>,  float  >, std::complex<float>  >::value;
    ok = ok && std::is_same< blas::complex_type< std::complex<double>, float  >, std::complex<double> >::value;
    ok = ok && std::is_same< blas::complex_type< std::complex<double>, double >, std::complex<double> >::value;
    ok = ok && std::is_same< blas::complex_type< std::complex<float>, std::complex<double> >, std::complex<double> >::value;

    // triples
    ok = ok && std::is_same< blas::complex_type< float, float,  float >, std::complex<float>  >::value;
    ok = ok && std::is_same< blas::complex_type< float, double, float >, std::complex<double> >::value;
    ok = ok && std::is_same< blas::complex_type< float, float,  std::complex<float>  >, std::complex<float>  >::value;
    ok = ok && std::is_same< blas::complex_type< float, double, std::complex<double> >, std::complex<double> >::value;
    ok = ok && std::is_same< blas::complex_type< float, double, std::complex<float>  >, std::complex<double> >::value;

    assert( ok );
}

// -----------------------------------------------------------------------------
void test_scalar_type()
{
    // For some reason, assert( std::is_same<...>::value ) won't compile:
    // error: macro "assert" passed 2 arguments, but takes just 1

    bool ok = true;
    ok = ok && std::is_same< blas::scalar_type< float  >, float  >::value;
    ok = ok && std::is_same< blas::scalar_type< double >, double >::value;
    ok = ok && std::is_same< blas::scalar_type< std::complex<float>  >, std::complex<float>  >::value;
    ok = ok && std::is_same< blas::scalar_type< std::complex<double> >, std::complex<double> >::value;

    // pairs
    ok = ok && std::is_same< blas::scalar_type< float,  float  >, float  >::value;
    ok = ok && std::is_same< blas::scalar_type< float,  double >, double >::value;
    ok = ok && std::is_same< blas::scalar_type< double, double >, double >::value;
    ok = ok && std::is_same< blas::scalar_type< float,  std::complex<float>  >, std::complex<float>  >::value;
    ok = ok && std::is_same< blas::scalar_type< double, std::complex<float>  >, std::complex<double> >::value;
    ok = ok && std::is_same< blas::scalar_type< float,  std::complex<double> >, std::complex<double> >::value;
    ok = ok && std::is_same< blas::scalar_type< double, std::complex<double> >, std::complex<double> >::value;
    ok = ok && std::is_same< blas::scalar_type< std::complex<float>,  float  >, std::complex<float>  >::value;
    ok = ok && std::is_same< blas::scalar_type< std::complex<double>, float  >, std::complex<double> >::value;
    ok = ok && std::is_same< blas::scalar_type< std::complex<double>, double >, std::complex<double> >::value;
    ok = ok && std::is_same< blas::scalar_type< std::complex<float>, std::complex<double> >, std::complex<double> >::value;

    // triples
    ok = ok && std::is_same< blas::scalar_type< float, float,  float >, float  >::value;
    ok = ok && std::is_same< blas::scalar_type< float, double, float >, double >::value;
    ok = ok && std::is_same< blas::scalar_type< float, float,  std::complex<float>  >, std::complex<float>  >::value;
    ok = ok && std::is_same< blas::scalar_type< float, double, std::complex<double> >, std::complex<double> >::value;
    ok = ok && std::is_same< blas::scalar_type< float, double, std::complex<float>  >, std::complex<double> >::value;

    assert( ok );
}

// -----------------------------------------------------------------------------
void test_make_scalar()
{
    float  sxr = 3.141592653589793f;
    double dxr = 3.141592653589793;
    float  sxi = 2.718281828459045f;
    double dxi = 2.718281828459045;

    auto sx = blas::make_scalar<float> ( sxr, sxi );
    auto dx = blas::make_scalar<double>( dxr, dxi );
    auto cx = blas::make_scalar< std::complex<float>  >( sxr, sxi );
    auto zx = blas::make_scalar< std::complex<double> >( dxr, dxi );

    assert( sx == sxr );
    assert( dx == dxr );
    assert( cx == std::complex<float> ( sxr, sxi ) );
    assert( zx == std::complex<double>( dxr, dxi ) );
}

// -----------------------------------------------------------------------------
void test_util( Params& params, bool run )
{
    int64_t m = params.dim.m();
    (void)m; // unused; just so `./tester --dim 100 util` works.

    if (! run)
        return;

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

    params.okay() = true;
}
