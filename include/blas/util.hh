// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_UTIL_HH
#define BLAS_UTIL_HH

#include <exception>
#include <complex>
#include <cstdarg>
#include <limits>
#include <vector>
#include <algorithm>

#include <assert.h>

namespace blas {

/// Use to silence compiler warning of unused variable.
#define blas_unused( var ) ((void)var)

// For printf, int64_t could be long (%ld), which is >= 32 bits,
// or long long (%lld), guaranteed >= 64 bits.
// Cast to llong to ensure printing 64 bits.
using llong = long long;

//------------------------------------------------------------------------------
/// Exception class for BLAS errors.
class Error: public std::exception {
public:
    /// Constructs BLAS error
    Error():
        std::exception()
    {}

    /// Constructs BLAS error with message
    Error( std::string const& msg ):
        std::exception(),
        msg_( msg )
    {}

    /// Constructs BLAS error with message: "msg, in function func"
    Error( const char* msg, const char* func ):
        std::exception(),
        msg_( std::string(msg) + ", in function " + func )
    {}

    /// Returns BLAS error message
    virtual const char* what() const noexcept override
        { return msg_.c_str(); }

private:
    std::string msg_;
};

// -----------------------------------------------------------------------------
enum class Layout : char { ColMajor = 'C', RowMajor = 'R' };
enum class Op     : char { NoTrans  = 'N', Trans    = 'T', ConjTrans = 'C' };
enum class Uplo   : char { Upper    = 'U', Lower    = 'L', General   = 'G' };
enum class Diag   : char { NonUnit  = 'N', Unit     = 'U' };
enum class Side   : char { Left     = 'L', Right    = 'R' };

extern const char* Layout_help;
extern const char* Op_help;
extern const char* Uplo_help;
extern const char* Diag_help;
extern const char* Side_help;

// -----------------------------------------------------------------------------
// Convert enum to LAPACK-style char.

inline char to_char( Layout value ) { return char( value ); }
inline char to_char( Op     value ) { return char( value ); }
inline char to_char( Uplo   value ) { return char( value ); }
inline char to_char( Diag   value ) { return char( value ); }
inline char to_char( Side   value ) { return char( value ); }

//------------------------------------------------------------------------------
// Convert enum to LAPACK-style C string (const char*).

inline const char* to_c_string( Layout value )
{
    switch (value) {
        case Layout::ColMajor: return "col";
        case Layout::RowMajor: return "row";
    }
    return "?";
}

inline const char* to_c_string( Op value )
{
    switch (value) {
        case Op::NoTrans:   return "notrans";
        case Op::Trans:     return "trans";
        case Op::ConjTrans: return "conj";
    }
    return "?";
}

inline const char* to_c_string( Uplo value )
{
    switch (value) {
        case Uplo::Lower:   return "lower";
        case Uplo::Upper:   return "upper";
        case Uplo::General: return "general";
    }
    return "?";
}

inline const char* to_c_string( Diag value )
{
    switch (value) {
        case Diag::NonUnit: return "nonunit";
        case Diag::Unit:    return "unit";
    }
    return "?";
}

inline const char* to_c_string( Side value )
{
    switch (value) {
        case Side::Left:  return "left";
        case Side::Right: return "right";
    }
    return "?";
}

//------------------------------------------------------------------------------
// Convert enum to LAPACK-style C++ string.

inline std::string to_string( Layout value )
{
    return to_c_string( value );
}

inline std::string to_string( Op value )
{
    return to_c_string( value );
}

inline std::string to_string( Uplo value )
{
    return to_c_string( value );
}

inline std::string to_string( Diag value )
{
    return to_c_string( value );
}

inline std::string to_string( Side value )
{
    return to_c_string( value );
}

//------------------------------------------------------------------------------
// Convert LAPACK-style char or string to enum.

inline void from_string( std::string const& str, Layout* val )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );
    if (str_ == "c" || str_ == "colmajor")
        *val = Layout::ColMajor;
    else if (str_ == "r" || str_ == "rowmajor")
        *val = Layout::RowMajor;
    else
        throw Error( "unknown Layout: " + str );
}

inline void from_string( std::string const& str, Op* val )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );
    if (str_ == "n" || str_ == "notrans")
        *val = Op::NoTrans;
    else if (str_ == "t" || str_ == "trans")
        *val = Op::Trans;
    else if (str_ == "c" || str_ == "conjtrans")
        *val = Op::ConjTrans;
    else
        throw Error( "unknown Op: " + str );
}

inline void from_string( std::string const& str, Uplo* val )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );
    if (str_ == "l" || str_ == "lower")
        *val = Uplo::Lower;
    else if (str_ == "u" || str_ == "upper")
        *val = Uplo::Upper;
    else if (str_ == "g" || str_ == "general")
        *val = Uplo::General;
    else
        throw Error( "unknown Uplo: " + str );
}

inline void from_string( std::string const& str, Diag* val )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );
    if (str_ == "n" || str_ == "nonunit")
        *val = Diag::NonUnit;
    else if (str_ == "u" || str_ == "unit")
        *val = Diag::Unit;
    else
        throw Error( "unknown Diag: " + str );
}

inline void from_string( std::string const& str, Side* val )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );
    if (str_ == "l" || str_ == "left")
        *val = Side::Left;
    else if (str_ == "r" || str_ == "right")
        *val = Side::Right;
    else
        throw Error( "unknown Side: " + str );
}

// -----------------------------------------------------------------------------
// 1-norm absolute value, |Re(x)| + |Im(x)|
template <typename T>
T abs1( T x )
{
    using std::abs;
    return abs( x );
}

template <typename T>
T abs1( std::complex<T> x )
{
    using std::abs;
    return abs( real( x ) ) + abs( imag( x ) );
}

// -----------------------------------------------------------------------------
// common_type_t is defined in C++14; here's a C++11 definition
#if __cplusplus >= 201402L
    using std::common_type_t;
    using std::decay_t;
#else
    template <typename... Ts>
    using common_type_t = typename std::common_type< Ts... >::type;

    template <typename... Ts>
    using decay_t = typename std::decay< Ts... >::type;
#endif

//------------------------------------------------------------------------------
/// True if T is std::complex<T2> for some type T2.
template <typename T>
struct is_complex:
    std::integral_constant<bool, false>
{};

// specialize for std::complex
template <typename T>
struct is_complex< std::complex<T> >:
    std::integral_constant<bool, true>
{};

template <typename T>
constexpr bool is_complex_v = is_complex<T>::value;

// -----------------------------------------------------------------------------
// Previously extended real and imag to real types. Belatedly discovered that
// C++11 extends std::real and std::imag to float and integer types,
// so just use those now.
using std::real;
using std::imag;

/// Extend conj to real datatypes.
/// For real T, this returns type T, whereas C++11 returns complex<T>.
/// Usage:
///     using blas::conj;
///     scalar_t x = ...
///     scalar_t y = conj( x );
/// That will use std::conj for complex types, and blas::conj for other types.
/// This prohibits complex types; it can't be called as y = blas::conj( x ).
///
template <typename T>
inline T conj( T x )
{
    static_assert(
        ! is_complex_v<T>,
        "Usage: using blas::conj; y = conj(x); NOT: y = blas::conj(x);" );
    return x;
}

// -----------------------------------------------------------------------------
// Based on C++14 common_type implementation from
// http://www.cplusplus.com/reference/type_traits/common_type/
// Adds promotion of complex types based on the common type of the associated
// real types. This fixes various cases:
//
// std::common_type_t< double, complex<float> > is complex<float>  (wrong)
//        scalar_type< double, complex<float> > is complex<double> (right)
//
// std::common_type_t< int, complex<long> > is not defined (compile error)
//        scalar_type< int, complex<long> > is complex<long> (right)

// for zero types
template <typename... Types>
struct scalar_type_traits;

// define scalar_type<> type alias
template <typename... Types>
using scalar_type = typename scalar_type_traits< Types... >::type;

// for one type
template <typename T>
struct scalar_type_traits< T >
{
    using type = decay_t<T>;
};

// for two types
// relies on type of ?: operator being the common type of its two arguments
template <typename T1, typename T2>
struct scalar_type_traits< T1, T2 >
{
    using type = decay_t< decltype( true ? std::declval<T1>() : std::declval<T2>() ) >;
};

// for either or both complex,
// find common type of associated real types, then add complex
template <typename T1, typename T2>
struct scalar_type_traits< std::complex<T1>, T2 >
{
    using type = std::complex< common_type_t< T1, T2 > >;
};

template <typename T1, typename T2>
struct scalar_type_traits< T1, std::complex<T2> >
{
    using type = std::complex< common_type_t< T1, T2 > >;
};

template <typename T1, typename T2>
struct scalar_type_traits< std::complex<T1>, std::complex<T2> >
{
    using type = std::complex< common_type_t< T1, T2 > >;
};

// for three or more types
template <typename T1, typename T2, typename... Types>
struct scalar_type_traits< T1, T2, Types... >
{
    using type = scalar_type< scalar_type< T1, T2 >, Types... >;
};

// -----------------------------------------------------------------------------
// for any combination of types, determine associated real, scalar,
// and complex types.
//
// real_type< float >                               is float
// real_type< float, double, complex<float> >       is double
//
// scalar_type< float >                             is float
// scalar_type< float, complex<float> >             is complex<float>
// scalar_type< float, double, complex<float> >     is complex<double>
//
// complex_type< float >                            is complex<float>
// complex_type< float, double >                    is complex<double>
// complex_type< float, double, complex<float> >    is complex<double>

// for zero types
template <typename... Types>
struct real_type_traits;

// define real_type<> type alias
template <typename... Types>
using real_type = typename real_type_traits< Types... >::real_t;

// define complex_type<> type alias
template <typename... Types>
using complex_type = std::complex< real_type< Types... > >;

// for one type
template <typename T>
struct real_type_traits<T>
{
    using real_t = T;
};

// for one complex type, strip complex
template <typename T>
struct real_type_traits< std::complex<T> >
{
    using real_t = T;
};

// for two or more types
template <typename T1, typename... Types>
struct real_type_traits< T1, Types... >
{
    using real_t = scalar_type< real_type<T1>, real_type< Types... > >;
};

// -----------------------------------------------------------------------------
// max that works with different data types: int64_t = max( int, int64_t )
// and any number of arguments: max( a, b, c, d )

// one argument
template <typename T>
T max( T x )
{
    return x;
}

// two arguments
template <typename T1, typename T2>
scalar_type< T1, T2 >
    max( T1 x, T2 y )
{
    return (x >= y ? x : y);
}

// three or more arguments
template <typename T1, typename... Types>
scalar_type< T1, Types... >
    max( T1 first, Types... args )
{
    return max( first, max( args... ) );
}

// -----------------------------------------------------------------------------
// min that works with different data types: int64_t = min( int, int64_t )
// and any number of arguments: min( a, b, c, d )

// one argument
template <typename T>
T min( T x )
{
    return x;
}

// two arguments
template <typename T1, typename T2>
scalar_type< T1, T2 >
    min( T1 x, T2 y )
{
    return (x <= y ? x : y);
}

// three or more arguments
template <typename T1, typename... Types>
scalar_type< T1, Types... >
    min( T1 first, Types... args )
{
    return min( first, min( args... ) );
}

// -----------------------------------------------------------------------------
// Generate a scalar from real and imaginary parts.
// For real scalars, the imaginary part is ignored.

// For real scalar types.
template <typename real_t>
struct MakeScalarTraits {
    static real_t make( real_t re, real_t im )
        { return re; }
};

// For complex scalar types.
template <typename real_t>
struct MakeScalarTraits< std::complex<real_t> > {
    static std::complex<real_t> make( real_t re, real_t im )
        { return std::complex<real_t>( re, im ); }
};

template <typename scalar_t>
scalar_t make_scalar( blas::real_type<scalar_t> re,
                      blas::real_type<scalar_t> im=0 )
{
    return MakeScalarTraits<scalar_t>::make( re, im );
}

// -----------------------------------------------------------------------------
/// Type-safe sgn function
/// @see Source: https://stackoverflow.com/a/4609795/5253097
///
template <typename real_t>
int sgn( real_t val )
{
    return (real_t(0) < val) - (val < real_t(0));
}

// -----------------------------------------------------------------------------
// Macros to compute scaling constants
//
// __Further details__
//
// Anderson E (2017) Algorithm 978: Safe scaling in the level 1 BLAS.
// ACM Trans Math Softw 44:. https://doi.org/10.1145/3061665

/// Unit in Last Place
template <typename real_t>
inline const real_t ulp()
{
    return std::numeric_limits< real_t >::epsilon();
}

/// Safe Minimum such that 1/safe_min() is representable
template <typename real_t>
inline const real_t safe_min()
{
    const int fradix = std::numeric_limits<real_t>::radix;
    const int expm = std::numeric_limits<real_t>::min_exponent;
    const int expM = std::numeric_limits<real_t>::max_exponent;

    return max( pow(fradix, expm-1), pow(fradix, 1-expM) );
}

/// Safe Maximum such that 1/safe_max() is representable (SAFMAX := 1/SAFMIN)
template <typename real_t>
inline const real_t safe_max()
{
    const int fradix = std::numeric_limits<real_t>::radix;
    const int expm = std::numeric_limits<real_t>::min_exponent;
    const int expM = std::numeric_limits<real_t>::max_exponent;

    return min( pow(fradix, 1-expm), pow(fradix, expM-1) );
}

/// Safe Minimum such that its square is representable
template <typename real_t>
inline const real_t root_min()
{
    return sqrt( safe_min<real_t>() / ulp<real_t>() );
}

/// Safe Maximum such that its square is representable
template <typename real_t>
inline const real_t root_max()
{
    return sqrt( safe_max<real_t>() * ulp<real_t>() );
}

//==============================================================================
namespace internal {

// -----------------------------------------------------------------------------
// internal helper function; throws Error if cond is true
// called by blas_error_if macro
inline void throw_if( bool cond, const char* condstr, const char* func )
{
    if (cond) {
        throw Error( condstr, func );
    }
}

#if defined(_MSC_VER)
    #define BLASPP_ATTR_FORMAT(I, F)
#else
    #define BLASPP_ATTR_FORMAT(I, F) __attribute__((format( printf, I, F )))
#endif

// -----------------------------------------------------------------------------
// internal helper function; throws Error if cond is true
// uses printf-style format for error message
// called by blas_error_if_msg macro
// condstr is ignored, but differentiates this from other version.
inline void throw_if( bool cond, const char* condstr, const char* func, const char* format, ... )
    BLASPP_ATTR_FORMAT(4, 5);

inline void throw_if( bool cond, const char* condstr, const char* func, const char* format, ... )
{
    if (cond) {
        char buf[80];
        va_list va;
        va_start( va, format );
        vsnprintf( buf, sizeof(buf), format, va );
        throw Error( buf, func );
    }
}

// -----------------------------------------------------------------------------
// internal helper function; aborts if cond is true
// uses printf-style format for error message
// called by blas_error_if_msg macro
inline void abort_if( bool cond, const char* func,  const char* format, ... )
    BLASPP_ATTR_FORMAT(3, 4);

inline void abort_if( bool cond, const char* func,  const char* format, ... )
{
    if (cond) {
        char buf[80];
        va_list va;
        va_start( va, format );
        vsnprintf( buf, sizeof(buf), format, va );

        fprintf( stderr, "Error: %s, in function %s\n", buf, func );
        abort();
    }
}

#undef BLASPP_ATTR_FORMAT

}  // namespace internal

// -----------------------------------------------------------------------------
// internal macros to handle error checks
#if defined(BLAS_ERROR_NDEBUG) || (defined(BLAS_ERROR_ASSERT) && defined(NDEBUG))

    // blaspp does no error checking;
    // lower level BLAS may still handle errors via xerbla
    #define blas_error_if( cond ) \
        ((void)0)

    #define blas_error_if_msg( cond, ... ) \
        ((void)0)

#elif defined(BLAS_ERROR_ASSERT)

    // blaspp aborts on error
    #define blas_error_if( cond ) \
        blas::internal::abort_if( cond, __func__, "%s", #cond )

    #define blas_error_if_msg( cond, ... ) \
        blas::internal::abort_if( cond, __func__, __VA_ARGS__ )

#else

    // blaspp throws errors (default)
    // internal macro to get string #cond; throws Error if cond is true
    // ex: blas_error_if( a < b );
    #define blas_error_if( cond ) \
        blas::internal::throw_if( cond, #cond, __func__ )

    // internal macro takes cond and printf-style format for error message.
    // throws Error if cond is true.
    // ex: blas_error_if_msg( a < b, "a %d < b %d", a, b );
    #define blas_error_if_msg( cond, ... ) \
        blas::internal::throw_if( cond, #cond, __func__, __VA_ARGS__ )

#endif

//------------------------------------------------------------------------------
/// Integer division rounding up instead of down
/// @return ceil( x / y ), for integer types T1, T2.
template <typename T1, typename T2>
inline constexpr std::common_type_t<T1, T2> ceildiv( T1 x, T2 y )
{
    using T = std::common_type_t<T1, T2>;
    return T((x + y - 1) / y);
}

}  // namespace blas

#endif        //  #ifndef BLAS_UTIL_HH
