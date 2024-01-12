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

#include <assert.h>

#include <blas/defines.h>

#ifdef BLAS_HAVE_CUBLAS
#include <cuda_fp16.h>
#elif defined(BLAS_HAVE_ROCBLAS)
#include <hip/hip_fp16.h>
#elif defined(BLAS_HAVE_MKL)
#include <mkl_types.h>
#endif

namespace blas {

#ifdef BLAS_HAVE_ISO_FLOAT16
  using float16 = _Float16;

#elif defined(BLAS_HAVE_CUBLAS)
  using float16 = __half;

#elif defined(BLAS_HAVE_ROCBLAS)
  using float16 = rocblas_half;

#else
class float16 {
#if defined(BLAS_HAVE_MKL)
    using float16_ = MKL_F16;
#else
    using float16_ = uint16_t;
#endif

public:
    float16() : data_( 0.0f ) { }
    
    // TODO manipulate the bits here
    float16( float v ) { data_ = float_to_float16( v ); }

    // TODO manipulate the bits here
    operator float() const {
        return float16_to_float( data_ );
    }

private:
    float16_ data_;

    typedef union {
      float16_  data;
      struct {
        unsigned int frac : 10;
        unsigned int exp  :  5;
        unsigned int sign :  1;
      } bits;
    } float16_repr_data_t;

    typedef union {
      float data;
      struct {
        unsigned int frac : 23;
        unsigned int exp  :  8;
        unsigned int sign :  1;
      } bits;
    } float_repr_data_t;

    static float float16_to_float(float16_ x) {
        float16_repr_data_t src;
        float_repr_data_t dst;

        src.data = x;
        dst.data = 0;
        dst.bits.sign = src.bits.sign;

        if (src.bits.exp == 0x01fU) {
            dst.bits.exp  = 0xffU;
            if (src.bits.frac > 0) {
                dst.bits.frac = ((src.bits.frac | 0x200U) << 13);
            }
        } else if (src.bits.exp > 0x00U) {
            dst.bits.exp  = src.bits.exp + ((1 << 7) - (1 << 4));
            dst.bits.frac = (src.bits.frac << 13);
        } else {
            unsigned int v = (src.bits.frac << 13);

            if (v > 0) {
                dst.bits.exp = 0x71;
                while ((v & 0x800000UL) == 0) {
                    dst.bits.exp --;
                    v <<= 1;
                }
                dst.bits.frac = v;
            }
        }

        return dst.data;
    }

    static float16_ float_to_float16(float x) {
        float_repr_data_t src;
        float16_repr_data_t dst;

        src.data = x;
        dst.data = 0;
        dst.bits.sign = src.bits.sign;

        if (src.bits.exp == 0x0ffU) {
            dst.bits.exp  = 0x01fU;
            dst.bits.frac = (src.bits.frac >> 13);
            if (src.bits.frac > 0) dst.bits.frac |= 0x200U;
        } else if (src.bits.exp >= 0x08fU) {
            dst.bits.exp  = 0x01fU;
            dst.bits.frac = 0x000U;
        } else if (src.bits.exp >= 0x071U){
            dst.bits.exp  = src.bits.exp + ((1 << 4) - (1 << 7));
            dst.bits.frac = (src.bits.frac >> 13);
        } else if (src.bits.exp >= 0x067U){
            dst.bits.exp  = 0x000;
            if (src.bits.frac > 0) {
                dst.bits.frac = (((1U << 23) | src.bits.frac) >> 14);
            } else {
                dst.bits.frac = 1;
            }
        }

        return dst.data;
    }
};
#endif

inline float real(float16& a) { return float( a ); }
inline float imag(float16& a) { return 0.0f; }

/// Use to silence compiler warning of unused variable.
#define blas_unused( var ) ((void)var)

// For printf, int64_t could be long (%ld), which is >= 32 bits,
// or long long (%lld), guaranteed >= 64 bits.
// Cast to llong to ensure printing 64 bits.
using llong = long long;

// -----------------------------------------------------------------------------
enum class Layout : char { ColMajor = 'C', RowMajor = 'R' };
enum class Op     : char { NoTrans  = 'N', Trans    = 'T', ConjTrans = 'C' };
enum class Uplo   : char { Upper    = 'U', Lower    = 'L', General   = 'G' };
enum class Diag   : char { NonUnit  = 'N', Unit     = 'U' };
enum class Side   : char { Left     = 'L', Right    = 'R' };
enum class Format : char { LAPACK   = 'L', Tile     = 'T' };

// -----------------------------------------------------------------------------
// Convert enum to LAPACK-style char.
inline char layout2char( Layout layout ) { return char(layout); }
inline char     op2char( Op     op     ) { return char(op);     }
inline char   uplo2char( Uplo   uplo   ) { return char(uplo);   }
inline char   diag2char( Diag   diag   ) { return char(diag);   }
inline char   side2char( Side   side   ) { return char(side);   }
inline char format2char( Format format ) { return char(format); }

// -----------------------------------------------------------------------------
// Convert enum to LAPACK-style string.
inline const char* layout2str( Layout layout )
{
    switch (layout) {
        case Layout::ColMajor: return "col";
        case Layout::RowMajor: return "row";
    }
    return "";
}

inline const char* op2str( Op op )
{
    switch (op) {
        case Op::NoTrans:   return "notrans";
        case Op::Trans:     return "trans";
        case Op::ConjTrans: return "conj";
    }
    return "";
}

inline const char* uplo2str( Uplo uplo )
{
    switch (uplo) {
        case Uplo::Lower:   return "lower";
        case Uplo::Upper:   return "upper";
        case Uplo::General: return "general";
    }
    return "";
}

inline const char* diag2str( Diag diag )
{
    switch (diag) {
        case Diag::NonUnit: return "nonunit";
        case Diag::Unit:    return "unit";
    }
    return "";
}

inline const char* side2str( Side side )
{
    switch (side) {
        case Side::Left:  return "left";
        case Side::Right: return "right";
    }
    return "";
}

inline const char* format2str( Format format )
{
    switch (format) {
        case Format::LAPACK: return "lapack";
        case Format::Tile: return "tile";
    }
    return "";
}

// -----------------------------------------------------------------------------
// Convert LAPACK-style char to enum.
inline Layout char2layout( char layout )
{
    layout = (char) toupper( layout );
    assert( layout == 'C' || layout == 'R' );
    return Layout( layout );
}

inline Op char2op( char op )
{
    op = (char) toupper( op );
    assert( op == 'N' || op == 'T' || op == 'C' );
    return Op( op );
}

inline Uplo char2uplo( char uplo )
{
    uplo = (char) toupper( uplo );
    assert( uplo == 'L' || uplo == 'U' || uplo == 'G' );
    return Uplo( uplo );
}

inline Diag char2diag( char diag )
{
    diag = (char) toupper( diag );
    assert( diag == 'N' || diag == 'U' );
    return Diag( diag );
}

inline Side char2side( char side )
{
    side = (char) toupper( side );
    assert( side == 'L' || side == 'R' );
    return Side( side );
}

inline Format char2format( char format )
{
    format = (char) toupper( format );
    assert( format == 'L' || format == 'T' );
    return Format( format );
}

// -----------------------------------------------------------------------------
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
        ! is_complex<T>::value,
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

}  // namespace blas

#endif        //  #ifndef BLAS_UTIL_HH
