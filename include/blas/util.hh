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
#include <stdint.h>

namespace blas {

/// Use to silence compiler warning of unused variable.
#define blas_unused( var ) ((void)var)

// For printf, int64_t could be long (%ld), which is >= 32 bits,
// or long long (%lld), guaranteed >= 64 bits.
// Cast to llong to ensure printing 64 bits.
using llong = long long;

//------------------------------------------------------------------------------
/// @brief Exception class for BLAS errors.
///
/// This exception is thrown when BLAS++ detects an error condition, such as
/// invalid parameters or unsupported operations. The error message can be
/// retrieved using the what() method.
///
/// @ingroup util
class Error: public std::exception {
public:
    /// @brief Constructs a BLAS error with no message.
    Error():
        std::exception()
    {}

    /// @brief Constructs a BLAS error with the specified message.
    /// @param[in] msg Error message describing the error condition.
    Error( std::string const& msg ):
        std::exception(),
        msg_( msg )
    {}

    /// @brief Constructs a BLAS error with message and function name.
    /// @param[in] msg Error message describing the error condition.
    /// @param[in] func Name of the function where the error occurred.
    Error( const char* msg, const char* func ):
        std::exception(),
        msg_( std::string(msg) + ", in function " + func )
    {}

    /// @brief Returns the error message.
    /// @return C-string containing the error message.
    virtual const char* what() const noexcept override
        { return msg_.c_str(); }

private:
    std::string msg_;
};

// -----------------------------------------------------------------------------
/// @brief Matrix storage layout.
/// @ingroup enum
enum class Layout : char { 
    ColMajor = 'C',  ///< Column-major storage (Fortran style)
    RowMajor = 'R'   ///< Row-major storage (C style)
};

/// @brief Matrix transpose operation.
/// @ingroup enum
enum class Op : char { 
    NoTrans  = 'N',  ///< No transpose: $op(A) = A$
    Trans    = 'T',  ///< Transpose: $op(A) = A^T$
    ConjTrans = 'C'  ///< Conjugate transpose: $op(A) = A^H$
};

/// @brief Upper or lower triangle of a matrix.
/// @ingroup enum
enum class Uplo : char { 
    Upper    = 'U',  ///< Upper triangle
    Lower    = 'L',  ///< Lower triangle
    General   = 'G'  ///< General (full) matrix
};

/// @brief Matrix diagonal type.
/// @ingroup enum
enum class Diag : char { 
    NonUnit  = 'N',  ///< Non-unit diagonal
    Unit     = 'U'   ///< Unit diagonal (all 1's)
};

/// @brief Matrix multiplication side.
/// @ingroup enum
enum class Side : char { 
    Left     = 'L',  ///< Multiply from the left
    Right    = 'R'   ///< Multiply from the right
};

extern const char* Layout_help;
extern const char* Op_help;
extern const char* Uplo_help;
extern const char* Diag_help;
extern const char* Side_help;

// -----------------------------------------------------------------------------
// Convert enum to LAPACK-style char.

/// @brief Convert Layout enum to LAPACK-style character.
/// @param[in] value Layout enum value
/// @return 'C' for ColMajor, 'R' for RowMajor
/// @ingroup util
inline char to_char( Layout value ) { return char( value ); }

/// @brief Convert Op enum to LAPACK-style character.
/// @param[in] value Op enum value
/// @return 'N' for NoTrans, 'T' for Trans, 'C' for ConjTrans
/// @ingroup util
inline char to_char( Op     value ) { return char( value ); }

/// @brief Convert Uplo enum to LAPACK-style character.
/// @param[in] value Uplo enum value
/// @return 'U' for Upper, 'L' for Lower, 'G' for General
/// @ingroup util
inline char to_char( Uplo   value ) { return char( value ); }

/// @brief Convert Diag enum to LAPACK-style character.
/// @param[in] value Diag enum value
/// @return 'N' for NonUnit, 'U' for Unit
/// @ingroup util
inline char to_char( Diag   value ) { return char( value ); }

/// @brief Convert Side enum to LAPACK-style character.
/// @param[in] value Side enum value
/// @return 'L' for Left, 'R' for Right
/// @ingroup util
inline char to_char( Side   value ) { return char( value ); }

//------------------------------------------------------------------------------
// Convert enum to LAPACK-style C string (const char*).

/// @brief Convert Layout enum to C-style string.
/// @param[in] value Layout enum value
/// @return "col" for ColMajor, "row" for RowMajor, "?" for unknown
/// @ingroup util
inline const char* to_c_string( Layout value )
{
    switch (value) {
        case Layout::ColMajor: return "col";
        case Layout::RowMajor: return "row";
    }
    return "?";
}

/// @brief Convert Op enum to C-style string.
/// @param[in] value Op enum value
/// @return "notrans" for NoTrans, "trans" for Trans, "conj" for ConjTrans, "?" for unknown
/// @ingroup util
inline const char* to_c_string( Op value )
{
    switch (value) {
        case Op::NoTrans:   return "notrans";
        case Op::Trans:     return "trans";
        case Op::ConjTrans: return "conj";
    }
    return "?";
}

/// @brief Convert Uplo enum to C-style string.
/// @param[in] value Uplo enum value
/// @return "lower", "upper", "general", or "?" for unknown
/// @ingroup util
inline const char* to_c_string( Uplo value )
{
    switch (value) {
        case Uplo::Lower:   return "lower";
        case Uplo::Upper:   return "upper";
        case Uplo::General: return "general";
    }
    return "?";
}

/// @brief Convert Diag enum to C-style string.
/// @param[in] value Diag enum value
/// @return "nonunit" for NonUnit, "unit" for Unit, "?" for unknown
/// @ingroup util
inline const char* to_c_string( Diag value )
{
    switch (value) {
        case Diag::NonUnit: return "nonunit";
        case Diag::Unit:    return "unit";
    }
    return "?";
}

/// @brief Convert Side enum to C-style string.
/// @param[in] value Side enum value
/// @return "left" for Left, "right" for Right, "?" for unknown
/// @ingroup util
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

/// @brief Convert Layout enum to C++ string.
/// @param[in] value Layout enum value
/// @return String representation of the layout
/// @ingroup util
inline std::string to_string( Layout value )
{
    return to_c_string( value );
}

/// @brief Convert Op enum to C++ string.
/// @param[in] value Op enum value
/// @return String representation of the transpose operation
/// @ingroup util
inline std::string to_string( Op value )
{
    return to_c_string( value );
}

/// @brief Convert Uplo enum to C++ string.
/// @param[in] value Uplo enum value
/// @return String representation of the triangle specification
/// @ingroup util
inline std::string to_string( Uplo value )
{
    return to_c_string( value );
}

/// @brief Convert Diag enum to C++ string.
/// @param[in] value Diag enum value
/// @return String representation of the diagonal type
/// @ingroup util
inline std::string to_string( Diag value )
{
    return to_c_string( value );
}

/// @brief Convert Side enum to C++ string.
/// @param[in] value Side enum value
/// @return String representation of the side
/// @ingroup util
inline std::string to_string( Side value )
{
    return to_c_string( value );
}

//------------------------------------------------------------------------------
// Convert LAPACK-style char or string to enum.

/// @brief Convert string to Layout enum.
/// @param[in] str String to convert (case-insensitive): "c", "colmajor", "r", "rowmajor"
/// @param[out] val Pointer to Layout variable to set
/// @throws Error if string is not recognized
/// @ingroup util
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

/// @brief Convert string to Op enum.
/// @param[in] str String to convert (case-insensitive): "n", "notrans", "t", "trans", "c", "conjtrans"
/// @param[out] val Pointer to Op variable to set
/// @throws Error if string is not recognized
/// @ingroup util
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

/// @brief Convert string to Uplo enum.
/// @param[in] str String to convert (case-insensitive): "l", "lower", "u", "upper", "g", "general"
/// @param[out] val Pointer to Uplo variable to set
/// @throws Error if string is not recognized
/// @ingroup util
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

/// @brief Convert string to Diag enum.
/// @param[in] str String to convert (case-insensitive): "n", "nonunit", "u", "unit"
/// @param[out] val Pointer to Diag variable to set
/// @throws Error if string is not recognized
/// @ingroup util
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

/// @brief Convert string to Side enum.
/// @param[in] str String to convert (case-insensitive): "l", "left", "r", "right"
/// @param[out] val Pointer to Side variable to set
/// @throws Error if string is not recognized
/// @ingroup util
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
/// @brief Compute 1-norm absolute value: |Re(x)| + |Im(x)|.
/// For real types, this is equivalent to abs(x).
/// @param[in] x Value to compute absolute value of
/// @return 1-norm absolute value
/// @ingroup util
template <typename T>
T abs1( T x )
{
    using std::abs;
    return abs( x );
}

/// @brief Compute 1-norm absolute value for complex types: |Re(x)| + |Im(x)|.
/// @param[in] x Complex value
/// @return Sum of absolute values of real and imaginary parts
/// @ingroup util
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
/// @brief Type trait to detect if T is std::complex.
///
/// Evaluates to std::true_type if T is std::complex<T2> for some type T2,
/// otherwise std::false_type.
///
/// @tparam T Type to check
/// @ingroup util
template <typename T>
struct is_complex:
    std::integral_constant<bool, false>
{};

/// @brief Specialization for std::complex types.
/// @tparam T Base type of the complex number
/// @ingroup util
template <typename T>
struct is_complex< std::complex<T> >:
    std::integral_constant<bool, true>
{};

/// @brief Helper variable template for is_complex.
/// @tparam T Type to check
/// @ingroup util
template <typename T>
constexpr bool is_complex_v = is_complex<T>::value;

// -----------------------------------------------------------------------------
// Previously extended real and imag to real types. Belatedly discovered that
// C++11 extends std::real and std::imag to float and integer types,
// so just use those now.
using std::real;
using std::imag;

/// @brief Extend conjugate function to real datatypes.
///
/// For real T, returns type T unchanged, whereas C++11 std::conj returns complex<T>.
/// This allows generic code to work with both real and complex types.
///
/// Usage:
/// @code
///     using blas::conj;
///     scalar_t x = ...
///     scalar_t y = conj( x );  // Uses std::conj for complex, blas::conj for real
/// @endcode
///
/// @note This version prohibits complex types and will cause a static assertion
///       if called directly as `blas::conj(x)` rather than via ADL with `using blas::conj`.
///
/// @param[in] x Real value
/// @return x unchanged (conjugate of a real number is itself)
/// @tparam T Real scalar type
/// @ingroup util
template <typename T>
inline T conj( T x )
{
    static_assert(
        ! is_complex_v<T>,
        "Usage: using blas::conj; y = conj(x); NOT: y = blas::conj(x);" );
    return x;
}

// -----------------------------------------------------------------------------
/// @brief Type trait for determining common scalar type.
///
/// Based on C++14 common_type implementation but with improved handling of
/// complex types. Promotes complex types based on the common type of the
/// associated real types. This fixes several important cases:
///
/// - `std::common_type_t<double, complex<float>>` → `complex<float>` (wrong)
///   `scalar_type<double, complex<float>>` → `complex<double>` (correct)
///
/// - `std::common_type_t<int, complex<long>>` → undefined (compile error)
///   `scalar_type<int, complex<long>>` → `complex<long>` (correct)
///
/// @tparam Types... Variable number of types to find common type for
/// @ingroup util
template <typename... Types>
struct scalar_type_traits;

/// @brief Type alias for scalar_type_traits.
/// @tparam Types... Variable number of types
/// @ingroup util
template <typename... Types>
using scalar_type = typename scalar_type_traits< Types... >::type;

/// @brief Specialization for single type.
/// @tparam T Type to determine scalar type for
/// @ingroup util
template <typename T>
struct scalar_type_traits< T >
{
    using type = decay_t<T>;
};

/// @brief Specialization for two types.
/// Relies on the type of the ?: operator being the common type of its arguments.
/// @tparam T1 First type
/// @tparam T2 Second type
/// @ingroup util
template <typename T1, typename T2>
struct scalar_type_traits< T1, T2 >
{
    using type = decay_t< decltype( true ? std::declval<T1>() : std::declval<T2>() ) >;
};

/// @brief Specialization when first type is complex.
/// Finds common type of associated real types, then adds complex.
/// @tparam T1 Base type of complex
/// @tparam T2 Second type (may be real or complex)
/// @ingroup util
template <typename T1, typename T2>
struct scalar_type_traits< std::complex<T1>, T2 >
{
    using type = std::complex< common_type_t< T1, T2 > >;
};

/// @brief Specialization when second type is complex.
/// Finds common type of associated real types, then adds complex.
/// @tparam T1 First type (must be real)
/// @tparam T2 Base type of complex
/// @ingroup util
template <typename T1, typename T2>
struct scalar_type_traits< T1, std::complex<T2> >
{
    using type = std::complex< common_type_t< T1, T2 > >;
};

/// @brief Specialization when both types are complex.
/// Finds common type of associated real types, then adds complex.
/// @tparam T1 Base type of first complex
/// @tparam T2 Base type of second complex
/// @ingroup util
template <typename T1, typename T2>
struct scalar_type_traits< std::complex<T1>, std::complex<T2> >
{
    using type = std::complex< common_type_t< T1, T2 > >;
};

/// @brief Specialization for three or more types.
/// Recursively applies scalar_type to pairs of types.
/// @tparam T1 First type
/// @tparam T2 Second type
/// @tparam Types... Remaining types
/// @ingroup util
template <typename T1, typename T2, typename... Types>
struct scalar_type_traits< T1, T2, Types... >
{
    using type = scalar_type< scalar_type< T1, T2 >, Types... >;
};

// -----------------------------------------------------------------------------
/// @brief Type traits for determining associated real, scalar, and complex types.
///
/// For any combination of types, determines the associated real, scalar,
/// and complex types with proper promotion rules:
///
/// Examples:
/// - `real_type<float>` → `float`
/// - `real_type<float, double, complex<float>>` → `double`
/// - `scalar_type<float>` → `float`
/// - `scalar_type<float, complex<float>>` → `complex<float>`
/// - `scalar_type<float, double, complex<float>>` → `complex<double>`
/// - `complex_type<float>` → `complex<float>`
/// - `complex_type<float, double>` → `complex<double>`
/// - `complex_type<float, double, complex<float>>` → `complex<double>`
///
/// @tparam Types... Variable number of types
/// @ingroup util
template <typename... Types>
struct real_type_traits;

/// @brief Type alias for extracting the real type from type traits.
/// @tparam Types... Variable number of types
/// @ingroup util
template <typename... Types>
using real_type = typename real_type_traits< Types... >::real_t;

/// @brief Type alias for constructing complex type from real types.
/// @tparam Types... Variable number of types
/// @ingroup util
template <typename... Types>
using complex_type = std::complex< real_type< Types... > >;

/// @brief Specialization for single real type.
/// @tparam T Real type
/// @ingroup util
template <typename T>
struct real_type_traits<T>
{
    using real_t = T;
};

/// @brief Specialization for single complex type - extracts the base real type.
/// @tparam T Base real type of the complex number
/// @ingroup util
template <typename T>
struct real_type_traits< std::complex<T> >
{
    using real_t = T;
};

/// @brief Specialization for two or more types.
/// Recursively determines the common real type.
/// @tparam T1 First type
/// @tparam Types... Remaining types
/// @ingroup util
template <typename T1, typename... Types>
struct real_type_traits< T1, Types... >
{
    using real_t = scalar_type< real_type<T1>, real_type< Types... > >;
};

// -----------------------------------------------------------------------------
/// @brief Maximum value supporting mixed types and variadic arguments.
///
/// Works with different data types (e.g., `int64_t = max(int, int64_t)`)
/// and any number of arguments (e.g., `max(a, b, c, d)`).
///
/// @param[in] x Single value
/// @return x
/// @tparam T Value type
/// @ingroup util
template <typename T>
T max( T x )
{
    return x;
}

/// @brief Maximum of two values with type promotion.
/// @param[in] x First value
/// @param[in] y Second value
/// @return Maximum of x and y, promoted to common type
/// @tparam T1 Type of first value
/// @tparam T2 Type of second value
/// @ingroup util
template <typename T1, typename T2>
scalar_type< T1, T2 >
    max( T1 x, T2 y )
{
    return (x >= y ? x : y);
}

/// @brief Maximum of three or more values with type promotion.
/// @param[in] first First value
/// @param[in] args Remaining values
/// @return Maximum of all values, promoted to common type
/// @tparam T1 Type of first value
/// @tparam Types... Types of remaining values
/// @ingroup util
template <typename T1, typename... Types>
scalar_type< T1, Types... >
    max( T1 first, Types... args )
{
    return max( first, max( args... ) );
}

// -----------------------------------------------------------------------------
/// @brief Minimum value supporting mixed types and variadic arguments.
///
/// Works with different data types (e.g., `int64_t = min(int, int64_t)`)
/// and any number of arguments (e.g., `min(a, b, c, d)`).
///
/// @param[in] x Single value
/// @return x
/// @tparam T Value type
/// @ingroup util
template <typename T>
T min( T x )
{
    return x;
}

/// @brief Minimum of two values with type promotion.
/// @param[in] x First value
/// @param[in] y Second value
/// @return Minimum of x and y, promoted to common type
/// @tparam T1 Type of first value
/// @tparam T2 Type of second value
/// @ingroup util
template <typename T1, typename T2>
scalar_type< T1, T2 >
    min( T1 x, T2 y )
{
    return (x <= y ? x : y);
}

/// @brief Minimum of three or more values with type promotion.
/// @param[in] first First value
/// @param[in] args Remaining values
/// @return Minimum of all values, promoted to common type
/// @tparam T1 Type of first value
/// @tparam Types... Types of remaining values
/// @ingroup util
template <typename T1, typename... Types>
scalar_type< T1, Types... >
    min( T1 first, Types... args )
{
    return min( first, min( args... ) );
}

// -----------------------------------------------------------------------------
/// @brief Traits for constructing scalars from real and imaginary parts.
///
/// For real scalars, the imaginary part is ignored.
/// For complex scalars, both parts are used.
///
/// @tparam real_t Real scalar type or base type of complex
/// @ingroup util
template <typename real_t>
struct MakeScalarTraits {
    static real_t make( real_t re, real_t im )
        { return re; }
};

/// @brief Specialization for complex scalar types.
/// @tparam real_t Base real type of the complex number
/// @ingroup util
template <typename real_t>
struct MakeScalarTraits< std::complex<real_t> > {
    static std::complex<real_t> make( real_t re, real_t im )
        { return std::complex<real_t>( re, im ); }
};

/// @brief Generate a scalar from real and imaginary parts.
///
/// For real scalar types, the imaginary part is ignored.
/// For complex scalar types, constructs complex(re, im).
///
/// @param[in] re Real part
/// @param[in] im Imaginary part (default 0)
/// @return Scalar value of type scalar_t
/// @tparam scalar_t Target scalar type (real or complex)
/// @ingroup util
template <typename scalar_t>
scalar_t make_scalar( blas::real_type<scalar_t> re,
                      blas::real_type<scalar_t> im=0 )
{
    return MakeScalarTraits<scalar_t>::make( re, im );
}

// -----------------------------------------------------------------------------
/// @brief Type-safe sign function.
///
/// Returns:
/// - -1 if val < 0
/// - 0 if val == 0
/// - +1 if val > 0
///
/// @param[in] val Value to determine sign of
/// @return Sign of val as -1, 0, or +1
/// @tparam real_t Real value type
/// @see Source: https://stackoverflow.com/a/4609795/5253097
/// @ingroup util
template <typename real_t>
int sgn( real_t val )
{
    return (real_t(0) < val) - (val < real_t(0));
}

// -----------------------------------------------------------------------------
/// @brief Numerical scaling constants for safe computation.
///
/// These functions compute scaling constants used for numerically stable
/// algorithms, particularly in BLAS operations.
///
/// @see Anderson E (2017) Algorithm 978: Safe scaling in the level 1 BLAS.
///      ACM Trans Math Softw 44. https://doi.org/10.1145/3061665
/// @ingroup util

/// @brief Unit in Last Place (machine epsilon).
///
/// Returns the smallest positive number such that 1.0 + ulp() != 1.0.
/// This is equivalent to machine epsilon.
///
/// @return Machine epsilon for type real_t
/// @tparam real_t Real floating-point type
/// @ingroup util
template <typename real_t>
inline const real_t ulp()
{
    return std::numeric_limits< real_t >::epsilon();
}

/// @brief Safe minimum value.
///
/// Returns the smallest positive value such that 1/safe_min() is representable
/// without overflow. Useful for avoiding overflow in division and scaling operations.
///
/// @return Safe minimum value for type real_t
/// @tparam real_t Real floating-point type
/// @ingroup util
template <typename real_t>
inline const real_t safe_min()
{
    const int fradix = std::numeric_limits<real_t>::radix;
    const int expm = std::numeric_limits<real_t>::min_exponent;
    const int expM = std::numeric_limits<real_t>::max_exponent;

    return max( pow(fradix, expm-1), pow(fradix, 1-expM) );
}

/// @brief Safe maximum value.
///
/// Returns the largest value such that 1/safe_max() is representable
/// without underflow. Equals 1/safe_min().
///
/// @return Safe maximum value for type real_t
/// @tparam real_t Real floating-point type
/// @ingroup util
template <typename real_t>
inline const real_t safe_max()
{
    const int fradix = std::numeric_limits<real_t>::radix;
    const int expm = std::numeric_limits<real_t>::min_exponent;
    const int expM = std::numeric_limits<real_t>::max_exponent;

    return min( pow(fradix, 1-expm), pow(fradix, expM-1) );
}

/// @brief Safe minimum for operations involving square roots.
///
/// Returns the smallest positive value such that its square is representable
/// without underflow. Useful for avoiding underflow in norm computations.
///
/// @return Safe minimum for square operations
/// @tparam real_t Real floating-point type
/// @ingroup util
template <typename real_t>
inline const real_t root_min()
{
    return sqrt( safe_min<real_t>() / ulp<real_t>() );
}

/// @brief Safe maximum for operations involving square roots.
///
/// Returns the largest value such that its square is representable
/// without overflow. Useful for avoiding overflow in norm computations.
///
/// @return Safe maximum for square operations
/// @tparam real_t Real floating-point type
/// @ingroup util
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
/// @brief Integer division rounding up.
///
/// Computes ceiling division: ceil(x / y) using integer arithmetic.
/// Equivalent to (x + y - 1) / y but type-safe.
///
/// @param[in] x Dividend
/// @param[in] y Divisor
/// @return Ceiling of x / y
/// @tparam T1 Type of dividend
/// @tparam T2 Type of divisor
/// @ingroup util
template <typename T1, typename T2>
inline constexpr std::common_type_t<T1, T2> ceildiv( T1 x, T2 y )
{
    using T = std::common_type_t<T1, T2>;
    return T((x + y - 1) / y);
}

}  // namespace blas

#endif        //  #ifndef BLAS_UTIL_HH
