// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <string>
#include <cmath>

// prefer OpenMP get_wtime; else use gettimeofday
#ifdef _OPENMP
    #include <omp.h>
#else
    #include <sys/time.h>
#endif

#include "testsweeper.hh"

namespace testsweeper {

//==================================================
// Copied from blaspp/include/blas/util.hh
// -----------------------------------------------------------------------------
// for any combination of types, determine associated real, scalar,
// and complex types.
//
// real_type< float >                               is float
// real_type< float, double, complex<float> >       is double
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
//==================================================

//------------------------------------------------------------------------------
// Global constants

// numeric flag indicating no data; printed as "NA" instead of "nan"
double no_data_flag = nan("1234");

// -----------------------------------------------------------------------------
// ANSI color codes - enabled by default
#ifndef NO_COLOR
const char *ansi_esc     = "\x1b[";
const char *ansi_red     = "\x1b[31m";
const char *ansi_green   = "\x1b[92m";
const char *ansi_blue    = "\x1b[34m";
const char *ansi_cyan    = "\x1b[36m";
const char *ansi_magenta = "\x1b[35m";
const char *ansi_yellow  = "\x1b[33m";
const char *ansi_white   = "\x1b[37m";
const char *ansi_gray    = "\x1b[90m";  // "bright black"
const char *ansi_bold    = "\x1b[1m";
const char *ansi_normal  = "\x1b[0m";
#else
const char *ansi_esc     = "";
const char *ansi_red     = "";
const char *ansi_green   = "";
const char *ansi_blue    = "";
const char *ansi_cyan    = "";
const char *ansi_magenta = "";
const char *ansi_yellow  = "";
const char *ansi_white   = "";
const char *ansi_gray    = "";
const char *ansi_bold    = "";
const char *ansi_normal  = "";
#endif

// -----------------------------------------------------------------------------
// static class variables
std::vector< ParamBase* > ParamBase::s_params;

// -----------------------------------------------------------------------------
// Compare a == b, bitwise. Returns true if a and b are both the same NaN value,
// unlike (a == b) which is false for NaNs.
bool same( double a, double b )
{
    return (memcmp( &a, &b, sizeof(double) ) == 0);
}

// -----------------------------------------------------------------------------
/// Throws a std::runtime_error, using a printf-formatted message in format
/// and subsequent arguments.
void throw_error( const char* format, ... )
{
    char msg[1000];
    va_list va;
    va_start( va, format );
    vsnprintf( msg, sizeof(msg), format, va );
    throw std::runtime_error( msg );
}

//------------------------------------------------------------------------------
/// Scan string for metric or binary prefix and return multiplier.
/// E.g., for "k" returns 1000, for "Ki" returns 1024.
/// If no prefix is recognized, returns 1.
/// Also recognizes exponents "eN" for integer N >= 0.
///
/// metric                 |  binary
/// -----------------------|----------------------------
/// k: kilo 1000   = 1e3   |  Ki: kibi kilobinary 1024
/// M: mega 1000^2 = 1e6   |  Mi: mebi megabinary 1024^2
/// G: giga 1000^3 = 1e9   |  Gi: gibi gigabinary 1024^3
/// T: tera 1000^4 = 1e12  |  Ti: tebi terabinary 1024^4
/// P: peta 1000^5 = 1e15  |  Pi: pebi petabinary 1024^5
/// E: exa  1000^6 = 1e18  |  Ei: exbi  exabinary 1024^6
///
/// @param[in,out] pstr
///     Pointer to string to scan.
///     On exit, updated to point after any prefix that was scanned.
///
/// @return multiplier
///
int64_t scan_multiplier( const char **pstr )
{
    const int64_t i1024 = 1024;
    int exponent, bytes;

    // Metric is lowercase k, but IEC is uppercase Ki for binary;
    // ignore case for k only.
    if (tolower(**pstr) == 'k') {
        *pstr += 1;
        if (**pstr == 'i') {
            *pstr += 1;
            return i1024;
        }
        else {
            return 1e3;
        }
    }
    else if (**pstr == 'M') {
        *pstr += 1;
        if (**pstr == 'i') {
            *pstr += 1;
            return i1024 * i1024;
        }
        else {
            return 1e6;
        }
    }
    else if (**pstr == 'G') {
        *pstr += 1;
        if (**pstr == 'i') {
            *pstr += 1;
            return i1024 * i1024 * i1024;
        }
        else {
            return 1e9;
        }
    }
    else if (**pstr == 'T') {
        *pstr += 1;
        if (**pstr == 'i') {
            *pstr += 1;
            return i1024 * i1024 * i1024 * i1024;
        }
        else {
            return 1e12;
        }
    }
    else if (**pstr == 'P') {
        *pstr += 1;
        if (**pstr == 'i') {
            *pstr += 1;
            return i1024 * i1024 * i1024 * i1024 * i1024;
        }
        else {
            return 1e15;
        }
    }
    else if (**pstr == 'E') {
        *pstr += 1;
        if (**pstr == 'i') {
            *pstr += 1;
            return i1024 * i1024 * i1024 * i1024 * i1024 * i1024;
        }
        else {
            return 1e18;
        }
    }
    else if (sscanf( *pstr, "e%d%n", &exponent, &bytes ) == 1) {
        // pow() function can lead to undesireable rounding, 1e6 = 999999,
        // so multiply with loop.
        *pstr += bytes;
        int64_t mul = 1;
        for (int i = 0; i < exponent; ++i)
            mul *= 10;
        return mul;
    }
    else {
        return 1;
    }
}

///-----------------------------------------------------------------------------
/// Scans string for single integer or range of integers (start:end:step).
/// Advances the string to after the range or number.
///
/// @param[in,out] pstr: pointer to string containing an integer or range.
///                      On output, advanced to after the number or range.
/// @param[out] start: start of range
/// @param[out] end:   end of range
/// @param[out] step:  step size; default is start; 0 if start = end.
///
/// @retval 1: failure
/// @retval 0: success
///
int scan_range( const char** pstr, int64_t* start, int64_t* end, int64_t* step )
{
    long long start_ = 0, end_ = 0, step_ = 0;
    int bytes;
    const char* str = *pstr;

    // cnt of numbers scanned (start, end, step).
    int cnt = 0;

    // Start number.
    int info = sscanf( str, "%lld %n", &start_, &bytes );
    if (info == 1) {
        cnt += 1;
        str += bytes;
        start_ *= scan_multiplier( &str );

        // End number.
        info = sscanf( str, " : %lld %n", &end_, &bytes );
        if (info == 1) {
            cnt += 1;
            str += bytes;
            end_ *= scan_multiplier( &str );

            // Step number.
            info = sscanf( str, " : %lld %n", &step_, &bytes );
            if (info == 1) {
                cnt += 1;
                str += bytes;
                step_ *= scan_multiplier( &str );
            }
        }
    }

    *pstr  = str;
    *start = start_;
    *end   = end_;
    *step  = step_;
    if (cnt == 3) {
        if (*start == *end)
            *step = 0;
    }
    else if (cnt == 2) {
        if (*start == *end)
            *step = 0;
        else
            *step = *start;
    }
    else if (cnt == 1) {
        *end  = *start;
        *step = 0;
    }
    else {
        return 1;  // failure
    }
    return ! ((*step == 0 && *start == *end) ||
              (*step >  0 && *start <  *end) ||
              (*step <  0 && *start >  *end));
}

///-----------------------------------------------------------------------------
/// Scans string for a double or range of doubles (start:end:step).
/// Advances the string to after the number or range.
///
/// @param[in,out] pstr: pointer to string containing a double or range.
///                      On output, advanced to after the number or range.
/// @param[out] start: start of range
/// @param[out] end:   end of range
/// @param[out] step:  step size; default is start; 0 if start = end.
///
/// @retval 1: failure
/// @retval 0: success
///
int scan_range( const char** pstr, double* start, double* end, double* step )
{
    int bytes1, bytes2, bytes3, cnt;
    cnt = sscanf( *pstr, "%lf %n: %lf %n: %lf %n",
                  start, &bytes1, end, &bytes2, step, &bytes3 );
    if (cnt == 3) {
        if (*start == *end)
            *step = 0;
        *pstr += bytes3;
    }
    else if (cnt == 2) {
        *pstr += bytes2;
        if (*start == *end)
            *step = 0;
        else
            *step = *start;
    }
    else if (cnt == 1) {
        *pstr += bytes1;
        *end  = *start;
        *step = 0;
    }
    else {
        return 1;  // failure
    }
    return ! ((*step == 0 && *start == *end) ||
              (*step >  0 && *start <  *end) ||
              (*step <  0 && *start >  *end));
}

// -----------------------------------------------------------------------------
// Flushes cache by allocating buffer of 2*cache size (in MiB),
// and writing it in parallel.
void flush_cache( size_t cache_size )
{
    size_t len = 2 * cache_size * 1024 * 1024;
    unsigned char *buf = (unsigned char*) malloc( len );

    int nthread = 1;
    #pragma omp parallel
    #pragma omp master
    {
        #ifdef _OPENMP
        nthread = omp_get_num_threads();
        #endif
    }

    size_t per_core = len / nthread;

    #pragma omp parallel
    {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        for (size_t i = tid * per_core; i < (tid + 1) * per_core; ++i) {
            buf[i] = i % 256;
        }
    }

    free( buf );
}

// =============================================================================
// ParamBase class

// -----------------------------------------------------------------------------
// if name has \n, for line=0, print 1st line; for line=1, print 2nd line.
// otherwise,      for line=0, print blank;    for line=1, print name.
// virtual
void ParamBase::header( int line ) const
{
    if (used_ && width_ > 0) {
        size_t i = name_.find( '\n' );
        std::string str;
        if (i != std::string::npos) {
            str = (line == 0
                ? name_.substr( 0, i ).c_str()
                : name_.substr( i+1 ).c_str() );
        }
        else {
            str = (line == 0
                ? ""
                : name_.c_str() );
        }
        printf( "%*s  ", width_, str.c_str() );
    }
}

// -----------------------------------------------------------------------------
// virtual
void ParamBase::help() const
{
    if (type_ == ParamType::Value || type_ == ParamType::List) {
        printf( "    %-16s %s\n", option_.c_str(), help_.c_str() );
    }
}

// -----------------------------------------------------------------------------
// virtual
bool ParamBase::next()
{
    assert( index_ >= 0 && index_ < size() );
    if (index_ == size() - 1) {
        index_ = 0;
        return false;
    }
    else {
        index_ += 1;
        return true;
    }
}

// =============================================================================
// ParamInt class
// Integer parameters

// -----------------------------------------------------------------------------
// virtual
void ParamInt::parse( const char *str )
{
    while( true ) {
        int64_t start, end, step;
        if (scan_range( &str, &start, &end, &step ) != 0) {
            throw_error( "invalid argument at '%s',"
                         " expected integer or range start:end:step", str );
        }
        if (start == end) {
            push_back( start );
        }
        else {
            for (int64_t val = start; val <= end; val += step) {
                push_back( val );
            }
        }
        if (*str == '\0') {
            break;
        }
        if (*str != ',' && *str != ';') {
            throw_error( "invalid argument at '%s', expected comma", str );
        }
        str += 1;
    }
}

// -----------------------------------------------------------------------------
void ParamInt::push_back( int64_t val )
{
    if (val < min_value_ || val > max_value_) {
        throw_error( "invalid argument, %lld outside [%lld, %lld]",
                     (long long) val,
                     (long long) min_value_,
                     (long long) max_value_ );
    }
    TParamBase<int64_t>::push_back( val );
}

// -----------------------------------------------------------------------------
// virtual
void ParamInt::print() const
{
    if (used_ && width_ > 0) {
        printf( "%*lld  ", width_, (long long) values_[ index_ ] );
    }
}

// -----------------------------------------------------------------------------
// virtual
void ParamInt::help() const
{
    if (type_ == ParamType::Value || type_ == ParamType::List) {
        printf( "    %-16s %s; default %lld\n",
                option_.c_str(), help_.c_str(), (long long) default_value_ );
    }
}

// =============================================================================
// ParamOkay class
// same as ParamInt, but prints pass (for non-zero) or FAILED (for zero).

// -----------------------------------------------------------------------------
// virtual
void ParamOkay::print() const
{
    if (used_ && width_ > 0) {
        const char *msg = "";
        switch (values_[ index_ ]) {
            case  0: msg = "FAILED";   break;
            case  1: msg = "pass";     break;
            case -1: msg = "no check"; break;
        }
        printf( "%-*s  ", width_, msg );
    }
}

// =============================================================================
// ParamInt3 class
// Integer 3-tuple parameters for M x N x K dimensions

// -----------------------------------------------------------------------------
// virtual
void ParamInt3::parse( const char *str )
{
    int64_t m_start, m_end, m_step;
    int64_t n_start, n_end, n_step;
    int64_t k_start, k_end, k_step;
    int len;
    while (true) {
        // scan M
        if (scan_range( &str, &m_start, &m_end, &m_step ) != 0) {
            throw_error( "invalid m dimension at '%s', "
                         "expected integer or range start:end:step", str );
        }
        // if "*", use Cartesian product
        // if "x", use "inner" product
        // if "*" or "x", scan N; else K = N = M
        len = 0;
        bool cartesian = false;
        sscanf( str, " * %n", &len );
        if (len > 0)
            cartesian = true;
        else
            sscanf( str, " x %n", &len );
        if (len > 0) {
            str += len;
            if (scan_range( &str, &n_start, &n_end, &n_step ) != 0) {
                throw_error( "invalid n dimension at '%s', "
                             "expected integer or range start:end:step", str );
            }
            // if "*" or "x", scan K; else K = N
            len = 0;
            if (cartesian)
                sscanf( str, " * %n", &len );
            else
                sscanf( str, " x %n", &len );
            if (len > 0) {
                str += len;
                if (scan_range( &str, &k_start, &k_end, &k_step ) != 0) {
                    throw_error( "invalid k dimension at '%s', "
                                 "expected integer or range start:end:step", str );
                }
            }
            else {
                k_start = n_start;
                k_end   = n_end;
                k_step  = n_step;
            }
        }
        else {
            k_start = n_start = m_start;
            k_end   = n_end   = m_end;
            k_step  = n_step  = m_step;
        }

        if (m_start == m_end && n_start == n_end && k_start == k_end) {
            // single size
            int3_t dim = { m_start, n_start, k_start };
            push_back( dim );
        }
        else if (cartesian) {
            // Cartesian product of M x N x K
            // require non-zero step
            if (m_step == 0) m_step = 1;
            if (n_step == 0) n_step = 1;
            if (k_step == 0) k_step = 1;
            for (int64_t m = m_start;
                 (m_step >= 0 ? m <= m_end : m >= m_end);
                 m += m_step) {

                for (int64_t n = n_start;
                     (n_step >= 0 ? n <= n_end : n >= n_end);
                     n += n_step) {

                    for (int64_t k = k_start;
                         (k_step >= 0 ? k <= k_end : k >= k_end);
                         k += k_step)
                    {
                        int3_t dim = { m, n, k };
                        push_back( dim );
                    }
                }
            }
        }
        else {
            // inner product of M x N x K
            // at least one of the variables must advance
            assert( m_step != 0 || n_step != 0 || k_step != 0 );
            for (int64_t m = m_start,
                         n = n_start,
                         k = k_start;

                 (m_step >= 0 ? m <= m_end : m >= m_end) &&
                 (n_step >= 0 ? n <= n_end : n >= n_end) &&
                 (k_step >= 0 ? k <= k_end : k >= k_end);

                 m += m_step,
                 n += n_step,
                 k += k_step)
            {
                int3_t dim = { m, n, k };
                push_back( dim );
            }
        }
        if (*str == '\0') {
            break;
        }
        if (*str != ',' && *str != ';') {
            throw_error( "invalid argument at '%s', expected comma", str );
        }
        str += 1;
    }
}

// -----------------------------------------------------------------------------
void ParamInt3::push_back( int3_t val )
{
    if (val.m < min_value_ || val.m > max_value_ ||
        val.n < min_value_ || val.n > max_value_ ||
        val.k < min_value_ || val.k > max_value_)
    {
        throw_error( "invalid value, %lld x %lld x %lld outside [%lld, %lld]",
                     (long long) val.m,
                     (long long) val.n,
                     (long long) val.k,
                     (long long) min_value_,
                     (long long) max_value_ );
    }
    TParamBase<int3_t>::push_back( val );
}

// -----------------------------------------------------------------------------
// virtual
void ParamInt3::print() const
{
    if (width_ > 0) {
        if (used_ & m_mask) {
            printf( "%*lld  ", width_, (long long) values_[ index_ ].m );
        }
        if (used_ & n_mask) {
            printf( "%*lld  ", width_, (long long) values_[ index_ ].n );
        }
        if (used_ & k_mask) {
            printf( "%*lld  ", width_, (long long) values_[ index_ ].k );
        }
    }
}

// -----------------------------------------------------------------------------
// for line=0, print blanks
// for line=1, print whichever of m, n, k are used
// virtual
void ParamInt3::header( int line ) const
{
    if (width_ > 0) {
        if (used_ & m_mask) {
            printf( "%*s  ", width_, (line == 0 ? "" : m_name_.c_str()) );
        }
        if (used_ & n_mask) {
            printf( "%*s  ", width_, (line == 0 ? "" : n_name_.c_str()) );
        }
        if (used_ & k_mask) {
            printf( "%*s  ", width_, (line == 0 ? "" : k_name_.c_str()) );
        }
    }
}

// =============================================================================
// ParamComplex class
// -----------------------------------------------------------------------------
// Scans single real or complex value and lists of values.
//
std::complex<double> ParamComplex::scan_complex( const char** str )
{
    char op, suffix;
    double x, y;
    std::complex<double> value;
    int cnt, bytes1, bytes2, bytes3;
    cnt = sscanf( *str, "%lf %n %c %n %lf%c %n", &x, &bytes1, &op, &bytes2, &y, &suffix, &bytes3 );
    if (cnt == 4 && (op == '+' || op == '-') && suffix == 'i') {
        // x+yi or x-yi complex value
        if (op == '+')
            value = std::complex<double>( x, y );
        else
            value = std::complex<double>( x, -y );

        *str += bytes3;
    }
    else if (cnt == 2 && op == 'i') {
        // xi imaginary value
        value = std::complex<double>( 0, x );
        *str += bytes2;
    }
    else if (cnt == 1) {
        // x real value
        value = x;
        *str += bytes1;
    }
    else if (cnt > 1 && op == ',') { //handles case of two real values "2.34,1.11", or real and img "2.34,1.11i"
        // x real value
        value = x;
        *str += bytes2 - 1; //leave comma
    }
    else {
        std::string err = std::string("invalid value '") + *str
                        + "'; expected format like '1.2' or '1.2+3.4i'";
        throw std::runtime_error( err );
    }
    return value;
}
// -----------------------------------------------------------------------------
// virtual
void ParamComplex::parse( const char *str )
{
    //printf("ParamComplex::parse\n");
    while (true) {
        std::complex<double> val = scan_complex(&str);
        TParamBase< std::complex<double> >::push_back( val );
        if (*str == '\0') {
            break;
        }

        if (*str != ',' && *str != ';') {
            throw_error( "invalid argument at '%s', expected comma", str );
        }
        str += 1;
    }
}

//------------------------------------------------------------------------------
// Output the entire field in a fixed width W,
// different than the width passed here. W = 2*width + 3,
// to account for initial -, [+-] between real & complex parts, and i at end.
// For instance, print calls snprintf_value with width=4, precision=2,
// and then prints the resulting string in a field width 11 (%-11s).
//
template <typename scalar_t>
const char* snprintf_value(
    char* buf, size_t buf_len, int width, int precision, scalar_t value)
{
    using real_t = real_type<scalar_t>;

    real_t re = std::real( value );
    real_t im = std::imag( value );
    if (re == int(re) && im == int(im)) {
        // exactly integer, print without digits after decimal point
        if (im != 0) {
            snprintf(buf, buf_len, "% *.0f%c%.0fi",
                     width - precision, re,
                     (im >= 0 ? '+' : '-'),
                     std::abs(im));
        }
        else {
            snprintf(buf, buf_len, "% *.0f%*s",
                     width - precision, re, precision, "");
        }
    }
    else {
        // general case
        if (im != 0) {
            snprintf(buf, buf_len, "% *.*f%c%.*fi",
                     width, precision, re,
                     (im >= 0 ? '+' : '-'),
                     precision, std::abs(im));
        }
        else {
            snprintf(buf, buf_len, "% *.*f",
                     width, precision, re);
        }
    }
    return buf;
}


// -----------------------------------------------------------------------------
/// If field has been used, prints the value.
/// If value is set to no_data_flag, it prints "NA".
/// The output width and precision are set in the constructor.
// virtual
void ParamComplex::print() const
{
    char buf[ 1000 ];
    if (used_ && display_width_ > 0) {
        if (same( no_data_flag, values_[ index_ ].real() )) {  //TODO: check also for imaginary?
            printf( "%*s  ", display_width_, "NA" );
        }
        else {

            snprintf_value( buf, sizeof(buf), display_width_, precision_, values_[ index_ ] );
            printf( "%-*s  ",width_, buf);
        }
    }
}

// =============================================================================
// ParamDouble class
// Double precision parameter

// -----------------------------------------------------------------------------
// virtual
void ParamDouble::parse( const char *str )
{
    double start = 0, end = 0, step = 0;
    while (true) {
        if (scan_range( &str, &start, &end, &step ) != 0) {
            throw_error( "invalid argument at '%s', "
                         "expected float or range start:end:step", str );
        }
        if (start == end) {
            push_back( start );
        }
        else {
            end += step / 10.;  // avoid rounding issues
            for (double val = start; val <= end; val += step) {
                push_back( val );
            }
        }
        if (*str == '\0') {
            break;
        }
        if (*str != ',' && *str != ';') {
            throw_error( "invalid argument at '%s', expected comma", str );
        }
        str += 1;
    }
}

// -----------------------------------------------------------------------------
void ParamDouble::push_back( double val )
{
    if (val < min_value_ || val > max_value_) {
        throw_error( "invalid argument, %.*f outside [%.*f, %.*f]",
                     precision_, val,
                     precision_, min_value_,
                     precision_, max_value_ );
    }
    TParamBase<double>::push_back( val );
}

// -----------------------------------------------------------------------------
/// If field has been used, prints the floating point value.
/// If value is set to no_data_flag, it prints "NA".
/// If value < 1, it prints with precision (p) significant digits (%.pg).
/// Otherwise, it prints with precision (p) digits after the decimal point (%.pf).
/// The output width and precision are set in the constructor.
// virtual
void ParamDouble::print() const
{
    if (used_ && width_ > 0) {
        if (same( no_data_flag, values_[ index_ ] )) {
            printf( "%*s  ", width_, "NA" );
        }
        else {
            if (std::abs( values_[ index_ ] ) < 1)
                printf( "%#*.*g  ", width_, precision_, values_[ index_ ] );
            else
                printf( "%*.*f  ", width_, precision_, values_[ index_ ] );
        }
    }
}

// -----------------------------------------------------------------------------
// virtual
void ParamDouble::help() const
{
    if (type_ == ParamType::Value || type_ == ParamType::List) {
        printf( "    %-16s %s; default ",
                option_.c_str(), help_.c_str() );
        if (same( no_data_flag, default_value_ )) {
            printf( "NA\n" );
        }
        else {
            printf( "%.*f\n", precision_, default_value_ );
        }
    }
}

// =============================================================================
// ParamScientific class
// same as ParamDouble, but prints using scientific notation (%e)

// -----------------------------------------------------------------------------
// virtual
void ParamScientific::print() const
{
    if (used_ && width_ > 0) {
        if (same( no_data_flag, values_[ index_ ] )) {
            printf( "%*s  ", width_, "NA" );
        }
        else {
            printf( "%*.*e  ", width_, precision_, values_[ index_ ] );
        }
    }
}

// -----------------------------------------------------------------------------
// virtual
void ParamScientific::help() const
{
    if (type_ == ParamType::Value || type_ == ParamType::List) {
        printf( "    %-16s %s; default ",
                option_.c_str(), help_.c_str() );
        if (same( no_data_flag, default_value_ )) {
            printf( "NA\n" );
        }
        else {
            printf( "%.*e\n", precision_, default_value_ );
        }
    }
}

// =============================================================================
// ParamString class
// String parameters

// -----------------------------------------------------------------------------
// Return true if str is in the list of valid strings.
// If no valid strings are set, always returns true.
bool ParamString::is_valid( const std::string& str )
{
    if (valid_.size() == 0)
        return true;
    for (auto iter = valid_.begin(); iter != valid_.end(); ++iter) {
        if (str == *iter)
            return true;
    }
    return false;
}

// -----------------------------------------------------------------------------
// virtual
void ParamString::parse( const char *str )
{
    char* copy = strdup( str );
    char* token = strtok( copy, "," );
    while (token != nullptr) {
        push_back( token );
        token = strtok( nullptr, "," );
    }
    free( copy );
}

// -----------------------------------------------------------------------------
void ParamString::push_back( const char* str )
{
    if (! is_valid(str)) {
        throw_error( "invalid argument '%s'", str );
    }
    TParamBase< std::string >::push_back( str );
}

// -----------------------------------------------------------------------------
// virtual
void ParamString::print() const
{
    if (used_ && width_ > 0) {
        printf( "%-*s  ", width_, values_[ index_ ].c_str() );
    }
}

// -----------------------------------------------------------------------------
// virtual
void ParamString::help() const
{
    if (type_ == ParamType::Value || type_ == ParamType::List) {
        printf( "    %-16s %s; default '%s'",
                option_.c_str(), help_.c_str(),
                default_value_.c_str() );
        if (valid_.size() > 0) {
            printf( "; valid: " );
            for (auto iter = valid_.begin(); iter != valid_.end(); ++iter) {
                printf( "%s ", iter->c_str() );
            }
        }
        printf( "\n" );
    }
}

// -----------------------------------------------------------------------------
// Set the allowed strings
void ParamString::add_valid( const char* str )
{
    valid_.push_back( str );
}

// -----------------------------------------------------------------------------
// for line=0, print blanks
// for line=1, print name (left aligned)
// virtual
void ParamString::header( int line ) const
{
    if (used_ && width_ > 0) {
        printf( "%-*s  ", width_, (line == 0 ? "" : name_.c_str()) );
    }
}

// =============================================================================
// ParamChar class
// Character parameters

// -----------------------------------------------------------------------------
// virtual
void ParamChar::parse( const char *str )
{
    while (true) {
        char val;
        int len;
        int i = sscanf( str, "%c %n", &val, &len );
        if (i != 1) {
            throw_error( "invalid argument at '%s', expected one char", str );
        }
        str += len;
        push_back( val );
        if (*str == '\0') {
            break;
        }
        if (*str != ',' && *str != ';') {
            throw_error( "invalid argument at '%s', expected comma", str );
        }
        str += 1;
    }
}

// -----------------------------------------------------------------------------
void ParamChar::push_back( char val )
{
    if (valid_.find( val ) == std::string::npos) {  // not found
        throw_error( "invalid option, %c not in [%s]", val, valid_.c_str() );
    }
    TParamBase<char>::push_back( val );
}

// -----------------------------------------------------------------------------
// virtual
void ParamChar::print() const
{
    if (used_ && width_ > 0) {
        printf( "%*c  ", width_, values_[ index_ ] );
    }
}

// -----------------------------------------------------------------------------
// virtual
void ParamChar::help() const
{
    if (type_ == ParamType::Value || type_ == ParamType::List) {
        printf( "    %-16s %s; default %c; valid: [%s]\n",
                option_.c_str(), help_.c_str(),
                default_value_, valid_.c_str() );
    }
}

// =============================================================================
// ParamsBase class
// List of parameters

// -----------------------------------------------------------------------------
/// Throws QuitException if --help is encountered.
/// Throws std::runtime_error for errors.
void ParamsBase::parse( const char *routine, int n, char **args )
{
    // Usage: test [params] command
    for (int i = 0; i < n; ++i) {
        const char *arg = args[i];
        size_t len = strlen( arg );
        try {
            bool found = false;
            if (strncmp( arg, "-h", 2 ) == 0 ||
                strncmp( arg, "--help", 6 ) == 0)
            {
                throw QuitException();
            }
            for (auto param  = ParamBase::s_params.begin();
                      param != ParamBase::s_params.end();
                    ++param)
            {
                // handles both "--option value" (two arg)
                // and          "--option=value" (one arg)
                size_t plen = (*param)->option_.size();
                if ((len == plen || (len >= plen && arg[plen] == '=')) &&
                    strncmp( arg, (*param)->option_.c_str(), plen ) == 0)
                {
                    if (! (*param)->used()) {
                        throw_error( "invalid parameter for routine '%s'",
                                     routine );
                    }
                    const char *value = nullptr;
                    if (len == plen && i+1 < n) {
                        // --arg value (two arguments)
                        i += 1;
                        value = args[i];
                    }
                    else if (len > plen+1 && arg[plen] == '=') {
                        // --arg=value (one argument)
                        value = arg + (*param)->option_.size() + 1;
                    }
                    else {
                        throw_error( "requires an argument" );
                    }
                    (*param)->parse( value );
                    found = true;
                    break;
                }
            }
            if (! found) {
                throw_error( "invalid parameter" );
            }
        }
        // QuitException is not a runtime_error,
        // so it is caught at a higher level.
        catch (const std::runtime_error& ex) {
            // prepend arg to error message
            throw std::runtime_error( std::string(arg) + ": " + ex.what() );
        }
    }
}

// -----------------------------------------------------------------------------
bool ParamsBase::next()
{
    // "Cartesian product" iteration
    // uses reverse order so in output, parameters on right cycle fastest
    for (auto param = ParamBase::s_params.rbegin();
         param != ParamBase::s_params.rend(); ++param)
    {
        if ((*param)->next()) {
            return true;
        }
    }
    return false;
}

// -----------------------------------------------------------------------------
void ParamsBase::header()
{
    for (int line = 0; line < 2; ++line) {
        for (auto param  = ParamBase::s_params.begin();
                  param != ParamBase::s_params.end();
                ++param)
        {
            (*param)->header( line );
        }
        printf( "\n" );
    }
}

// -----------------------------------------------------------------------------
void ParamsBase::print()
{
    for (auto param  = ParamBase::s_params.begin();
              param != ParamBase::s_params.end();
            ++param)
    {
        (*param)->print();
    }
    printf( "\n" );
}

// -----------------------------------------------------------------------------
void ParamsBase::reset_output()
{
    for (auto param  = ParamBase::s_params.begin();
              param != ParamBase::s_params.end();
            ++param)
    {
        (*param)->reset_output();
    }
}

// -----------------------------------------------------------------------------
void ParamsBase::help( const char *routine )
{
    // todo: pass in argv[0] instead of assuming test.
    printf( "%sUsage:%s test [-h|--help]\n"
            "       test [-h|--help] routine\n"
            "       test [parameters] routine\n\n"
            "%sParameters for %s:%s\n",
            ansi_bold, ansi_normal,
            ansi_bold, routine, ansi_normal );
    for (auto param  = ParamBase::s_params.begin();
              param != ParamBase::s_params.end();
            ++param)
    {
        if ((*param)->used_ && (*param)->type_ == ParamType::Value)
            (*param)->help();
    }
    printf( "\n%sParameters that take comma-separated list of values and may be repeated:%s\n",
            ansi_bold, ansi_normal );
    for (auto param  = ParamBase::s_params.begin();
              param != ParamBase::s_params.end();
            ++param)
    {
        if ((*param)->used_ && (*param)->type_ == ParamType::List)
            (*param)->help();
    }
}

// -----------------------------------------------------------------------------
test_func_ptr find_tester(
    const char *name,
    std::vector< routines_t >& routines )
{
    for (size_t i = 0; i < routines.size(); ++i) {
        if (strcmp( name, routines[i].name ) == 0) {
            return routines[i].func;
        }
    }
    return nullptr;
}

// -----------------------------------------------------------------------------
void usage(
    int argc, char **argv,
    std::vector< routines_t >& routines,
    const char **section_names,
    int col_width, int ncols )
{
    using namespace testsweeper;
    printf( "%sUsage:%s %s [-h|--help]\n"
            "       %s [-h|--help] routine\n"
            "       %s [parameters] routine\n\n"
            "%sAvailable routines:%s",
            ansi_bold, ansi_normal,
            argv[0], argv[0], argv[0],
            ansi_bold, ansi_normal );
    int last_section = 0;
    int cnt = 0;
    for (size_t i = 0; i < routines.size(); ++i) {
        if (routines[i].section == 0) {
            cnt = 0;  // force newline before next routine
            continue;
        }
        if (routines[i].section != last_section) {
            last_section = routines[i].section;
            cnt = 0;
            printf( "\n\n%s%s%s\n",
                    ansi_bold, section_names[last_section], ansi_normal );
        }
        else if (cnt % ncols == 0) {
            printf( "\n" );
        }
        printf( "  %-*s", col_width, routines[i].name );
        cnt += 1;
    }
    printf( "\n" );
}

// -----------------------------------------------------------------------------
// prefer OpenMP get_wtime; else use gettimeofday
double get_wtime()
{
    #ifdef _OPENMP
        return omp_get_wtime();
    #else
        struct timeval tv;
        gettimeofday( &tv, nullptr );
        return tv.tv_sec + tv.tv_usec*1e-6;
    #endif
}

} // namespace testsweeper
