// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_INTERNAL_HH
#define BLAS_INTERNAL_HH

#include "blas/util.hh"

namespace blas {

//------------------------------------------------------------------------------
/// @see to_blas_int
///
inline blas_int to_blas_int_( int64_t x, const char* x_str )
{
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if_msg( x > std::numeric_limits<blas_int>::max(), "%s", x_str );
    }
    return blas_int( x );
}

//----------------------------------------
/// Convert int64_t to blas_int.
/// If blas_int is 64-bit, this does nothing.
/// If blas_int is 32-bit, throws if x > INT_MAX, so conversion would overflow.
///
/// Note this is in src/blas_internal.hh, so this macro won't pollute
/// the namespace when apps #include <blas.hh>.
///
#define to_blas_int( x ) to_blas_int_( x, #x )

}  // namespace blas

#endif // BLAS_INTERNAL_HH
