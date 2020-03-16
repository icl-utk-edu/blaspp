// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_DEVICE_HH
#define BLAS_DEVICE_HH

#if defined(BLASPP_WITH_CUBLAS)
// -----------------------------------------------------------------------------
// cuda/cublas headers
#include <cuda_runtime.h>
#include <cublas_v2.h>

#elif defined(HAVE_ROCBLAS)
// -----------------------------------------------------------------------------
// TODO: rocblas headers

#endif

#include "blas/util.hh"
#include "blas/device_types.hh"
#include "blas/device_internals.hh"
#include "blas/device_blas_names.hh"

#endif        //  #ifndef BLAS_DEVICE_HH

