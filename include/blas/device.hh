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

