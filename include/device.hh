#ifndef DEVICE_HH
#define DEVICE_HH

#if defined(BLASPP_WITH_CUBLAS)
// -----------------------------------------------------------------------------
// cuda/cublas headers
#include <cuda_runtime.h>
#include <cublas_v2.h>

#elif defined(HAVE_ROCBLAS)
// -----------------------------------------------------------------------------
// TODO: rocblas headers

#endif

#include "blas_util.hh"
#include "device_types.hh"
#include "device_internals.hh"
#include "device_blas_names.hh"

#endif        //  #ifndef DEVICE_HH

