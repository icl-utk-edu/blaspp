#ifndef DEVICE_HH
#define DEVICE_HH

#if defined(HAVE_CUBLAS)
// -----------------------------------------------------------------------------
// cuda/cublas headers
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#elif defined(HAVE_ROCBLAS)
// -----------------------------------------------------------------------------
// TODO: rocblas headers

#endif

#include "blas_util.hh"
#include "device_types.hh"
#include "device_error.hh"
#include "device_utils.hh"
#include "device_queue.hh"
#include "device_copy.hh"
#include "device_blas_names.hh"

#endif        //  #ifndef DEVICE_HH

