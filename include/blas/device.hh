// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_DEVICE_HH
#define BLAS_DEVICE_HH

#include "blas/util.hh"
#include "blas/device_types.hh"

#ifdef BLASPP_WITH_CUBLAS
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
#endif

namespace blas {

typedef int Device;

const int DEV_QUEUE_DEFAULT_BATCH_LIMIT = 50000;
const int DEV_QUEUE_FORK_SIZE           = 10;

//==============================================================================
// device queue
class Queue
{
public:
     Queue();
     Queue( blas::Device device, int64_t batch_size );
    ~Queue();

    blas::Device           device();
    device_blas_handle_t   handle();
    void                   sync();
    size_t                 get_batch_limit();
    void**                 get_devPtrArray();

    // switch from default stream to parallel streams
    void fork();

    // switch back to the default stream
    void join();

    // return the next-in-line stream (for both default and fork modes)
    void revolve();

    #ifdef BLASPP_WITH_CUBLAS
        cudaStream_t stream();
    #elif defined(HAVE_ROCBLAS)
        // TODO: add similar functionality for rocBLAS, if required
    #endif

private:
    // associated device ID
    blas::Device          device_;

    // associated device blas handle
    device_blas_handle_t  handle_;

    // max workspace allocated for a batch argument in a single call
    // (e.g. a pointer array)
    size_t                batch_limit_;

    // workspace for pointer arrays of batch routines
    void**                devPtrArray;

    #ifdef BLASPP_WITH_CUBLAS
        // the number of streams the queue is currently using for
        // launching kernels (1 by default)
        size_t           num_active_streams_;

        // an index to the current stream in use
        size_t           current_stream_index_;

        // pointer to current stream (default or fork mode)
        cudaStream_t     *current_stream_;

        // default CUDA stream for this queue; may be NULL
        cudaStream_t     default_stream_;

        // parallel streams in fork mode
        cudaStream_t     parallel_streams_[DEV_QUEUE_FORK_SIZE];

        cudaEvent_t      default_event_;
        cudaEvent_t      parallel_events_[DEV_QUEUE_FORK_SIZE];
    #elif defined(HAVE_ROCBLAS)
        // TODO: stream for rocBLAS
    #endif
};

// -----------------------------------------------------------------------------
// device errors
bool is_device_error( device_error_t error );

bool is_device_error( device_blas_status_t status );

const char* device_error_string( device_error_t error );

const char* device_error_string( device_blas_status_t status );

#if defined(BLAS_ERROR_NDEBUG) || (defined(BLAS_ERROR_ASSERT) && defined(NDEBUG))

    // blaspp does no error checking on device errors;
    #define blas_cuda_call( error ) \
        ((void)0)

    // blaspp does no status checking on device blas errors;
    #define blas_cublas_call( status ) \
        ((void)0)

#elif defined(BLAS_ERROR_ASSERT)

    // blaspp aborts on device errors
    #define blas_cuda_call( error ) \
        do { \
            device_error_t e = error; \
            blas::internal::abort_if( blas::is_device_error(e), __func__, \
                                      "%s", blas::device_error_string(e) ); \
        } while(0)

    // blaspp aborts on device blas errors
    #define blas_cublas_call( status )  \
        do { \
            device_blas_status_t s = status; \
            blas::internal::abort_if( blas::is_device_error(s), __func__, \
                                      "%s", blas::device_error_string(s) ); \
        } while(0)

#else

    // blaspp throws device errors (default)
    #define blas_cuda_call( error ) \
        do { \
            device_error_t e = error; \
            blas::internal::throw_if( blas::is_device_error(e), \
                                      blas::device_error_string(e), __func__ ); \
        } while(0)

    // blaspp throws device blas errors (default)
    #define blas_cublas_call( status ) \
        do { \
            device_blas_status_t s = status; \
            blas::internal::throw_if( blas::is_device_error(s), \
                                      blas::device_error_string(s), __func__ ); \
        } while(0)

#endif

// -----------------------------------------------------------------------------
// set/get device functions
void set_device( blas::Device device );
void get_device( blas::Device *device );

// -----------------------------------------------------------------------------
// conversion functions
device_trans_t   device_trans_const( blas::Op trans );
device_diag_t    device_diag_const( blas::Diag diag );
device_uplo_t    device_uplo_const( blas::Uplo uplo );
device_side_t    device_side_const( blas::Side side );

void device_free( void* ptr );
void device_free_pinned( void* ptr );

// -----------------------------------------------------------------------------
// Template functions declared here
// -----------------------------------------------------------------------------

//------------------------------------------------------------------------------
/// @return a device pointer to an allocated memory space
template <typename T>
T* device_malloc(
    int64_t nelements)
{
    T* ptr = nullptr;
    #ifdef BLASPP_WITH_CUBLAS
        blas_cuda_call(
            cudaMalloc( (void**)&ptr, nelements * sizeof(T) ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: allocation for AMD GPUs
    #endif
    return ptr;
}

//------------------------------------------------------------------------------
/// @return a host pointer to a pinned memory space
template <typename T>
T* device_malloc_pinned(
    int64_t nelements)
{
    T* ptr = nullptr;
    #ifdef BLASPP_WITH_CUBLAS
        blas_cuda_call(
            cudaMallocHost( (void**)&ptr, nelements * sizeof(T) ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: allocation using AMD driver API
    #endif
    return ptr;
}

//------------------------------------------------------------------------------
// device set matrix
template <typename T>
void device_setmatrix(
    int64_t m, int64_t n,
    T const* hostPtr, int64_t ldh,
    T* devPtr, int64_t ldd, Queue& queue)
{
    device_blas_int m_   = device_blas_int( m );
    device_blas_int n_   = device_blas_int( n );
    device_blas_int ldd_ = device_blas_int( ldd );
    device_blas_int ldh_ = device_blas_int( ldh );

    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasSetMatrixAsync(
                m_, n_, sizeof(T),
                hostPtr, ldh_,
                devPtr,  ldd_, queue.stream() ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocblas_set_matrix
    #endif
}

//------------------------------------------------------------------------------
// device get matrix
template <typename T>
void device_getmatrix(
    int64_t m, int64_t n,
    T const* devPtr,  int64_t ldd,
    T*       hostPtr, int64_t ldh, Queue& queue)
{
    device_blas_int m_   = device_blas_int( m );
    device_blas_int n_   = device_blas_int( n );
    device_blas_int ldd_ = device_blas_int( ldd );
    device_blas_int ldh_ = device_blas_int( ldh );

    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasGetMatrixAsync(
                m_, n_, sizeof(T),
                devPtr,  ldd_,
                hostPtr, ldh_, queue.stream() ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocblas_get_matrix
    #endif
}

//------------------------------------------------------------------------------
// device set vector
template <typename T>
void device_setvector(
    int64_t n,
    T const* hostPtr, int64_t inch,
    T*       devPtr,  int64_t incd, Queue& queue)
{
    device_blas_int n_    = device_blas_int( n );
    device_blas_int incd_ = device_blas_int( incd );
    device_blas_int inch_ = device_blas_int( inch );

    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasSetVectorAsync(
                n_, sizeof(T),
                hostPtr, inch_,
                devPtr,  incd_, queue.stream() ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocblas_set_vector
    #endif
}

//------------------------------------------------------------------------------
// device get vector
template <typename T>
void device_getvector(
    int64_t n,
    T const* devPtr,  int64_t incd,
    T*       hostPtr, int64_t inch, Queue& queue)
{
    device_blas_int n_    = device_blas_int( n );
    device_blas_int incd_ = device_blas_int( incd );
    device_blas_int inch_ = device_blas_int( inch );

    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasGetVectorAsync(
                n_, sizeof(T),
                devPtr,  incd_,
                hostPtr, inch_, queue.stream() ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocblas_get_vector
    #endif
}

}  // namespace blas

#endif        //  #ifndef BLAS_DEVICE_HH
