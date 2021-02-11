// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_DEVICE_HH
#define BLAS_DEVICE_HH

#include "blas/util.hh"
#include "blas/defines.h"

#ifdef BLAS_HAVE_CUBLAS
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
#elif defined(BLAS_HAVE_ROCBLAS)
    /* Default to HCC platform on ROCm */
    #if !defined(__HIP_PLATFORM_NVCC__) && !defined(__HIP_PLATFORM_HCC__)
      #define __HIP_PLATFORM_HCC__
    #endif
    #include <hip/hip_runtime.h>
    #include <rocblas.h>
#elif defined(BLAS_HAVE_ONEMKL)
#ifdef __GNUC__
#include <CL/sycl/detail/cl.h> // for CL version
#endif
#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#endif

namespace blas {

// -----------------------------------------------------------------------------
// types
#ifdef BLAS_HAVE_ONEMKL
typedef cl::sycl::device Device;
#else
typedef int Device;
#endif

#ifdef BLAS_HAVE_CUBLAS
    typedef int                  device_blas_int;
#elif defined(BLAS_HAVE_ROCBLAS)
    typedef int                  device_blas_int;
#elif defined(BLAS_HAVE_ONEMKL)
    typedef std::int64_t         device_blas_int;
#else
    typedef int                  device_blas_int;
#endif

// -----------------------------------------------------------------------------
enum class MemcpyKind : device_blas_int {
    HostToHost     = 0,
    HostToDevice   = 1,
    DeviceToHost   = 2,
    DeviceToDevice = 3,
    Default        = 4,
};

// -----------------------------------------------------------------------------
#if defined(BLAS_HAVE_CUBLAS)
    /// @return the corresponding cuda memcpy kind constant
    inline cudaMemcpyKind memcpy2cuda( MemcpyKind kind )
    {
        switch (kind) {
            case MemcpyKind::HostToHost:     return cudaMemcpyHostToHost;     break;
            case MemcpyKind::HostToDevice:   return cudaMemcpyHostToDevice;   break;
            case MemcpyKind::DeviceToHost:   return cudaMemcpyDeviceToHost;   break;
            case MemcpyKind::DeviceToDevice: return cudaMemcpyDeviceToDevice; break;
            case MemcpyKind::Default:        return cudaMemcpyDefault;
            default: throw blas::Error( "unknown memcpy direction" );
        }
    }
#elif defined(BLAS_HAVE_ROCBLAS)
    /// @return the corresponding hip memcpy kind constant
    inline hipMemcpyKind memcpy2hip( MemcpyKind kind )
    {
        switch (kind) {
            case MemcpyKind::HostToHost:     return hipMemcpyHostToHost;     break;
            case MemcpyKind::HostToDevice:   return hipMemcpyHostToDevice;   break;
            case MemcpyKind::DeviceToHost:   return hipMemcpyDeviceToHost;   break;
            case MemcpyKind::DeviceToDevice: return hipMemcpyDeviceToDevice; break;
            case MemcpyKind::Default:        return hipMemcpyDefault;
            default: throw blas::Error( "unknown memcpy direction" );
        }
    }
#elif defined(BLAS_HAVE_ONEMKL)
    /// @return the corresponding sycl memcpy kind constant
    /*
     * the memcpy method in the sycl::queue class does not accept
     * a direction (i.e. always operates in default mode).
     * for interface compatibility with cuda/hip, return a default integer
    */
    inline int64_t memcpy2sycl( MemcpyKind kind ) { return 0; }
#endif

// -----------------------------------------------------------------------------
// constants
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

    blas::Device           device() const { return device_; }
    void                   sync();
    size_t                 get_batch_limit() { return batch_limit_; }
    void**                 get_dev_ptr_array();

    // switch from default stream to parallel streams
    void fork();

    // switch back to the default stream
    void join();

    // return the next-in-line stream (for both default and fork modes)
    void revolve();

    #ifdef BLAS_HAVE_CUBLAS
        cudaStream_t   stream() const { return *current_stream_; }
        cublasHandle_t handle() const { return handle_; }
    #elif defined(BLAS_HAVE_ROCBLAS)
        hipStream_t    stream() const { return *current_stream_; }
        rocblas_handle handle() const { return handle_; }
    #elif defined(BLAS_HAVE_ONEMKL)
        cl::sycl::queue  stream()     const { return  default_stream_; }
        cl::sycl::queue* stream_ptr() const { return &default_stream_; }
    #endif

private:
    // associated device ID
    blas::Device device_;

    // max workspace allocated for a batch argument in a single call
    // (e.g. a pointer array)
    size_t batch_limit_;

    // workspace for pointer arrays of batch routines
    void** dev_ptr_array_;

    // the number of streams the queue is currently using for
    // launching kernels (1 by default)
    size_t num_active_streams_;

    // an index to the current stream in use
    size_t current_stream_index_;

    #ifdef BLAS_HAVE_CUBLAS
        // associated device blas handle
        cublasHandle_t handle_;

        // pointer to current stream (default or fork mode)
        cudaStream_t *current_stream_;

        // default CUDA stream for this queue; may be NULL
        cudaStream_t default_stream_;

        // parallel streams in fork mode
        cudaStream_t parallel_streams_[DEV_QUEUE_FORK_SIZE];

        cudaEvent_t  default_event_;
        cudaEvent_t  parallel_events_[DEV_QUEUE_FORK_SIZE];

    #elif defined(BLAS_HAVE_ROCBLAS)
        // associated device blas handle
        rocblas_handle handle_;

        // pointer to current stream (default or fork mode)
        hipStream_t  *current_stream_;

        // default CUDA stream for this queue; may be NULL
        hipStream_t  default_stream_;

        // parallel streams in fork mode
        hipStream_t  parallel_streams_[DEV_QUEUE_FORK_SIZE];

        hipEvent_t   default_event_;
        hipEvent_t   parallel_events_[DEV_QUEUE_FORK_SIZE];
    #elif defined(BLAS_HAVE_ONMKL)
        // default sycl queue for this blas queue
        cl::sycl::queue  default_stream_;

        cl::sycl::event  default_event_;
    #else
        // pointer to current stream (default or fork mode)
        void** current_stream_;

        // default CUDA stream for this queue; may be NULL
        void*  default_stream_;
    #endif
};

// -----------------------------------------------------------------------------
// Light wrappers around CUDA and cuBLAS functions.
#ifdef BLAS_HAVE_CUBLAS

inline bool is_device_error( cudaError_t error )
{
    return (error != cudaSuccess);
}

inline bool is_device_error( cublasStatus_t error )
{
    return (error != CUBLAS_STATUS_SUCCESS);
}

inline const char* device_error_string( cudaError_t error )
{
    return cudaGetErrorString( error );
}

// see device_error.cc
const char* device_error_string( cublasStatus_t error );

#endif  // HAVE_CUBLAS

// -----------------------------------------------------------------------------
// Light wrappers around HIP and rocBLAS functions.
#ifdef BLAS_HAVE_ROCBLAS

inline bool is_device_error( hipError_t error )
{
    return (error != hipSuccess);
}

inline bool is_device_error( rocblas_status error )
{
    return (error != rocblas_status_success);
}

inline const char* device_error_string( hipError_t error )
{
    return hipGetErrorString( error );
}

inline const char* device_error_string( rocblas_status error )
{
    return rocblas_status_to_string( error );
}

#endif  // HAVE_ROCBLAS

// -----------------------------------------------------------------------------
// device errors
#if defined(BLAS_ERROR_NDEBUG) || (defined(BLAS_ERROR_ASSERT) && defined(NDEBUG))

    // blaspp does no error checking on device errors;
    #define blas_dev_call( error ) \
        ((void)0)

#elif defined(BLAS_ERROR_ASSERT)

    // blaspp aborts on device errors
    #if defined(BLAS_HAVE_ONEMKL)
    #define blas_dev_call( error ) \
        try { \
            error; \
        } \
        catch (cl::sycl::exception const& e) { \
            blas::internal::abort_if( true, __func__, \
                                      "%s", e.what() ); \
        } \
        catch (std::exception const& e) { \
            blas::internal::abort_if( true, __func__, \
                                      "%s", e.what() ); \
        } \
        catch (const char* msg) { \
            blas::internal::abort_if( true, __func__, \
                                      "%s", msg ); \
        }
    #else
    #define blas_dev_call( error ) \
        do { \
            auto e = error; \
            blas::internal::abort_if( blas::is_device_error(e), __func__, \
                                      "%s", blas::device_error_string(e) ); \
        } while(0)
    #endif

#else

    // blaspp throws device errors (default)
    #if defined(BLAS_HAVE_ONEMKL)
    try { \
            error; \
        } \
        catch (cl::sycl::exception const& e) { \
            blas::internal::throw_if( true, \
                                      e.what(), __func__ ); \
        } \
        catch (std::exception const& e) { \
            blas::internal::throw_if( true, \
                                      e.what(), __func__ ); \
        } \
        catch (const char* msg) { \
            blas::internal::throw_if( true, \
                                      msg, __func__ ); \
        }
    #else
    #define blas_dev_call( error ) \
        do { \
            auto e = error; \
            blas::internal::throw_if( blas::is_device_error(e), \
                                      blas::device_error_string(e), __func__ ); \
        } while(0)
    #endif

#endif

// -----------------------------------------------------------------------------
// set/get device functions
void set_device( blas::Device device );
void get_device( blas::Device *device );
device_blas_int get_device_count();

// -----------------------------------------------------------------------------
// memory functions
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
    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cudaMalloc( (void**)&ptr, nelements * sizeof(T) ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
        blas_dev_call(
            hipMalloc( (void**)&ptr, nelements * sizeof(T) ) );

    #elif defined(BLAS_HAVE_ONEMKL)
        // SYCK requires a device or queue to malloc
        throw blas::Error( "unsupported function for sycl backend", __func__ );
    #endif
    return ptr;
}

//------------------------------------------------------------------------------
/// @return a device pointer to an allocated memory space on specific device
template <typename T>
T* device_malloc(
    blas::Device device, int64_t nelements)
{
    T* ptr = nullptr;
    #ifdef BLAS_HAVE_CUBLAS
        blas::set_device( device );
        blas_dev_call(
                cudaMalloc( (void**)&ptr, nelements * sizeof(T) ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
        blas::set_device( device );
        blas_dev_call(
                hipMalloc( (void**)&ptr, nelements * sizeof(T) ) );

    #ifdef BLAS_HAVE_ONEMKL
        cl::sycl::queue tmp_queue( device );
        blas_dev_call(
            ptr = (T*)cl::sycl::malloc_device(nelements*sizeof(T), tmp_queue) );

    #else

        throw blas::Error( "device BLAS not available", __func__ );
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
    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cudaMallocHost( (void**)&ptr, nelements * sizeof(T) ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
        blas_dev_call(
            hipHostMalloc( (void**)&ptr, nelements * sizeof(T) ) );

    #elif defined(BLAS_HAVE_ONEMKL)
        /*
         * disable for now on intel gpus
        */
        throw blas::Error( "unsupported function for sycl backend", __func__ );

    #else
        throw blas::Error( "device BLAS not available", __func__ );
    #endif
    return ptr;
}

//------------------------------------------------------------------------------
// device set matrix
template <typename T>
void device_setmatrix(
    int64_t m, int64_t n,
    T const* host_ptr, int64_t ldh,
    T* dev_ptr, int64_t ldd, Queue& queue)
{
    device_blas_int m_   = device_blas_int( m );
    device_blas_int n_   = device_blas_int( n );
    device_blas_int ldd_ = device_blas_int( ldd );
    device_blas_int ldh_ = device_blas_int( ldh );

    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cublasSetMatrixAsync(
                m_, n_, sizeof(T),
                host_ptr, ldh_,
                dev_ptr,  ldd_, queue.stream() ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
        blas_dev_call(
            rocblas_set_matrix_async(
                m_, n_, sizeof(T),
                host_ptr, ldh_,
                dev_ptr,  ldd_, queue.stream() ) );

    #elif defined(BLAS_HAVE_ONEMKL)
        if( ldh_ == m_ && ldd_ == m_ ) {
            /* one memcpy */
            blas_dev_call(
                (queue.stream_ptr())->memcpy(dev_ptr, host_ptr, m_*n_*sizeof(T));
            );
        }
        else {
            /* will have to do several mem-copies
             * sycl does not support set/get matrix
            */
            T *dptr=NULL, *hptr=NULL;
            for(int64_t ic = 0; ic < n_; ++ic) {
                dptr = dev_ptr[ic*ldd_];
                hptr = host_ptr[ic*ldh_];
                blas_dev_call(
                    (queue.stream_ptr())->memcpy(dptr, hptr, m_*sizeof(T)) );
            }
        }

    #else
        throw blas::Error( "device BLAS not available", __func__ );
        blas_unused( m_ );
        blas_unused( n_ );
        blas_unused( ldd_ );
        blas_unused( ldh_ );
    #endif
}

//------------------------------------------------------------------------------
// device get matrix
template <typename T>
void device_getmatrix(
    int64_t m, int64_t n,
    T const* dev_ptr,  int64_t ldd,
    T*       host_ptr, int64_t ldh, Queue& queue)
{
    device_blas_int m_   = device_blas_int( m );
    device_blas_int n_   = device_blas_int( n );
    device_blas_int ldd_ = device_blas_int( ldd );
    device_blas_int ldh_ = device_blas_int( ldh );

    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cublasGetMatrixAsync(
                m_, n_, sizeof(T),
                dev_ptr,  ldd_,
                host_ptr, ldh_, queue.stream() ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
        blas_dev_call(
            rocblas_get_matrix_async(
                m_, n_, sizeof(T),
                dev_ptr,  ldd_,
                host_ptr, ldh_, queue.stream() ) );

    #elif defined(BLAS_HAVE_ONEMKL)
        if( ldh_ == m_ && ldd_ == m_ ) {
            /* one memcpy */
            blas_dev_call(
                (queue.stream_ptr())->memcpy(host_ptr, dev_ptr, m_*n_*sizeof(T)) );
        }
        else {
            /* will have to do several mem-copies
             * sycl does not support set/get matrix
            */
            T *dptr=NULL, *hptr=NULL;
            for(int64_t ic = 0; ic < n_; ++ic) {
                dptr = dev_ptr[ic*ldd_];
                hptr = host_ptr[ic*ldh_];
                blas_dev_call(
                    (queue.stream_ptr())->memcpy(hptr, dptr, m_*sizeof(T)) );
            }
        }

    #else
        throw blas::Error( "device BLAS not available", __func__ );
        blas_unused( m_ );
        blas_unused( n_ );
        blas_unused( ldd_ );
        blas_unused( ldh_ );
    #endif
}

//------------------------------------------------------------------------------
// device set vector
template <typename T>
void device_setvector(
    int64_t n,
    T const* host_ptr, int64_t inch,
    T*       dev_ptr,  int64_t incd, Queue& queue)
{
    device_blas_int n_    = device_blas_int( n );
    device_blas_int incd_ = device_blas_int( incd );
    device_blas_int inch_ = device_blas_int( inch );

    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cublasSetVectorAsync(
                n_, sizeof(T),
                host_ptr, inch_,
                dev_ptr,  incd_, queue.stream() ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
        blas_dev_call(
            rocblas_set_vector_async(
                n_, sizeof(T),
                host_ptr, inch_,
                dev_ptr,  incd_, queue.stream() ) );

    #elif defined(BLAS_HAVE_ONEMKL)
        if( inch_ == incd_ ) {
            /* this could be slow if inc >> n */
            blas_dev_call(
                (queue.stream_ptr())->memcpy(host_ptr, dev_ptr, n_*inch_*sizeof(T)) );
        }
        else {
            T *hptr=NULL, *dptr=NULL;
            for(int64_t ie = 0; ie < n_; ++ie) {
                hptr = host_ptr + ie * inch;
                pptr = dev_ptr  + ie * incd;
                blas_dev_call(
                    (queue.stream_ptr())->memcpy(hptr, dptr, sizeof(T)) );
            }
        }

    #else
        throw blas::Error( "device BLAS not available", __func__ );
        blas_unused( n_ );
        blas_unused( incd_ );
        blas_unused( inch_ );
    #endif
}

//------------------------------------------------------------------------------
// device get vector
template <typename T>
void device_getvector(
    int64_t n,
    T const* dev_ptr,  int64_t incd,
    T*       host_ptr, int64_t inch, Queue& queue)
{
    device_blas_int n_    = device_blas_int( n );
    device_blas_int incd_ = device_blas_int( incd );
    device_blas_int inch_ = device_blas_int( inch );

    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cublasGetVectorAsync(
                n_, sizeof(T),
                dev_ptr,  incd_,
                host_ptr, inch_, queue.stream() ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
        blas_dev_call(
            rocblas_get_vector_async(
                n_, sizeof(T),
                dev_ptr,  incd_,
                host_ptr, inch_, queue.stream() ) );

    #elif defined(BLAS_HAVE_ONEMKL)
        if( inch_ == incd_ ) {
            /* this could be slow if inc >> n */
            blas_dev_call(
                (queue.stream_ptr())->memcpy(host_ptr, dev_ptr, n_*inch_*sizeof(T)) );
        }
        else {
            T *hptr=NULL, *dptr=NULL;
            for(int64_t ie = 0; ie < n_; ++ie) {
                hptr = host_ptr + ie * inch;
                pptr = dev_ptr  + ie * incd;
                blas_dev_call(
                    (queue.stream_ptr())->memcpy(hptr, dptr, sizeof(T)) );
            }
        }

    #else
        throw blas::Error( "device BLAS not available", __func__ );
        blas_unused( n_ );
        blas_unused( incd_ );
        blas_unused( inch_ );
    #endif
}

//------------------------------------------------------------------------------
// device memset
template <typename T>
void device_memset(
    T* ptr,
    int value, int64_t nelements, Queue& queue)
{
    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cudaMemsetAsync(
                ptr, value,
                nelements * sizeof(T), queue.stream() ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
        blas_dev_call(
            hipMemsetAsync(
                ptr, value,
                nelements * sizeof(T), queue.stream() ) );

    #elif defined(BLAS_HAVE_ONEMKL)
        blas_dev_call(
            (queue.stream_ptr())->memset(ptr, value, nelements * sizeof(T)) );

    #else
        throw blas::Error( "device BLAS not available", __func__ );
    #endif
}

//------------------------------------------------------------------------------
// device memcpy
template <typename T>
void device_memcpy(
    T*        dev_ptr,
    T const* host_ptr,
    int64_t nelements, MemcpyKind kind, Queue& queue)
{
    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cudaMemcpyAsync(
                dev_ptr, host_ptr, sizeof(T)*nelements,
                memcpy2cuda(kind), queue.stream() ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
        blas_dev_call(
            hipMemcpyAsync(
                dev_ptr, host_ptr, sizeof(T)*nelements,
                memcpy2hip(kind), queue.stream() ) );

    #elif defined(BLAS_HAVE_ONEMKL)
        blas_dev_call(
            (queue.stream_ptr())->memcpy(dev_ptr, host_ptr, sizeof(T)*nelements) );

    #else
        throw blas::Error( "device BLAS not available", __func__ );
    #endif
}

// overloaded device memcpy with memcpy direction set to default
template <typename T>
void device_memcpy(
    T*        dev_ptr,
    T const* host_ptr,
    int64_t nelements, Queue& queue)
{
    device_memcpy<T>(
        dev_ptr,
        host_ptr,
        nelements, MemcpyKind::Default, queue);
}

//------------------------------------------------------------------------------
// device memcpy 2D
template <typename T>
void device_memcpy_2d(
    T*        dev_ptr, int64_t  dev_pitch,
    T const* host_ptr, int64_t host_pitch,
    int64_t width, int64_t height, MemcpyKind kind, Queue& queue)
{
    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cudaMemcpy2DAsync(
                 dev_ptr, sizeof(T)* dev_pitch,
                host_ptr, sizeof(T)*host_pitch,
                sizeof(T)*width, height, memcpy2cuda(kind), queue.stream() ) );
    #elif defined(BLAS_HAVE_ROCBLAS)
         blas_dev_call(
            hipMemcpy2DAsync(
                 dev_ptr, sizeof(T)* dev_pitch,
                host_ptr, sizeof(T)*host_pitch,
                sizeof(T)*width, height, memcpy2hip(kind), queue.stream() ) );

    #elif defined(BLAS_HAVE_ONEMKL)
        throw blas::Error( "unsupported function for sycl backend", __func__ );

    #else
        throw blas::Error( "device BLAS not available", __func__ );
    #endif
}

// overloaded device memcpy 2D with memcpy direction set to default
template <typename T>
void device_memcpy_2d(
    T*        dev_ptr, int64_t  dev_pitch,
    T const* host_ptr, int64_t host_pitch,
    int64_t width, int64_t height, Queue& queue)
{
    device_memcpy_2d<T>(
         dev_ptr,  dev_pitch,
        host_ptr, host_pitch,
        width, height, MemcpyKind::Default, queue);
}

}  // namespace blas

#endif        //  #ifndef BLAS_DEVICE_HH
