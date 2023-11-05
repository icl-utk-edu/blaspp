// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_DEVICE_HH
#define BLAS_DEVICE_HH

#include "blas/util.hh"
#include "blas/defines.h"

#if defined( BLAS_HAVE_CUBLAS ) \
    || defined( BLAS_HAVE_ROCBLAS ) \
    || defined( BLAS_HAVE_SYCL )
    #define BLAS_HAVE_DEVICE
#endif

#ifdef BLAS_HAVE_CUBLAS
    #include <cuda_runtime.h>
    #include <cublas_v2.h>

#elif defined(BLAS_HAVE_ROCBLAS)
    // Default to HCC platform on ROCm
    #if ! defined(__HIP_PLATFORM_NVCC__) && ! defined(__HIP_PLATFORM_HCC__)
        #define __HIP_PLATFORM_HCC__
        #define BLAS_HIP_PLATFORM_HCC
    #endif

    #include <hip/hip_runtime.h>

    // Headers moved in ROCm 5.2
    #if HIP_VERSION >= 50200000
        #include <rocblas/rocblas.h>
    #else
        #include <rocblas.h>
    #endif

    // If we defined __HIP_PLATFORM_HCC__, undef it.
    #ifdef BLAS_HIP_PLATFORM_HCC
        #undef __HIP_PLATFORM_HCC__
        #undef BLAS_HIP_PLATFORM_HCC
    #endif

#elif defined(BLAS_HAVE_SYCL)
    #include <sycl/detail/cl.h>  // For CL version
    #include <sycl.hpp>

#endif

namespace blas {

// -----------------------------------------------------------------------------
// types
[[deprecated("use int. Remove 2023-12.")]]
typedef int Device;

#ifdef BLAS_HAVE_CUBLAS
    typedef int                  device_blas_int;
#elif defined(BLAS_HAVE_ROCBLAS)
    typedef int                  device_blas_int;
#elif defined(BLAS_HAVE_SYCL)
    typedef std::int64_t         device_blas_int;
#else
    typedef int                  device_blas_int;
#endif

// -----------------------------------------------------------------------------
/// Direction to copy, one of:
///
/// - MemcpyKind::Default:        [recommended] determine direction to copy
///                               based on virtual addresses where src
///                               and dst are allocated.
/// - MemcpyKind::HostToHost:     both src and dst on CPU host.
/// - MemcpyKind::HostToDevice:   src on CPU host, dst on GPU device.
/// - MemcpyKind::DeviceToHost:   src on GPU device, dst on CPU host.
/// - MemcpyKind::DeviceToDevice: both src and dst on GPU devices,
///                               which may be 2 different devices.
///
/// MemcpyKind may be deprecated in the future.
///
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
#elif defined(BLAS_HAVE_SYCL)
    /// @return the corresponding sycl memcpy kind constant
    /// The memcpy method in the sycl::queue class does not accept
    /// a direction (i.e. always operates in default mode).
    /// For interface compatibility with cuda/hip, return a default value
    inline int64_t memcpy2sycl( MemcpyKind kind ) { return 0; }
#endif

// -----------------------------------------------------------------------------
// constants
const int MaxBatchChunk = 50000;

#if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )
    const int MaxForkSize = 10;
#else
    // SYCL and no GPU code doesn't support fork mode.
    const int MaxForkSize = 1;
#endif

//==============================================================================
/// Queue for executing GPU device routines.
/// This wraps CUDA stream and cuBLAS handle,
/// HIP stream and rocBLAS handle,
/// or SYCL queue.
///
class Queue
{
public:
    // Define generic names for vendor types.
    #if defined( BLAS_HAVE_CUBLAS )
        using stream_t = cudaStream_t;
        using event_t  = cudaEvent_t;
        using handle_t = cublasHandle_t;

    #elif defined( BLAS_HAVE_ROCBLAS )
        using stream_t = hipStream_t;
        using event_t  = hipEvent_t;
        using handle_t = rocblas_handle;

    #elif defined( BLAS_HAVE_SYCL )
        using stream_t = sycl::queue;

    #else
        // No GPU code.
        using stream_t = void*;  // unused
    #endif

    Queue();
    Queue( int device );

    [[deprecated("use Queue( device ). Batch size is handled automatically. To be removed 2024-05.")]]
    Queue( int device, int64_t batch_chunk )
        : Queue( device )
    {}

    Queue( int device, stream_t& stream );

    #if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )
        Queue( int device, handle_t handle );
    #endif

    // Disable copying; must construct anew.
    Queue( Queue const& ) = delete;
    Queue& operator=( Queue const& ) = delete;

    ~Queue();

    int  device() const { return device_; }
    void sync();

    /// @return device workspace.
    void* work() { return (void*) work_; }

    /// @return size of device workspace, in scalar_t elements.
    template <typename scalar_t>
    size_t work_size() const { return lwork_ / sizeof(scalar_t); }

    template <typename scalar_t>
    void work_ensure_size( size_t lwork );

    template <typename scalar_t>
    [[deprecated("Use work_ensure_size(). To be removed 2024-05.")]]
    void work_resize( size_t lwork ) { work_ensure_size<scalar_t>( lwork ); }

    // switch from default stream to parallel streams
    void fork( int num_streams=MaxForkSize );

    // switch back to the default stream
    void join();

    // return the next-in-line stream (for both default and fork modes)
    void revolve();

    #if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )
        // Common for CUDA, ROCm.
        void set_handle( handle_t& in_handle );
        handle_t handle() const { return handle_; }
    #endif

    // Common for all: CUDA, ROCm, SYCL, no GPU.
    void set_stream( stream_t& in_stream );

    stream_t& stream()
    {
        #if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )
            return streams_[ current_stream_index_ ];
        #else
            return streams_[ 0 ];
        #endif
    }

private:
    // Workspace for pointer arrays of batch routines or other purposes.
    char* work_;
    size_t lwork_;

    // streams_[ 0 ] is default stream; rest are parallel streams in fork mode.
    stream_t streams_[ MaxForkSize ];

    #if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )
        // Associated device BLAS handle.
        handle_t handle_;

        event_t events_[ MaxForkSize ];

        // The number of streams the queue is currently using for
        // launching kernels (1 by default).
        int num_active_streams_;

        // Index to the current stream in use.
        int current_stream_index_;

        // Whether the queue owns the BLAS handle and default stream,
        // or the user provided them.
        bool own_handle_;
        bool own_default_stream_;
    #endif

    // Associated device ID.
    int device_;
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

    // BLAS++ does no error checking on device errors;
    #define blas_dev_call( error ) \
        error

#elif defined(BLAS_ERROR_ASSERT)

    // BLAS++ aborts on device errors
    #if defined(BLAS_HAVE_SYCL)
        #define blas_dev_call( error ) \
            do { \
                try { \
                    error; \
                } \
                catch (sycl::exception const& e) { \
                    blas::internal::abort_if( true, __func__, \
                                              "%s", e.what() ); \
                } \
                catch (std::exception const& e) { \
                    blas::internal::abort_if( true, __func__, \
                                              "%s", e.what() ); \
                } \
                catch (...) { \
                    blas::internal::abort_if( true, __func__, \
                                              "%s", "unknown exception" ); \
                } \
            } while(0)

    #else
        #define blas_dev_call( error ) \
            do { \
                auto e = error; \
                blas::internal::abort_if( blas::is_device_error(e), __func__, \
                                          "%s", blas::device_error_string(e) ); \
            } while(0)
    #endif

#else

    // BLAS++ throws device errors (default)
    #if defined(BLAS_HAVE_SYCL)
        #define blas_dev_call( error ) \
            do { \
                try { \
                        error; \
                } \
                catch (sycl::exception const& e) { \
                    blas::internal::throw_if( true, \
                                              e.what(), __func__ ); \
                } \
                catch (std::exception const& e) { \
                    blas::internal::throw_if( true, \
                                              e.what(), __func__ ); \
                } \
                catch (...) { \
                    blas::internal::throw_if( true, \
                                              "unknown exception", __func__ ); \
                } \
            } while(0)

    #else
        #define blas_dev_call( error ) \
            do { \
                auto e = error; \
                blas::internal::throw_if( blas::is_device_error(e), \
                                          blas::device_error_string(e), \
                                          __func__ ); \
            } while(0)
    #endif

#endif

// -----------------------------------------------------------------------------
// set/get device functions
[[deprecated("use blas::Queues& with all BLAS++ calls. To be removed 2024-05.")]]
void set_device( int device );

// private, internal routine; sets device for CUDA, ROCm; nothing for SYCL
void internal_set_device( int device );

[[deprecated("use blas::Queues& with all BLAS++ calls. To be removed 2024-05.")]]
void get_device( int *device );

int get_device_count();

// -----------------------------------------------------------------------------
// memory functions

/// @deprecated: use device_free( ptr, queue ).
[[deprecated("use device_free( ptr, queue )")]]
void device_free( void* ptr );

void device_free( void* ptr, blas::Queue &queue );

/// @deprecated: use host_free_pinned( ptr, queue ).
// todo: does this really need to be deprecated
void host_free_pinned( void* ptr );

void host_free_pinned( void* ptr, blas::Queue &queue );

/// @deprecated: use host_free_pinned( ptr, queue ).
[[deprecated("use device_free_pinned( ptr, queue )")]]
inline void device_free_pinned( void* ptr ) {
    host_free_pinned( ptr );
}

/// @deprecated: use host_free_pinned( ptr, queue ).
[[deprecated("use host_free_pinned( ptr, queue )")]]
inline void device_free_pinned( void* ptr, blas::Queue &queue )
{
    host_free_pinned( ptr, queue );
}

// -----------------------------------------------------------------------------
// Template functions declared here
// -----------------------------------------------------------------------------

//------------------------------------------------------------------------------
/// @deprecated: use device_malloc( nelements, queue ).
///
/// @return a device pointer to an allocated memory space
/// In CUDA and ROCm, this version uses current device;
/// in SYCL, this doesn't work since there is no concept of a current device.
///
template <typename T>
[[deprecated("use device_malloc( nelements, queue )")]]
T* device_malloc(
    int64_t nelements)
{
    blas_error_if( nelements < 0 );

    T* ptr = nullptr;
    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cudaMalloc( (void**)&ptr, nelements * sizeof(T) ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
        blas_dev_call(
            hipMalloc( (void**)&ptr, nelements * sizeof(T) ) );

    #elif defined(BLAS_HAVE_SYCL)
        // SYCL requires a device or queue to malloc
        throw blas::Error( "unsupported function for sycl backend", __func__ );

    #else

        throw blas::Error( "device BLAS not available", __func__ );
    #endif
    return ptr;
}

//------------------------------------------------------------------------------
/// @return a pointer to an allocated GPU device memory.
///
/// @param[in] nelements
///     Number of elements of type T to allocate.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///     Determines the GPU device on which to allocate memory.
///
template <typename T>
T* device_malloc(
    int64_t nelements, blas::Queue &queue )
{
    blas_error_if( nelements < 0 );

    T* ptr = nullptr;
    #ifdef BLAS_HAVE_CUBLAS
        blas::internal_set_device( queue.device() );
        blas_dev_call(
                cudaMalloc( (void**)&ptr, nelements * sizeof(T) ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
        blas::internal_set_device( queue.device() );
        blas_dev_call(
                hipMalloc( (void**)&ptr, nelements * sizeof(T) ) );

    #elif defined(BLAS_HAVE_SYCL)
        blas_dev_call(
            ptr = (T*)sycl::malloc_shared( nelements*sizeof(T), queue.stream() ) );

    #else
        throw blas::Error( "device BLAS not available", __func__ );
    #endif
    return ptr;
}

//------------------------------------------------------------------------------
/// @deprecated: use host_malloc_pinned( nelements, queue ).
///
/// @return a host pointer to a pinned memory space.
/// In CUDA and ROCm, this version uses current device;
/// in SYCL, this doesn't work since there is no concept of a current device.
///
template <typename T>
[[deprecated("use host_malloc_pinned( nelements, queue )")]]
T* device_malloc_pinned(
    int64_t nelements)
{
    blas_error_if( nelements < 0 );

    T* ptr = nullptr;
    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cudaMallocHost( (void**)&ptr, nelements * sizeof(T) ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
        blas_dev_call(
            hipHostMalloc( (void**)&ptr, nelements * sizeof(T) ) );

    #elif defined(BLAS_HAVE_SYCL)
        // SYCL requires a device or queue to malloc
        throw blas::Error( "unsupported function for sycl backend", __func__ );

    #else
        throw blas::Error( "device BLAS not available", __func__ );
    #endif
    return ptr;
}

//------------------------------------------------------------------------------
/// @return a pointer to an allocated CPU host memory.
/// In CUDA and ROCm, the memory will be pinned.
/// SYCL does not have an explicit pinned memory allocation, so this
/// just calls sycl::malloc_host.
///
/// @param[in] nelements
///     Number of elements of type T to allocate.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///     In CUDA and ROCm, queue is ignored.
///     In SYCL, queue is passed to sycl::malloc_host to provide context.
///
template <typename T>
T* host_malloc_pinned(
    int64_t nelements, blas::Queue &queue )
{
    blas_error_if( nelements < 0 );

    T* ptr = nullptr;
    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cudaMallocHost( (void**)&ptr, nelements * sizeof(T) ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
        blas_dev_call(
            hipHostMalloc( (void**)&ptr, nelements * sizeof(T) ) );

    #elif defined(BLAS_HAVE_SYCL)
        blas_dev_call(
            ptr = (T*)sycl::malloc_host( nelements*sizeof(T), queue.stream() ) );

    #else
        throw blas::Error( "device BLAS not available", __func__ );
    #endif
    return ptr;
}

//------------------------------------------------------------------------------
/// @deprecated: use host_malloc_pinned( nelements, queue ).
///
template <typename T>
[[deprecated("device_malloc_pinned( nelements, queue )")]]
T* device_malloc_pinned(
    int64_t nelements, blas::Queue &queue )
{
    return host_malloc_pinned<T>( nelements, queue );
}

//------------------------------------------------------------------------------
/// Sets each byte of memory to a constant value (often 0).
/// Asynchronous with respect to host.
///
/// @param[out] ptr
///     Pointer to memory to set.
///
/// @param[in] value
///     Value to set each byte; cast to unsigned char.
///
/// @param[in] nelements
///     Number of elements of type T to set. Sets nelements * sizeof(T) bytes.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename T>
void device_memset(
    T* ptr,
    int value, int64_t nelements, Queue& queue)
{
    blas_error_if( nelements < 0 );

    #ifdef BLAS_HAVE_CUBLAS
        blas::internal_set_device( queue.device() );
        blas_dev_call(
            cudaMemsetAsync(
                ptr, value,
                nelements * sizeof(T), queue.stream() ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
        blas::internal_set_device( queue.device() );
        blas_dev_call(
            hipMemsetAsync(
                ptr, value,
                nelements * sizeof(T), queue.stream() ) );

    #elif defined(BLAS_HAVE_SYCL)
        blas_dev_call(
            queue.stream().memset( ptr, value, nelements * sizeof(T) ) );

    #else
        throw blas::Error( "device BLAS not available", __func__ );
    #endif
}

//------------------------------------------------------------------------------
/// @deprecated: recommend using
///     blas::device_memcpy( dst, src, nelements, queue )
/// instead, which sets kind = MemcpyKind::Default.
///
/// @copydoc device_memcpy(T*,T const*,int64_t,Queue&)
///
/// @param[in] kind
/// @copydoc MemcpyKind
///
/// @see device_memcpy(T*,T const*,int64_t,Queue&)
///
template <typename T>
void device_memcpy(
    T*       dst,
    T const* src,
    int64_t nelements, MemcpyKind kind, Queue& queue)
{
    blas_error_if( nelements < 0 );

    #ifdef BLAS_HAVE_CUBLAS
        blas::internal_set_device( queue.device() );
        blas_dev_call(
            cudaMemcpyAsync(
                dst, src, sizeof(T)*nelements,
                memcpy2cuda(kind), queue.stream() ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
        blas::internal_set_device( queue.device() );
        blas_dev_call(
            hipMemcpyAsync(
                dst, src, sizeof(T)*nelements,
                memcpy2hip(kind), queue.stream() ) );

    #elif defined(BLAS_HAVE_SYCL)
        blas_dev_call(
            queue.stream().memcpy( dst, src, sizeof(T)*nelements ) );

    #else
        throw blas::Error( "device BLAS not available", __func__ );
    #endif
}

//------------------------------------------------------------------------------
/// Copy nelements of type T from src to dst memory region.
/// src and dst regions must not overlap.
/// May be asynchronous with respect to host, depending on memory types;
/// host memory may need to be pinned for this to be async.
///
/// @param[out] dst
///     Pointer to destination memory region of size nelements.
///
/// @param[in] src
///     Pointer to source memory region of size nelements.
///
/// @param[in] nelements
///     Number of elements of type T to copy.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename T>
void device_memcpy(
    T*       dst,
    T const* src,
    int64_t nelements, Queue& queue)
{
    device_memcpy<T>(
        dst, src,
        nelements, MemcpyKind::Default, queue);
}

//------------------------------------------------------------------------------
/// @deprecated: recommend using
///     device_memcpy_2d( dst, dst_pitch, src, src_pitch, width, height, queue )
/// instead, which sets kind = MemcpyKind::Default.
///
/// @copydoc device_memcpy_2d(T*,int64_t,T const*,int64_t,int64_t,int64_t,Queue&)
///
/// @param[in] kind
/// @copydoc MemcpyKind
///
/// @see device_memcpy_2d(T*,int64_t,T const*,int64_t,int64_t,int64_t,Queue&)
///
template <typename T>
void device_memcpy_2d(
    T*       dst, int64_t dst_pitch,
    T const* src, int64_t src_pitch,
    int64_t width, int64_t height, MemcpyKind kind, Queue& queue)
{
    blas_error_if( width  < 0 );
    blas_error_if( height < 0 );
    blas_error_if( dst_pitch < width );
    blas_error_if( src_pitch < width );

    #ifdef BLAS_HAVE_CUBLAS
        blas::internal_set_device( queue.device() );
        blas_dev_call(
            cudaMemcpy2DAsync(
                dst, sizeof(T)*dst_pitch,
                src, sizeof(T)*src_pitch,
                sizeof(T)*width, height, memcpy2cuda(kind), queue.stream() ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
        blas::internal_set_device( queue.device() );
        blas_dev_call(
            hipMemcpy2DAsync(
                dst, sizeof(T)*dst_pitch,
                src, sizeof(T)*src_pitch,
                sizeof(T)*width, height, memcpy2hip(kind), queue.stream() ) );

    #elif defined(BLAS_HAVE_SYCL)
        if (dst_pitch == width && src_pitch == width) {
            // one contiguous memcpy
            blas_dev_call(
                queue.stream().memcpy( dst, src, width * height * sizeof(T) ) );
        }
        else {
            // Copy each contiguous image row (matrix column).
            // SYCL does not support set/get/lacpy matrix.
            for (int64_t i = 0; i < height; ++i) {
                T*       dst_row = dst + i*dst_pitch;
                T const* src_row = src + i*src_pitch;
                blas_dev_call(
                    queue.stream().memcpy( dst_row, src_row, width*sizeof(T) ) );
            }
        }
    #else
        throw blas::Error( "device BLAS not available", __func__ );
    #endif
}

//------------------------------------------------------------------------------
/// Copy width-by-height sub-array of type T from src to dst memory region.
/// Sub-arrays of src and dst must not overlap.
/// May be asynchronous with respect to host, depending on memory types;
/// host memory may need to be pinned for this to be async.
///
/// Memory here refers to 2D images, which by convention are
/// width-by-height (e.g., 1024 x 768), and stored with contiguous rows.
/// Each row has width elements, and may have padding, making a row
/// pitch (stride, leading dimension) = width + padding.
///
/// If declared as a C-style row-major matrix, the y-coordinate is first,
/// the x-coordinate second: A[ height ][ row_pitch ].
///
/// For a column-major matrix A, m == width, n == height, lda = pitch.
///
/// @param[out] dst
///     The destination width-by-height sub-array,
///     in a dst_pitch-by-height array.
///     Each row of width elements is contiguous.
///
/// @param[in] dst_pitch
///     Stride (leading dimension) between rows of the dst array.
///     dst_pitch >= width.
///
/// @param[in] src
///     The source width-by-height sub-array,
///     in a src_pitch-by-height array.
///     Each row of width elements is contiguous.
///
/// @param[in] src_pitch
///     Stride (leading dimension) between rows of the src array.
///     src_pitch >= width.
///
/// @param[in] width
///     Number of columns in each contiguous row to copy. width >= 0.
///
/// @param[in] height
///     Number of rows to copy. height >= 0.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename T>
void device_memcpy_2d(
    T*       dst, int64_t dst_pitch,
    T const* src, int64_t src_pitch,
    int64_t width, int64_t height, Queue& queue)
{
    device_memcpy_2d<T>(
        dst, dst_pitch,
        src, src_pitch,
        width, height, MemcpyKind::Default, queue);
}

//------------------------------------------------------------------------------
/// Copy n-element vector from host or device memory, to host or device memory.
/// May be asynchronous with respect to host, depending on memory types;
/// host memory may need to be pinned for this to be async.
///
/// @param[in] n
///     Number of elements to copy. n >= 0.
///
/// @param[in] src
///     The source n-element vector.
///
/// @param[in] inc_src
///     Stride between elements of src. inc_src >= 1.
///
/// @param[in] dst
///     The destination n-element vector.
///
/// @param[in] inc_dst
///     Stride between elements of src. inc_dst >= 1.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename T>
void device_copy_vector(
    int64_t n,
    T const* src, int64_t inc_src,
    T*       dst, int64_t inc_dst, Queue& queue)
{
    if (inc_src == 1 && inc_dst == 1) {
        // Copy contiguous vector.
        device_memcpy( dst, src, n, queue );
    }
    else {
        // Interpret as copying one row from inc-by-n matrix.
        device_memcpy_2d( dst, inc_dst, src, inc_src, 1, n, queue );
    }
}

//------------------------------------------------------------------------------
/// Copy m-by-n column-major matrix in ld-by-n array
/// from host or device memory, to host or device memory.
/// May be asynchronous with respect to host, depending on memory types;
/// host memory may need to be pinned for this to be async.
///
/// This is exactly the same as device_memcpy_2d, but with conventions
/// consistent with BLAS/LAPACK routines: matrices are column-major;
/// argument order is dimensions, src, dst.
///
/// @param[in] m
///     Number of rows. m >= 0.
///
/// @param[in] n
///     Number of columns. n >= 0.
///
/// @param[in] src
///     The source m-by-n matrix, in ld_src-by-n array.
///
/// @param[in] ld_src
///     Leading dimension of src. ld_src >= m.
///
/// @param[in] dst
///     The destination m-by-n matrix, in ld_dst-by-n array.
///
/// @param[in] ld_dst
///     Leading dimension of dst. ld_dst >= m.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename T>
void device_copy_matrix(
    int64_t m, int64_t n,
    T const* src, int64_t ld_src,
    T*       dst, int64_t ld_dst, Queue& queue)
{
    device_memcpy_2d( dst, ld_dst, src, ld_src, m, n, queue );
}

//------------------------------------------------------------------------------
/// @deprecated: recommend using
///     device_copy_vector( n, src, inc_src, dst, inc_dst, queue )
/// instead, which can copy from host or device, to host or device memory.
///
/// Copy n-element vector from host to device memory.
///
/// @see device_copy_vector
///
template <typename T>
[[deprecated("recommend device_copy_vector( n, any_src, inc_src, any_dst, inc_dst, queue )")]]
void device_setvector(
    int64_t n,
    T const* src_host, int64_t inc_src,
    T*       dst_dev,  int64_t inc_dst, Queue& queue)
{
    device_copy_vector( n, src_host, inc_src, dst_dev, inc_dst, queue );
}

//------------------------------------------------------------------------------
/// @deprecated: recommend using
///     device_copy_vector( n, src, inc_src, dst, inc_dst, queue )
/// instead, which can copy from host or device, to host or device memory.
///
/// Copy n-element vector from device to host memory.
///
/// @see device_copy_vector
///
template <typename T>
[[deprecated("recommend device_copy_vector( n, any_src, inc_src, any_dst, inc_dst, queue )")]]
void device_getvector(
    int64_t n,
    T const* src_dev,  int64_t inc_src,
    T*       dst_host, int64_t inc_dst, Queue& queue)
{
    device_copy_vector( n, src_dev, inc_src, dst_host, inc_dst, queue );
}

//------------------------------------------------------------------------------
/// @deprecated: recommend using
///     device_copy_matrix( m, n, src, ld_src, dst, ld_dst, queue )
/// instead, which can copy from host or device, to host or device memory.
///
/// Copy m-by-n matrix in ld-by-n array from host to device memory.
///
/// @see device_copy_matrix
///
template <typename T>
[[deprecated("recommend device_copy_matrix( m, n, any_src, ld_src, any_dst, ld_dst, queue )")]]
void device_setmatrix(
    int64_t m, int64_t n,
    T const* src_host, int64_t ld_src,
    T*       dst_dev,  int64_t ld_dst, Queue& queue)
{
    device_copy_matrix( m, n, src_host, ld_src, dst_dev, ld_dst, queue );
}

//------------------------------------------------------------------------------
/// @deprecated: recommend using
///     device_copy_matrix( m, n, src, ld_src, dst, ld_dst, queue )
/// instead, which can copy from host or device, to host or device memory.
///
/// Copy m-by-n matrix in ld-by-n array from device to host memory.
///
/// @see device_copy_matrix
///
template <typename T>
[[deprecated("recommend device_copy_matrix( m, n, any_src, ld_src, any_dst, ld_dst, queue )")]]
void device_getmatrix(
    int64_t m, int64_t n,
    T const* src_dev,  int64_t ld_src,
    T*       dst_host, int64_t ld_dst, Queue& queue)
{
    device_copy_matrix( m, n, src_dev, ld_src, dst_host, ld_dst, queue );
}

//------------------------------------------------------------------------------
/// Ensures GPU device workspace is of size at least lwork elements of
/// scalar_t, synchronizing and reallocating if needed.
/// Allocates at least 3 * MaxBatchChunk * sizeof(void*), needed for
/// batch gemm.
///
/// @param[in] lwork
///     Minimum size of workspace.
///
template <typename scalar_t>
void Queue::work_ensure_size( size_t lwork )
{
    lwork *= sizeof(scalar_t);
    if (lwork > lwork_) {
        sync();
        if (work_) {
            device_free( work_, *this );
        }
        lwork_ = max( lwork, 3*MaxBatchChunk*sizeof(void*) );
        work_ = device_malloc<char>( lwork_, *this );
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_DEVICE_HH
