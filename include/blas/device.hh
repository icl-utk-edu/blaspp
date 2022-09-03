// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
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
    // Default to HCC platform on ROCm
    #if ! defined(__HIP_PLATFORM_NVCC__) && ! defined(__HIP_PLATFORM_HCC__)
        #define __HIP_PLATFORM_HCC__
        #define BLAS_HIP_PLATFORM_HCC
    #endif

    #include <hip/hip_runtime.h>
    #include <rocblas.h>

    // If we defined __HIP_PLATFORM_HCC__, undef it.
    #ifdef BLAS_HIP_PLATFORM_HCC
        #undef __HIP_PLATFORM_HCC__
        #undef BLAS_HIP_PLATFORM_HCC
    #endif

#elif defined(BLAS_HAVE_ONEMKL)
    #include <CL/sycl/detail/cl.h>  // For CL version
    #include <CL/sycl.hpp>

#endif

namespace blas {

// -----------------------------------------------------------------------------
// types
typedef int Device;

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
#elif defined(BLAS_HAVE_ONEMKL)
    /// @return the corresponding sycl memcpy kind constant
    /// The memcpy method in the sycl::queue class does not accept
    /// a direction (i.e. always operates in default mode).
    /// For interface compatibility with cuda/hip, return a default value
    inline int64_t memcpy2sycl( MemcpyKind kind ) { return 0; }
#endif

// -----------------------------------------------------------------------------
// constants
const int DEV_QUEUE_DEFAULT_BATCH_LIMIT = 50000;
const int DEV_QUEUE_FORK_SIZE           = 10;

//==============================================================================
/// Queue for executing GPU device routines.
/// This wraps CUDA stream and cuBLAS handle,
/// HIP stream and rocBLAS handle,
/// or SYCL queue for oneMKL.
///
class Queue
{
public:
    Queue();
    Queue( blas::Device device, int64_t batch_size );
    // Disable copying; must construct anew.
    Queue( Queue const& ) = delete;
    Queue& operator=( Queue const& ) = delete;
    ~Queue();

    blas::Device           device() const { return device_; }
    void                   sync();
    size_t                 get_batch_limit() { return batch_limit_; }
    void**                 get_dev_ptr_array();

    /// @return device workspace.
    void* work() { return (void*) work_; }

    /// @return size of device workspace, in scalar_t elements.
    template <typename scalar_t>
    size_t work_size() const { return lwork_ / sizeof(scalar_t); }

    template <typename scalar_t>
    void work_resize( size_t lwork );

    // switch from default stream to parallel streams
    void fork();

    // switch back to the default stream
    void join();

    // return the next-in-line stream (for both default and fork modes)
    void revolve();

    #ifdef BLAS_HAVE_CUBLAS
        cudaStream_t   stream()        const { return *current_stream_; }
        cublasHandle_t handle()        const { return handle_; }
    #elif defined(BLAS_HAVE_ROCBLAS)
        hipStream_t    stream()        const { return *current_stream_; }
        rocblas_handle handle()        const { return handle_; }
    #elif defined(BLAS_HAVE_ONEMKL)
        cl::sycl::device sycl_device() const { return sycl_device_; }
        cl::sycl::queue  stream()      const { return *default_stream_; }
    #endif

private:
    // associated device ID
    blas::Device device_;

    // max workspace allocated for a batch argument in a single call
    // (e.g. a pointer array)
    size_t batch_limit_;

    // Workspace for pointer arrays of batch routines or other purposes.
    char* work_;
    size_t lwork_;

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

    #elif defined(BLAS_HAVE_ONEMKL)
        // in addition to the integer device_ member, we need
        // the sycl device id
        cl::sycl::device  sycl_device_;

        // default sycl queue for this blas queue
        cl::sycl::queue *default_stream_;
        cl::sycl::event  default_event_;

        // pointer to current stream (default or fork mode)
        cl::sycl::queue *current_stream_;

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
        error

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
        catch (...) { \
            blas::internal::abort_if( true, __func__, \
                                      "%s", "unknown exception" ); \
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
    #define blas_dev_call( error ) \
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
        catch (...) { \
            blas::internal::throw_if( true, \
                                      "unknown exception", __func__ ); \
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
#ifdef BLAS_HAVE_ONEMKL
void enumerate_devices(std::vector<cl::sycl::device> &devices);
#endif

// -----------------------------------------------------------------------------
// memory functions
void device_free( void* ptr );
void device_free( void* ptr, blas::Queue &queue );

void device_free_pinned( void* ptr );
void device_free_pinned( void* ptr, blas::Queue &queue );

// -----------------------------------------------------------------------------
// Template functions declared here
// -----------------------------------------------------------------------------

//------------------------------------------------------------------------------
/// @return a device pointer to an allocated memory space
/// In CUDA and ROCm, this version uses current device;
/// in SYCL, this doesn't work since there is no concept of a current device.
///
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
        // SYCL requires a device or queue to malloc
        throw blas::Error( "unsupported function for sycl backend", __func__ );

    #else

        throw blas::Error( "device BLAS not available", __func__ );
    #endif
    return ptr;
}

//------------------------------------------------------------------------------
/// @return a device pointer to an allocated memory space on specific device.
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
    T* ptr = nullptr;
    #ifdef BLAS_HAVE_CUBLAS
        blas::set_device( queue.device() );
        blas_dev_call(
                cudaMalloc( (void**)&ptr, nelements * sizeof(T) ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
        blas::set_device( queue.device() );
        blas_dev_call(
                hipMalloc( (void**)&ptr, nelements * sizeof(T) ) );

    #elif defined(BLAS_HAVE_ONEMKL)
        blas_dev_call(
            ptr = (T*)cl::sycl::malloc_shared( nelements*sizeof(T), queue.stream() ) );

    #else
        throw blas::Error( "device BLAS not available", __func__ );
    #endif
    return ptr;
}

//------------------------------------------------------------------------------
/// @return a host pointer to a pinned memory space.
/// In CUDA and ROCm, this version uses current device;
/// in SYCL, this doesn't work since there is no concept of a current device.
///
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
        // SYCL requires a device or queue to malloc
        throw blas::Error( "unsupported function for sycl backend", __func__ );

    #else
        throw blas::Error( "device BLAS not available", __func__ );
    #endif
    return ptr;
}

//------------------------------------------------------------------------------
/// @return a host pointer to a pinned memory space using a specific device queue.
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
T* device_malloc_pinned(
    int64_t nelements, blas::Queue &queue )
{
    T* ptr = nullptr;
    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cudaMallocHost( (void**)&ptr, nelements * sizeof(T) ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
        blas_dev_call(
            hipHostMalloc( (void**)&ptr, nelements * sizeof(T) ) );

    #elif defined(BLAS_HAVE_ONEMKL)
        blas_dev_call(
            ptr = (T*)cl::sycl::malloc_host( nelements*sizeof(T), queue.stream() ) );

    #else
        throw blas::Error( "device BLAS not available", __func__ );
    #endif
    return ptr;
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
    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cudaMemcpyAsync(
                dst, src, sizeof(T)*nelements,
                memcpy2cuda(kind), queue.stream() ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
        blas_dev_call(
            hipMemcpyAsync(
                dst, src, sizeof(T)*nelements,
                memcpy2hip(kind), queue.stream() ) );

    #elif defined(BLAS_HAVE_ONEMKL)
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
/// host memory may need to be pinned for this to by async.
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
    blas_error_if( dst_pitch < width );
    blas_error_if( src_pitch < width );

    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cudaMemcpy2DAsync(
                dst, sizeof(T)*dst_pitch,
                src, sizeof(T)*src_pitch,
                sizeof(T)*width, height, memcpy2cuda(kind), queue.stream() ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
         blas_dev_call(
            hipMemcpy2DAsync(
                dst, sizeof(T)*dst_pitch,
                src, sizeof(T)*src_pitch,
                sizeof(T)*width, height, memcpy2hip(kind), queue.stream() ) );

    #elif defined(BLAS_HAVE_ONEMKL)
        if (dst_pitch == width && src_pitch == width) {
            // one contiguous memcpy
            blas_dev_call(
                queue.stream().memcpy( dst, src, width * height * sizeof(T) ) );
        }
        else {
            // Copy each contiguous image row (matrix column).
            // SYCL does not support set/get/lacpy matrix.
            for (int64_t i = 0; i < height; ++i) {
                const T* dst_row = dst + i*dst_pitch;
                T*       src_row = src + i*src_pitch;
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
/// host memory may need to be pinned for this to by async.
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
/// host memory may need to be pinned for this to by async.
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
/// host memory may need to be pinned for this to by async.
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
///
/// @param[in] lwork
///     Minimum size of workspace.
///
template <typename scalar_t>
void Queue::work_resize( size_t lwork )
{
    lwork *= sizeof(scalar_t);
    if (lwork > lwork_) {
        sync();
        if (work_) {
            device_free( work_, *this );
        }
        lwork_ = lwork;
        work_ = device_malloc<char>( lwork, *this );
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_DEVICE_HH
