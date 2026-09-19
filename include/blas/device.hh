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
    // Default to AMD platform on ROCm
    #if ! defined(__HIP_PLATFORM_NVCC__) && ! defined(__HIP_PLATFORM_AMD__)
        #define __HIP_PLATFORM_AMD__
        #define BLAS_HIP_PLATFORM_AMD
    #endif

    #include <hip/hip_runtime.h>
    #include <hip/hip_complex.h>

    // Headers moved in ROCm 5.2
    #if HIP_VERSION >= 50200000
        #include <rocblas/rocblas.h>
    #else
        #include <rocblas.h>
    #endif

    // If we defined __HIP_PLATFORM_AMD__, undef it.
    #ifdef BLAS_HIP_PLATFORM_AMD
        #undef __HIP_PLATFORM_AMD__
        #undef BLAS_HIP_PLATFORM_AMD
    #endif

#elif defined(BLAS_HAVE_SYCL)
    #include <sycl/detail/cl.h>  // For CL version
    #include <sycl/sycl.hpp>

#endif

namespace blas {

// -----------------------------------------------------------------------------
// types

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
// constants
const int MaxBatchChunk = 50000;

#if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )
    const int MaxForkSize = 10;
#else
    // SYCL and no GPU code doesn't support fork mode.
    const int MaxForkSize = 1;
#endif

//==============================================================================
/// @brief Queue for executing GPU device routines.
///
/// This class provides a unified interface for GPU operations across different
/// backends:
/// - CUDA: Wraps cudaStream_t and cublasHandle_t
/// - HIP/ROCm: Wraps hipStream_t and rocblas_handle
/// - SYCL: Wraps sycl::queue
///
/// The Queue manages device memory workspace, stream/handle lifecycle,
/// and supports fork/join parallelism for batch operations.
///
/// @ingroup device
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

    /// @brief Default constructor. Uses device 0.
    Queue();
    
    /// @brief Construct queue for specified device.
    /// @param[in] device Device ID to use
    Queue( int device );

    /// @brief Construct queue with specified device and stream.
    /// @param[in] device Device ID to use
    /// @param[in] stream Pre-existing stream to use
    Queue( int device, stream_t& stream );

    #if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )
        /// @brief Construct queue with specified device and BLAS handle.
        /// @param[in] device Device ID to use
        /// @param[in] handle Pre-existing cuBLAS or rocBLAS handle
        Queue( int device, handle_t handle );
    #endif

    // Disable copying; must construct anew.
    Queue( Queue const& ) = delete;
    Queue& operator=( Queue const& ) = delete;

    /// @brief Destructor. Synchronizes and frees resources.
    ~Queue();

    /// @brief Get device ID associated with this queue.
    /// @return Device ID
    int  device() const { return device_; }
    
    /// @brief Synchronize all operations on this queue.
    /// Blocks until all queued operations complete.
    void sync();

    /// @brief Get pointer to device workspace.
    /// @return Pointer to device workspace memory
    void* work() { return (void*) work_; }

    /// @brief Get size of device workspace.
    /// @return Size of workspace in number of scalar_t elements
    /// @tparam scalar_t Element type for size calculation
    template <typename scalar_t>
    size_t work_size() const { return lwork_ / sizeof(scalar_t); }

    /// @brief Ensure device workspace is at least specified size.
    /// Reallocates if necessary, synchronizing first.
    /// @param[in] lwork Minimum workspace size in scalar_t elements
    /// @tparam scalar_t Element type for size calculation
    template <typename scalar_t>
    void work_ensure_size( size_t lwork );

    /// @brief Switch from default stream to parallel streams.
    /// Enables concurrent kernel execution across multiple streams.
    /// @param[in] num_streams Number of parallel streams (default: MaxForkSize)
    void fork( int num_streams=MaxForkSize );

    /// @brief Switch back to the default stream.
    /// Synchronizes all parallel streams and returns to single-stream mode.
    void join();

    /// @brief Rotate to the next stream in the queue.
    /// Used with fork() to distribute work across parallel streams.
    void revolve();

    #if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )
        /// @brief Set the BLAS handle for this queue.
        /// @param[in] in_handle cuBLAS or rocBLAS handle to use
        void set_handle( handle_t& in_handle );
        
        /// @brief Get the BLAS handle for this queue.
        /// @return Current BLAS handle
        handle_t handle() const { return handle_; }
    #endif

    /// @brief Set the stream for this queue.
    /// @param[in] in_stream Stream to use
    void set_stream( stream_t& in_stream );

    /// @brief Get the current stream.
    /// @return Reference to current stream (may be parallel stream in fork mode)
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

/// @brief Internal function to set current device (CUDA/ROCm only).
/// @param[in] device Device ID to set as current
/// @ingroup device
void internal_set_device( int device );

/// @brief Get number of available GPU devices.
/// @return Number of devices, or 0 if no GPU support
/// @ingroup device
int get_device_count();

// -----------------------------------------------------------------------------
// memory functions

/// @brief Free device memory.
/// @param[in] ptr Pointer to device memory to free
/// @param[in] queue Queue that allocated the memory
/// @ingroup device
void device_free( void* ptr, blas::Queue &queue );

/// @brief Free pinned host memory.
/// @param[in] ptr Pointer to pinned host memory to free
/// @param[in] queue Queue that allocated the memory
/// @ingroup device
void host_free_pinned( void* ptr, blas::Queue &queue );

/// @brief Check if pointer is to device memory.
/// @param[in] A Pointer to check
/// @param[in] queue Queue for device context
/// @return True if A points to device memory, false otherwise
/// @ingroup device
bool is_devptr( const void* A, blas::Queue &queue );

// -----------------------------------------------------------------------------
// Template functions declared here
// -----------------------------------------------------------------------------

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
    blas_error_if( nelements < 0 );

    #ifdef BLAS_HAVE_CUBLAS
        blas::internal_set_device( queue.device() );
        blas_dev_call(
            cudaMemcpyAsync(
                dst, src, sizeof(T)*nelements,
                cudaMemcpyDefault, queue.stream() ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
        blas::internal_set_device( queue.device() );
        blas_dev_call(
            hipMemcpyAsync(
                dst, src, sizeof(T)*nelements,
                hipMemcpyDefault, queue.stream() ) );

    #elif defined(BLAS_HAVE_SYCL)
        blas_dev_call(
            queue.stream().memcpy( dst, src, sizeof(T)*nelements ) );

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
                sizeof(T)*width, height,
                cudaMemcpyDefault, queue.stream() ) );

    #elif defined(BLAS_HAVE_ROCBLAS)
        blas::internal_set_device( queue.device() );
        blas_dev_call(
            hipMemcpy2DAsync(
                dst, sizeof(T)*dst_pitch,
                src, sizeof(T)*src_pitch,
                sizeof(T)*width, height,
                hipMemcpyDefault, queue.stream() ) );

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

//------------------------------------------------------------------------------
template <typename scalar_t>
void shift_vec( int64_t n, scalar_t* v, scalar_t c, blas::Queue& queue );

//------------------------------------------------------------------------------
template <typename scalar_t>
void conj(
    int64_t n,
    scalar_t const* x, int64_t incx,
    scalar_t*       y, int64_t incy,
    blas::Queue& queue );

#ifndef BLAS_HAVE_DEVICE

//----------------------------------------
template <typename scalar_t>
void conj(
    int64_t n,
    scalar_t const* x, int64_t incx,
    scalar_t*       y, int64_t incy,
    blas::Queue& queue )
{
    throw blas::Error( "device BLAS not available", __func__ );
}

#elif defined(BLAS_HAVE_SYCL)

//----------------------------------------
template <typename scalar_t>
void conj(
    int64_t n,
    scalar_t const* x, int64_t incx,
    scalar_t*       y, int64_t incy,
    blas::Queue& queue )
{
    blas_error_if( n < 0 );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    if (n == 0)
        return;

    int64_t ix = (incx > 0 ? 0 : (1 - n) * incx);
    int64_t iy = (incy > 0 ? 0 : (1 - n) * incy);

    queue.stream().submit( [&]( sycl::handler& h ) {
        h.parallel_for( sycl::range<1>(n), [=]( sycl::id<1> i ) {
            y[ i*incy + iy ] = conj( x[ i*incx + ix ] );
        } );
    } );
}

#endif // BLAS_HAVE_SYCL

//------------------------------------------------------------------------------
template <typename scalar_t>
void geadd(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t const* A, int64_t lda,
    scalar_t beta,  scalar_t*       B, int64_t ldb,
    blas::Queue& queue );

#ifndef BLAS_HAVE_DEVICE

//----------------------------------------
template <typename scalar_t>
void geadd(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t const* A, int64_t lda,
    scalar_t beta,  scalar_t*       B, int64_t ldb,
    blas::Queue& queue )
{
    throw blas::Error( "device BLAS not available", __func__ );
}

#elif defined(BLAS_HAVE_SYCL)

//----------------------------------------
template <typename scalar_t>
void geadd(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t const* A, int64_t lda,
    scalar_t beta,  scalar_t*       B, int64_t ldb,
    blas::Queue& queue )
{
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (layout == Layout::ColMajor) {
        if (trans == Op::NoTrans)
            blas_error_if( lda < m );
        else
            blas_error_if( lda < n );

        blas_error_if( ldb < m );
    }
    else {
        if (trans == Op::NoTrans)
            blas_error_if( lda < n );
        else
            blas_error_if( lda < m );

        blas_error_if( ldb < n );
    }

    if (m == 0 || n == 0)
        return;

    if (layout == Layout::RowMajor) {
        std::swap(m, n);
    }

    const int64_t Block_x = 32;
    const int64_t Block_y = 32;

    sycl::range<2> threads( min( Block_x, m ), min( Block_y, n ) );
    sycl::range<2> blocks( ceildiv( m, Block_x ), ceildiv( n, Block_y ) );

    queue.stream().submit( [&]( sycl::handler& h ) {
        h.parallel_for( sycl::nd_range<2>( blocks * threads, threads ), [=]( sycl::nd_item<2> item ) {
            int64_t i = item.get_global_id(0);
            int64_t j = item.get_global_id(1);

            if (i < m && j < n) {
                scalar_t A_op = 0;
                switch (trans) {
                    case Op::NoTrans:   A_op = A[ i + j*lda ]; break;
                    case Op::Trans:     A_op = A[ j + i*lda ]; break;
                    case Op::ConjTrans: A_op = conj( A[ j + i*lda ] ); break;
                }
                B[ i + j*ldb ] = beta * B[ i + j*ldb ] + alpha * A_op;
            }
        } );
    } );
}

#endif // BLAS_HAVE_SYCL

//------------------------------------------------------------------------------
template <typename scalar_t>
void tzadd(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t const* A, int64_t lda,
    scalar_t beta,  scalar_t*       B, int64_t ldb,
    blas::Queue& queue );

#ifndef BLAS_HAVE_DEVICE

//----------------------------------------
template <typename scalar_t>
void tzadd(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t const* A, int64_t lda,
    scalar_t beta,  scalar_t*       B, int64_t ldb,
    blas::Queue& queue )
{
    throw blas::Error( "device BLAS not available", __func__ );
}

#elif defined(BLAS_HAVE_SYCL)

//----------------------------------------
template <typename scalar_t>
void tzadd(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t m, int64_t n,
    scalar_t alpha, scalar_t const* A, int64_t lda,
    scalar_t beta,  scalar_t*       B, int64_t ldb,
    blas::Queue& queue )
{
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (layout == Layout::ColMajor) {
        if (trans == Op::NoTrans)
            blas_error_if( lda < m );
        else
            blas_error_if( lda < n );

        blas_error_if( ldb < m );
    }
    else {
        if (trans == Op::NoTrans)
            blas_error_if( lda < n );
        else
            blas_error_if( lda < m );

        blas_error_if( ldb < n );
    }

    if (m == 0 || n == 0)
        return;

    if (layout == Layout::RowMajor) {
        std::swap(m, n);
    }

    const int64_t Block_x = 32;
    const int64_t Block_y = 32;

    sycl::range<2> threads( min( Block_x, m ), min( Block_y, n ) );
    sycl::range<2> blocks( ceildiv( m, Block_x ), ceildiv( n, Block_y ) );

    queue.stream().submit( [&]( sycl::handler& h ) {
        h.parallel_for( sycl::nd_range<2>( blocks * threads, threads ), [=]( sycl::nd_item<2> item ) {
            int64_t i = item.get_global_id(0);
            int64_t j = item.get_global_id(1);

            int64_t j_lower = (uplo == Uplo::Lower ? 0 : max( 0, i - max( 0, m - n ) ));
            int64_t j_upper = (uplo == Uplo::Upper ? n : min( n, i + 1 ));

            if (i < m && j >= j_lower && j < j_upper) {
                scalar_t A_op = 0;
                switch (trans) {
                    case Op::NoTrans:   A_op = A[ i + j*lda ]; break;
                    case Op::Trans:     A_op = A[ j + i*lda ]; break;
                    case Op::ConjTrans: A_op = conj( A[ j + i*lda ] ); break;
                }
                B[ i + j*ldb ] = beta * B[ i + j*ldb ] + alpha * A_op;
            }
        } );
    } );
}

#endif // BLAS_HAVE_SYCL

}  // namespace blas

#endif        //  #ifndef BLAS_DEVICE_HH
