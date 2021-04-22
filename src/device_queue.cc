// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device.hh"

namespace blas {

// =============================================================================
// Light wrappers around CUDA and cuBLAS functions.
#if defined(BLAS_HAVE_CUBLAS)

// -----------------------------------------------------------------------------
void stream_create( cudaStream_t* stream )
{
    blas_dev_call( cudaStreamCreate( stream ) );
}

// -----------------------------------------------------------------------------
void stream_destroy( cudaStream_t stream )
{
    blas_dev_call( cudaStreamDestroy( stream ) );
}

// -----------------------------------------------------------------------------
void stream_synchronize( cudaStream_t stream )
{
    blas_dev_call( cudaStreamSynchronize( stream ) );
}

// -----------------------------------------------------------------------------
void handle_create( cublasHandle_t* handle )
{
    blas_dev_call( cublasCreate( handle ) );
}

// -----------------------------------------------------------------------------
void handle_destroy( cublasHandle_t handle )
{
    blas_dev_call( cublasDestroy( handle ) );
}

// -----------------------------------------------------------------------------
void handle_set_stream( cublasHandle_t handle, cudaStream_t stream )
{
    blas_dev_call( cublasSetStream( handle, stream ) );
}

// -----------------------------------------------------------------------------
void event_create( cudaEvent_t* event )
{
    blas_dev_call( cudaEventCreate( event ) );
}

// -----------------------------------------------------------------------------
void event_destroy( cudaEvent_t event )
{
    blas_dev_call( cudaEventDestroy( event ) );
}

// -----------------------------------------------------------------------------
void event_record( cudaEvent_t event, cudaStream_t stream )
{
    blas_dev_call( cudaEventRecord( event, stream ) );
}

// -----------------------------------------------------------------------------
void stream_wait_event( cudaStream_t stream, cudaEvent_t event, unsigned int flags )
{
    blas_dev_call( cudaStreamWaitEvent( stream, event, flags ) );
}

// =============================================================================
// Light wrappers around HIP and rocBLAS functions.
#elif defined(BLAS_HAVE_ROCBLAS)

// -----------------------------------------------------------------------------
void stream_create( hipStream_t* stream )
{
    blas_dev_call( hipStreamCreate( stream ) );
}

// -----------------------------------------------------------------------------
void stream_destroy( hipStream_t stream )
{
    blas_dev_call( hipStreamDestroy( stream ) );
}

// -----------------------------------------------------------------------------
void stream_synchronize( hipStream_t stream )
{
    blas_dev_call( hipStreamSynchronize( stream ) );
}

// -----------------------------------------------------------------------------
void handle_create( rocblas_handle* handle )
{
    blas_dev_call( rocblas_create_handle( handle ) );
}

// -----------------------------------------------------------------------------
void handle_destroy( rocblas_handle handle )
{
    blas_dev_call( rocblas_destroy_handle( handle ) );
}

// -----------------------------------------------------------------------------
void handle_set_stream( rocblas_handle handle, hipStream_t stream )
{
    blas_dev_call( rocblas_set_stream( handle, stream ) );
}

// -----------------------------------------------------------------------------
void event_create( hipEvent_t* event )
{
    blas_dev_call( hipEventCreate( event ) );
}

// -----------------------------------------------------------------------------
void event_destroy( hipEvent_t event )
{
    blas_dev_call( hipEventDestroy( event ) );
}

// -----------------------------------------------------------------------------
void event_record( hipEvent_t event, hipStream_t stream )
{
    blas_dev_call( hipEventRecord( event, stream ) );
}

// -----------------------------------------------------------------------------
void stream_wait_event( hipStream_t stream, hipEvent_t event, unsigned int flags )
{
    blas_dev_call( hipStreamWaitEvent( stream, event, flags ) );
}

#endif

// =============================================================================

/** queue member functions **/

// -----------------------------------------------------------------------------
/// Default constructor.
Queue::Queue()
{
    // get the currently set device ID
    get_device( &device_ );
    batch_limit_ = DEV_QUEUE_DEFAULT_BATCH_LIMIT;
    // compute workspace for pointer arrays in the queue
    // fork size + 1 (def. stream), each need 3 pointer arrays
    size_t workspace_size = 3 * batch_limit_ * ( DEV_QUEUE_FORK_SIZE + 1 );
    dev_ptr_array_ = device_malloc<void*>( workspace_size );

    #if defined(BLAS_HAVE_CUBLAS) || defined(BLAS_HAVE_ROCBLAS)
        // default stream
        stream_create( &default_stream_ );
        handle_create( &handle_ );
        handle_set_stream( handle_, default_stream_ );
        current_stream_       = &default_stream_;
        num_active_streams_   = 1;
        current_stream_index_ = 0;

        // create parallel streams
        for (size_t i = 0; i < DEV_QUEUE_FORK_SIZE; ++i) {
            stream_create( &parallel_streams_[ i ] );
        }

        // create default and parallel events
        event_create( &default_event_ );
        for (size_t i = 0; i < DEV_QUEUE_FORK_SIZE; ++i) {
            event_create(&parallel_events_[ i ]);
        }
    #endif
}

// -----------------------------------------------------------------------------
/// Constructor with device and batch init.
// todo: merge with default constructor.
Queue::Queue( blas::Device device, int64_t batch_size )
{
    #if defined(BLAS_HAVE_CUBLAS) || defined(BLAS_HAVE_ROCBLAS)
        device_ = device;
        batch_limit_ = batch_size;
        set_device( device_ );
        // compute workspace for pointer arrays in the queue
        // fork size + 1 (def. stream), each need 3 pointer arrays
        size_t workspace_size = 3 * batch_limit_ * ( DEV_QUEUE_FORK_SIZE + 1 );
        dev_ptr_array_ = device_malloc<void*>( workspace_size );

        stream_create( &default_stream_ );
        handle_create( &handle_ );
        handle_set_stream( handle_, default_stream_ );
        current_stream_       = &default_stream_;
        num_active_streams_   = 1;
        current_stream_index_ = 0;

        // create parallel streams
        for (size_t i = 0; i < DEV_QUEUE_FORK_SIZE; ++i) {
            stream_create(&parallel_streams_[ i ]);
        }

        // create default and parallel events
        event_create( &default_event_ );
        for (size_t i = 0; i < DEV_QUEUE_FORK_SIZE; ++i) {
            event_create( &parallel_events_[ i ] );
        }
    #endif
}

// -----------------------------------------------------------------------------
// Default destructor.
Queue::~Queue()
{
    try {
        #if defined(BLAS_HAVE_CUBLAS) || defined(BLAS_HAVE_ROCBLAS)
            device_free( dev_ptr_array_ );
            handle_destroy( handle_ );
            stream_destroy( default_stream_ );

            // destroy parallel streams
            for (size_t i = 0; i < DEV_QUEUE_FORK_SIZE; ++i) {
                stream_destroy( parallel_streams_[ i ] );
            }

            // destroy events
            event_destroy( default_event_ );
            for (size_t i = 0; i < DEV_QUEUE_FORK_SIZE; ++i) {
                event_destroy( parallel_events_[ i ] );
            }
        #endif
    }
    catch (...) {
        // Destructors can't leak exceptions.
        // todo: best way to handle?
    }
}

// -----------------------------------------------------------------------------
/// Synchronize with queue.
void Queue::sync()
{
    #if defined(BLAS_HAVE_CUBLAS) || defined(BLAS_HAVE_ROCBLAS)
        // in default mode, sync with default stream
        // otherwise, sync against the parallel streams
        if (current_stream_ == &default_stream_) {
            stream_synchronize( default_stream_ );
        }
        else {
            for (size_t i = 0; i < DEV_QUEUE_FORK_SIZE; ++i) {
                stream_synchronize( parallel_streams_[ i ] );
            }
        }
    #endif
}

// -----------------------------------------------------------------------------
/// Get device array pointer for the current stream.
void**  Queue::get_dev_ptr_array()
{
    // in default (join) mode, return dev_ptr_array with no offset
    if (current_stream_ == &default_stream_)
        return dev_ptr_array_;

    // in fork mode, return dev_ptr_array_ + offset
    size_t offset = (current_stream_index_ + 1) * 3 * batch_limit_;
    return (dev_ptr_array_ + offset);
}

// -----------------------------------------------------------------------------
/// Forks the kernel launches assigned to this queue to parallel streams.
/// This function is not nested (you must join after each fork).
void Queue::fork()
{
    #if defined(BLAS_HAVE_CUBLAS) || defined(BLAS_HAVE_ROCBLAS)
        // check if queue is already in fork mode
        if (current_stream_ != &default_stream_)
            return;

        // make sure dependencies are respected
        event_record(default_event_, default_stream_);
        for (size_t i = 0; i < DEV_QUEUE_FORK_SIZE; ++i) {
            stream_wait_event( parallel_streams_[i], default_event_, 0 );
        }

        // assign current stream
        current_stream_index_ = 0;
        num_active_streams_   = DEV_QUEUE_FORK_SIZE;
        current_stream_       = &parallel_streams_[ current_stream_index_ ];

        // assign cublas handle to current stream
        handle_set_stream( handle_, *current_stream_ );
    #endif
}

// -----------------------------------------------------------------------------
/// Switch executions on this queue back from parallel streams to the default
/// stream. This function is not nested (you must join after each fork).
void Queue::join()
{
    #if defined(BLAS_HAVE_CUBLAS) || defined(BLAS_HAVE_ROCBLAS)
        // check if queue is already joined
        if (current_stream_ == &default_stream_)
            return;

        // make sure dependencies are respected
        for (size_t i = 0; i < DEV_QUEUE_FORK_SIZE; ++i) {
            event_record( parallel_events_[i], parallel_streams_[i] );
            stream_wait_event( default_stream_, parallel_events_[i], 0 );
        }

        // assign current stream
        current_stream_index_ = 0;
        num_active_streams_   = 1;
        current_stream_       = &default_stream_;

        // assign current stream to blas handle
        handle_set_stream( handle_, *current_stream_ );
    #endif
}

// -----------------------------------------------------------------------------
/// In fork mode, switch execution to the next-in-line stream.
/// In join mode, no effect.
void Queue::revolve()
{
    #if defined(BLAS_HAVE_CUBLAS) || defined(BLAS_HAVE_ROCBLAS)
        // return if not in fork mode
        if (current_stream_ == &default_stream_)
            return;

        // choose the next-in-line stream
        current_stream_index_ = (current_stream_index_ + 1) % num_active_streams_;
        current_stream_       = &parallel_streams_[ current_stream_index_ ];

        // assign current stream to blas handle
        handle_set_stream( handle_, *current_stream_ );
    #endif
}

}  // namespace blas
