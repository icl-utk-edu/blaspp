// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device.hh"
#include "device_internal.hh"

namespace blas {

// =============================================================================
// Queue member functions

// -----------------------------------------------------------------------------
/// Default constructor.
/// For CUDA and ROCm, creates a Queue on the current device.
/// For SYCL, throws an error.
/// todo: SYCL has a default device, how to use it?
Queue::Queue()
  : work_( nullptr ),
    lwork_( 0 )
    #if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )
        ,
        streams_ { stream_create() },  // remaining streams are null
        handle_( handle_create( streams_[ 0 ] ) ),
        events_  { nullptr },  // all events are null
        num_active_streams_( 1 ),
        current_stream_index_( 0 ),
        own_handle_        ( true ),
        own_default_stream_( true )
        // todo device_( get_device() )
    #endif
{
    #if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )
        // get the currently set device ID
        #if defined( BLAS_HAVE_CUBLAS )
            blas_dev_call( cudaGetDevice( &device_ ) );
        #elif defined( BLAS_HAVE_ROCBLAS )
            blas_dev_call( hipGetDevice( &device_ ) );
        #endif

    #elif defined( BLAS_HAVE_SYCL )
        throw blas::Error( "SYCL requires a device; use Queue( device )",
                           __func__ );
    #endif
}

//------------------------------------------------------------------------------
// SYCL needs this property in Queue constructor.
#if defined( BLAS_HAVE_SYCL )

    static const sycl::property_list s_property_in_order
        { sycl::property::queue::in_order() };

#endif

// -----------------------------------------------------------------------------
/// Constructor with device.
Queue::Queue( int device )
  : work_( nullptr ),
    lwork_( 0 ),
    #if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )
        streams_ { stream_create( device ) },  // remaining streams are null
        handle_( handle_create( streams_[ 0 ] ) ),
        events_  { nullptr },  // all events are null
        num_active_streams_( 1 ),
        current_stream_index_( 0 ),
        own_handle_        ( true ),
        own_default_stream_( true ),

    #elif defined( BLAS_HAVE_SYCL )
        streams_{ sycl::queue( DeviceList::at( device ), s_property_in_order ) },

    #else
        // No GPU
        streams_{ nullptr },
    #endif
    device_( device )
{}

#if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )
    // -------------------------------------------------------------------------
    /// Constructor taking a cuBLAS or rocBLAS handle.
    /// This gets the stream from the given handle.
    /// The user retains ownership of the stream and handle,
    /// which must exist whenever this queue is used.
    Queue::Queue( int device, handle_t handle )
      : work_( nullptr ),
        lwork_( 0 ),
        streams_ { handle_get_stream( handle ) },  // remaining streams are null
        handle_( handle ),
        events_  { nullptr },  // all events  are null
        num_active_streams_( 1 ),
        current_stream_index_( 0 ),
        own_handle_        ( false ),
        own_default_stream_( false ),
        device_( device )
    {}

    // -------------------------------------------------------------------------
    /// Constructor taking a CUDA or HIP stream.
    /// This allocates a cublasHandle_t or rocblas_handle,
    /// and associates the given stream with it.
    /// The user retains ownership of the stream,
    /// which must exist whenever this queue is used.
    Queue::Queue( int device, stream_t& stream )
      : work_( nullptr ),
        lwork_( 0 ),
        streams_ { stream },   // remaining streams are null
        handle_( nullptr ),    // created below due to set_device
        events_  { nullptr },  // all events  are null
        num_active_streams_( 1 ),
        current_stream_index_( 0 ),
        own_handle_        ( true  ),
        own_default_stream_( false ),
        device_( device )
    {
        internal_set_device( device_ );
        handle_ = handle_create( stream );
    }

    // -------------------------------------------------------------------------
    /// Change the CUDA or HIP stream used in the BLAS++ queue.
    /// Kernels executing on the current stream will continue.
    /// Throws an error if in fork mode.
    void Queue::set_stream( stream_t& stream )
    {
        if (num_active_streams_ > 1)
            throw blas::Error( "can't set stream in fork mode", __func__ );

        if (own_default_stream_) {
            stream_destroy( streams_[ 0 ] );
            own_default_stream_ = false;
        }
        streams_[ 0 ] = stream;
        handle_set_stream( handle_, streams_[ 0 ] );
    }

    // -------------------------------------------------------------------------
    /// Change the cuBLAS or rocBLAS handle used in the BLAS++ queue.
    /// Gets the stream from the handle.
    /// Throws an error if in fork mode.
    void Queue::set_handle( handle_t& handle )
    {
        if (num_active_streams_ > 1)
            throw blas::Error( "can't set stream in fork mode", __func__ );

        if (own_default_stream_) {
            stream_destroy( streams_[ 0 ] );
            own_default_stream_ = false;
        }
        if (own_handle_) {
            handle_destroy( handle_ );
            own_handle_ = false;
        }
        handle_ = handle;
        streams_[ 0 ] = handle_get_stream( handle_ );
    }

#elif defined( BLAS_HAVE_SYCL )
    // -------------------------------------------------------------------------
    /// Constructor taking a SYCL queue (stream).
    /// Unlike in CUDA/ROCm, this copies the SYCL queue.
    /// Note BLAS++ generally assumes streams are in-order, which SYCL
    /// queues are not by default. See sycl::property::queue::in_order.
    // todo: check that stream is in-order?
    Queue::Queue( int device, stream_t& stream )
      : work_( nullptr ),
        lwork_( 0 ),
        streams_{ stream },
        device_( device )
    {}

    // -------------------------------------------------------------------------
    /// Change the SYCL queue (stream) used in the BLAS++ queue.
    /// Kernels executing on the current stream will continue.
    /// Note BLAS++ generally assumes streams are in-order, which SYCL
    /// queues are not by default. See sycl::property::queue::in_order.
    // todo: check that stream is in-order?
    void Queue::set_stream( stream_t& stream )
    {
        streams_[ 0 ] = stream;
    }

#endif // HAVE_CUBLAS or HAVE_ROCBLAS

// -----------------------------------------------------------------------------
// Default destructor.
Queue::~Queue()
{
    try {
        #if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )
            internal_set_device( device_ );
            device_free( work_, *this );

            if (own_handle_) {
                handle_destroy( handle_ );
            }
            handle_ = nullptr;

            if (own_default_stream_) {
                stream_destroy( streams_[ 0 ] );
            }
            streams_[ 0 ] = nullptr;

            // Destroy streams (after default stream 0, handled above).
            for (int i = 1; i < MaxForkSize; ++i) {
                if (streams_[ i ] != nullptr) {
                    stream_destroy( streams_[ i ] );
                    streams_[ i ] = nullptr;
                }
            }

            // Destroy events (including 0).
            for (int i = 0; i < MaxForkSize; ++i) {
                if (events_[ i ] != nullptr) {
                    event_destroy( events_[ i ] );
                    events_[ i ] = nullptr;
                }
            }

        #elif defined( BLAS_HAVE_SYCL )
            // wait for any completions
            streams_[ 0 ].wait();
            // free any work allocation
            device_free( work_, *this );
            // sycl::queue implicitly destructed.
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
    #if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )
        for (int i = 0; i < num_active_streams_; ++i) {
            stream_synchronize( streams_[ i ] );
        }

    #elif defined( BLAS_HAVE_SYCL )
        // todo: see wait_and_throw()
        streams_[ 0 ].wait();
    #endif
}

// -----------------------------------------------------------------------------
/// Forks the kernel launches assigned to this queue to parallel streams.
/// Limits the actual number of streams to <= MaxForkSize.
/// This function is not nested (you must join after each fork).
void Queue::fork( int num_streams )
{
    #if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )
        if (num_active_streams_ > 1)
            throw blas::Error( "can't nest fork regions", __func__ );

        assert( current_stream_index_ == 0 );

        // Create streams and events on first call with >= num_active_streams_.
        // streams_[ 0 ] already exists.
        num_active_streams_ = min( num_streams, MaxForkSize );
        if (streams_[ num_active_streams_ - 1 ] == nullptr) {
            for (int i = 1; i < num_active_streams_; ++i) {
                if (streams_[ i ] == nullptr) {
                    streams_[ i ] = stream_create();
                    events_[ i ]  = event_create();
                }
            }
        }

        // Create default event on first call.
        if (events_[ 0 ] == nullptr) {
            events_[ 0 ] = event_create();
        }

        // Make sure dependencies are respected:
        // all other streams wait for streams_[ 0 ].
        event_record( events_[ 0 ], streams_[ 0 ] );
        for (int i = 1; i < num_active_streams_; ++i) {
            stream_wait_event( streams_[ i ], events_[ 0 ], 0 );
        }

    #elif defined( BLAS_HAVE_SYCL )
        // todo: see possible implementations for sycl
        return;
    #endif
}

// -----------------------------------------------------------------------------
/// Switch executions on this queue back from parallel streams to the default
/// stream. This function is not nested (you must join after each fork).
void Queue::join()
{
    #if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )
        // Make sure dependencies are respected:
        // streams_[ 0 ] waits for all other streams.
        for (int i = 1; i < num_active_streams_; ++i) {
            event_record( events_[ i ], streams_[ i ] );
            stream_wait_event( streams_[ 0 ], events_[ i ], 0 );
        }

        // Assign current stream.
        current_stream_index_ = 0;
        num_active_streams_   = 1;

        // Assign current stream to BLAS handle.
        handle_set_stream( handle_, streams_[ current_stream_index_ ] );

    #elif defined( BLAS_HAVE_SYCL )
        // todo: see possible implementations for sycl
        return;
    #endif
}

// -----------------------------------------------------------------------------
/// In fork mode, switch execution to the next-in-line stream.
/// In join mode, no effect.
void Queue::revolve()
{
    #if defined( BLAS_HAVE_CUBLAS ) || defined( BLAS_HAVE_ROCBLAS )
        // Choose the next-in-line stream.
        current_stream_index_ = (current_stream_index_ + 1) % num_active_streams_;

        // Assign current stream to BLAS handle.
        handle_set_stream( handle_, streams_[ current_stream_index_ ] );

    #elif defined( BLAS_HAVE_SYCL )
        // todo: see possible implementations for sycl
        return;
    #endif
}

}  // namespace blas
