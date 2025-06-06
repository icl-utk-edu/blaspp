// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_DEVICE_NAMES_HH
#define BLAS_DEVICE_NAMES_HH

#include "blas/device.hh"

#include <complex>

namespace blas {

//------------------------------------------------------------------------------
/// @see to_device_blas_int
///
inline device_blas_int to_device_blas_int_( int64_t x, const char* x_str )
{
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if_msg( std::abs( x ) > std::numeric_limits<device_blas_int>::max(),
                           "%s", x_str );
    }
    return device_blas_int( x );
}

//----------------------------------------
/// Convert int64_t to device_blas_int.
/// If device_blas_int is 64-bit, this does nothing.
/// If device_blas_int is 32-bit, throws if x > INT_MAX, so conversion would overflow.
///
/// Note this is in src/device_internal.hh, so this macro won't pollute
/// the namespace when apps #include <blas.hh>.
///
#define to_device_blas_int( x ) to_device_blas_int_( x, #x )


#if defined( BLAS_HAVE_SYCL )

//==============================================================================
/// Thread-safe Scott Meyers' Singleton to enumerate devices on first call.
///
class DeviceList
{
public:
    //----------------------------------------
    /// @return DeviceList singleton, which will be created on the first call.
    ///
    static DeviceList& get()
    {
        static DeviceList s_list;
        return s_list;
    }

    //----------------------------------------
    /// @return SYCL GPU device at index `device`.
    /// Queries SYCL GPU devices on first call.
    ///
    /// @param[in] device
    ///     Index of device to fetch.
    ///
    static sycl::device const& at( int device )
    {
        DeviceList& s_list = get();
        return s_list.devices_.at( device );
    }

    //----------------------------------------
    /// @return Number of SYCL GPU devices.
    /// Queries SYCL GPU devices on first call.
    ///
    static int size()
    {
        DeviceList& s_list = get();
        return int( s_list.devices_.size() );
    }

private:
    //----------------------------------------
    /// Construct DeviceList, which queries SYCL for GPU devices on all
    /// platforms.
    ///
    DeviceList()
    {
        auto platforms = sycl::platform::get_platforms();
        for (auto &platform : platforms) {
            auto all_devices = platform.get_devices();
            for (auto &device : all_devices) {
                if (device.is_gpu()) {
                    devices_.push_back( device );
                }
            }
        }
    }

    /// Copy of singleton prohibited.
    DeviceList( DeviceList const& ) = delete;

    /// Assignment of singleton prohibited.
    DeviceList& operator = ( DeviceList const& ) = delete;

    /// Vector of SYCL GPU devices.
    std::vector< sycl::device > devices_;
};

#endif // SYCL


//==============================================================================
// Light wrappers around CUDA and cuBLAS functions.
// These are used in src/device_queue.cc and test/test_util.cc
// to access vendor-specific implementations.

#if defined( BLAS_HAVE_CUBLAS )

//------------------------------------------------------------------------------
inline cudaStream_t stream_create()
{
    cudaStream_t stream;
    blas_dev_call( cudaStreamCreate( &stream ) );
    return stream;
}

//------------------------------------------------------------------------------
inline cudaStream_t stream_create( int device )
{
    internal_set_device( device );
    cudaStream_t stream;
    blas_dev_call( cudaStreamCreate( &stream ) );
    return stream;
}

//------------------------------------------------------------------------------
inline void stream_destroy( cudaStream_t stream )
{
    blas_dev_call( cudaStreamDestroy( stream ) );
}

//------------------------------------------------------------------------------
inline void stream_synchronize( cudaStream_t stream )
{
    blas_dev_call( cudaStreamSynchronize( stream ) );
}

//------------------------------------------------------------------------------
inline cublasHandle_t handle_create( cudaStream_t stream )
{
    cublasHandle_t handle;
    blas_dev_call( cublasCreate( &handle ) );
    blas_dev_call( cublasSetStream( handle, stream ) );
    return handle;
}

//------------------------------------------------------------------------------
inline void handle_destroy( cublasHandle_t handle )
{
    blas_dev_call( cublasDestroy( handle ) );
}

//------------------------------------------------------------------------------
inline void handle_set_stream( cublasHandle_t handle, cudaStream_t stream )
{
    blas_dev_call( cublasSetStream( handle, stream ) );
}

//------------------------------------------------------------------------------
inline cudaStream_t handle_get_stream( cublasHandle_t handle )
{
    cudaStream_t stream;
    blas_dev_call( cublasGetStream( handle, &stream ) );
    return stream;
}

//------------------------------------------------------------------------------
inline cudaEvent_t event_create()
{
    cudaEvent_t event;
    blas_dev_call( cudaEventCreate( &event ) );
    return event;
}

//------------------------------------------------------------------------------
inline void event_destroy( cudaEvent_t event )
{
    blas_dev_call( cudaEventDestroy( event ) );
}

//------------------------------------------------------------------------------
inline void event_record( cudaEvent_t event, cudaStream_t stream )
{
    blas_dev_call( cudaEventRecord( event, stream ) );
}

//------------------------------------------------------------------------------
inline void stream_wait_event(
    cudaStream_t stream, cudaEvent_t event, unsigned int flags )
{
    blas_dev_call( cudaStreamWaitEvent( stream, event, flags ) );
}

//==============================================================================
// Light wrappers around HIP and rocBLAS functions.

#elif defined( BLAS_HAVE_ROCBLAS )

//------------------------------------------------------------------------------
/// Creates stream on current device.
inline hipStream_t stream_create()
{
    hipStream_t stream;
    blas_dev_call( hipStreamCreate( &stream ) );
    return stream;
}

//------------------------------------------------------------------------------
/// Creates stream on given device.
inline hipStream_t stream_create( int device )
{
    internal_set_device( device );
    hipStream_t stream;
    blas_dev_call( hipStreamCreate( &stream ) );
    return stream;
}

//------------------------------------------------------------------------------
inline void stream_destroy( hipStream_t stream )
{
    blas_dev_call( hipStreamDestroy( stream ) );
}

//------------------------------------------------------------------------------
inline void stream_synchronize( hipStream_t stream )
{
    blas_dev_call( hipStreamSynchronize( stream ) );
}

//------------------------------------------------------------------------------
inline rocblas_handle handle_create( hipStream_t stream )
{
    rocblas_handle handle;
    blas_dev_call( rocblas_create_handle( &handle ) );
    blas_dev_call( rocblas_set_stream( handle, stream ) );
    return handle;
}

//------------------------------------------------------------------------------
inline void handle_destroy( rocblas_handle handle )
{
    blas_dev_call( rocblas_destroy_handle( handle ) );
}

//------------------------------------------------------------------------------
inline void handle_set_stream( rocblas_handle handle, hipStream_t stream )
{
    blas_dev_call( rocblas_set_stream( handle, stream ) );
}

//------------------------------------------------------------------------------
inline hipStream_t handle_get_stream( rocblas_handle handle )
{
    hipStream_t stream;
    blas_dev_call( rocblas_get_stream( handle, &stream ) );
    return stream;
}

//------------------------------------------------------------------------------
inline hipEvent_t event_create()
{
    hipEvent_t event;
    blas_dev_call( hipEventCreate( &event ) );
    return event;
}

//------------------------------------------------------------------------------
inline void event_destroy( hipEvent_t event )
{
    blas_dev_call( hipEventDestroy( event ) );
}

//------------------------------------------------------------------------------
inline void event_record( hipEvent_t event, hipStream_t stream )
{
    blas_dev_call( hipEventRecord( event, stream ) );
}

//------------------------------------------------------------------------------
inline void stream_wait_event(
    hipStream_t stream, hipEvent_t event, unsigned int flags )
{
    blas_dev_call( hipStreamWaitEvent( stream, event, flags ) );
}

#endif  // BLAS_HAVE_ROCBLAS

namespace internal {

//==============================================================================
// Level 1 BLAS - Device Interfaces
// Alphabetical order

//------------------------------------------------------------------------------
void asum(
    device_blas_int n,
    float const* dx, device_blas_int incdx,
    float* result,
    blas::Queue& queue );

void asum(
    device_blas_int n,
    double const* dx, device_blas_int incdx,
    double* result,
    blas::Queue& queue );

void asum(
    device_blas_int n,
    std::complex<float> const* dx, device_blas_int incdx,
    float* result,
    blas::Queue& queue );

void asum(
    device_blas_int n,
    std::complex<double> const* dx, device_blas_int incdx,
    double* result,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void axpy(
    device_blas_int n,
    float alpha,
    float const* dx, device_blas_int incdx,
    float *dy, device_blas_int incdy,
    blas::Queue& queue );

void axpy(
    device_blas_int n,
    double alpha,
    double const* dx, device_blas_int incdx,
    double *dy, device_blas_int incdy,
    blas::Queue& queue );

void axpy(
    device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const* dx, device_blas_int incdx,
    std::complex<float> *dy, device_blas_int incdy,
    blas::Queue& queue );

void axpy(
    device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const* dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void dot(
    device_blas_int n,
    float const *dx, device_blas_int incdx,
    float const *dy, device_blas_int incdy,
    float *result,
    blas::Queue& queue );

void dot(
    device_blas_int n,
    double const *dx, device_blas_int incdx,
    double const *dy, device_blas_int incdy,
    double *result,
    blas::Queue& queue );

void dot(
    device_blas_int n,
    std::complex<float> const *dx, device_blas_int incdx,
    std::complex<float> const *dy, device_blas_int incdy,
    std::complex<float> *result,
    blas::Queue& queue );

void dot(
    device_blas_int n,
    std::complex<double> const *dx, device_blas_int incdx,
    std::complex<double> const *dy, device_blas_int incdy,
    std::complex<double> *result,
    blas::Queue& queue );

void dotu(
    device_blas_int n,
    std::complex<float> const *dx, device_blas_int incdx,
    std::complex<float> const *dy, device_blas_int incdy,
    std::complex<float> *result,
    blas::Queue& queue );

void dotu(
    device_blas_int n,
    std::complex<double> const *dx, device_blas_int incdx,
    std::complex<double> const *dy, device_blas_int incdy,
    std::complex<double> *result,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void iamax(
    int64_t n,
    float const* x, int64_t incx,
    int64_t* result,
    blas::Queue& queue );

void iamax(
    int64_t n,
    double const* x, int64_t incx,
    int64_t* result,
    blas::Queue& queue );

void iamax(
    int64_t n,
    std::complex<float> const* x, int64_t incx,
    int64_t* result,
    blas::Queue& queue );

void iamax(
    int64_t n,
    std::complex<double> const* x, int64_t incx,
    int64_t* result,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void nrm2(
    device_blas_int n,
    float const* dx, device_blas_int incdx,
    float *result,
    blas::Queue& queue );

void nrm2(
    device_blas_int n,
    double const* dx, device_blas_int incdx,
    double *result,
    blas::Queue& queue );

void nrm2(
    device_blas_int n,
    std::complex<float> const* dx, device_blas_int incdx,
    float *result,
    blas::Queue& queue );

void nrm2(
    device_blas_int n,
    std::complex<double> const* dx, device_blas_int incdx,
    double *result,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void rot(
    device_blas_int n,
    float* dx, device_blas_int incdx,
    float* dy, device_blas_int incdy,
    const float c,
    const float s,
    blas::Queue& queue );

void rot(
    device_blas_int n,
    double* dx, device_blas_int incdx,
    double* dy, device_blas_int incdy,
    const double c,
    const double s,
    blas::Queue& queue );

void rot(
    device_blas_int n,
    std::complex<float>* dx, device_blas_int incdx,
    std::complex<float>* dy, device_blas_int incdy,
    const float c,
    const float s,
    blas::Queue& queue );

void rot(
    device_blas_int n,
    std::complex<double>* dx, device_blas_int incdx,
    std::complex<double>* dy, device_blas_int incdy,
    const double c,
    const double s,
    blas::Queue& queue );

void rot(
    device_blas_int n,
    std::complex<float>* dx, device_blas_int incdx,
    std::complex<float>* dy, device_blas_int incdy,
    const float c,
    const std::complex<float> s,
    blas::Queue& queue );

void rot(
    device_blas_int n,
    std::complex<double>* dx, device_blas_int incdx,
    std::complex<double>* dy, device_blas_int incdy,
    const double c,
    const std::complex<double> s,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void rotg(
    float* da,
    float* db,
    float* dc,
    float* ds,
    blas::Queue& queue );

void rotg(
    double* da,
    double* db,
    double* dc,
    double* ds,
    blas::Queue& queue );

void rotg(
    std::complex<float>* da,
    std::complex<float>* db,
    float* dc,
    std::complex<float>* ds,
    blas::Queue& queue );

void rotg(
    std::complex<double>* da,
    std::complex<double>* db,
    double* dc,
    std::complex<double>* ds,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void rotm(
    device_blas_int n,
    float* dx, device_blas_int incdx,
    float* dy, device_blas_int incdy,
    const float* param,
    blas::Queue& queue );

void rotm(
    device_blas_int n,
    double* dx, device_blas_int incdx,
    double* dy, device_blas_int incdy,
    const double* param,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void rotmg(
    float* d1,
    float* d2,
    float* x1,
    float* y1,
    float* param,
    blas::Queue& queue );

void rotmg(
    double* d1,
    double* d2,
    double* x1,
    double* y1,
    double* param,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void scal(
    device_blas_int n,
    float alpha,
    float *dx, device_blas_int incdx,
    blas::Queue& queue );

void scal(
    device_blas_int n,
    double alpha,
    double *dx, device_blas_int incdx,
    blas::Queue& queue );

void scal(
    device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> *dx, device_blas_int incdx,
    blas::Queue& queue );

void scal(
    device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> *dx, device_blas_int incdx,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void swap(
    device_blas_int n,
    float *dx, device_blas_int incdx,
    float *dy, device_blas_int incdy,
    blas::Queue& queue );

void swap(
    device_blas_int n,
    double *dx, device_blas_int incdx,
    double *dy, device_blas_int incdy,
    blas::Queue& queue );

void swap(
    device_blas_int n,
    std::complex<float> *dx, device_blas_int incdx,
    std::complex<float> *dy, device_blas_int incdy,
    blas::Queue& queue );

void swap(
    device_blas_int n,
    std::complex<double> *dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void copy(
    device_blas_int n,
    float const *dx, device_blas_int incdx,
    float *dy, device_blas_int incdy,
    blas::Queue& queue );

void copy(
    device_blas_int n,
    double const *dx, device_blas_int incdx,
    double *dy, device_blas_int incdy,
    blas::Queue& queue );

void copy(
    device_blas_int n,
    std::complex<float> const *dx, device_blas_int incdx,
    std::complex<float> *dy, device_blas_int incdy,
    blas::Queue& queue );

void copy(
    device_blas_int n,
    std::complex<double> const *dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy,
    blas::Queue& queue );


//==============================================================================
// Level 2 BLAS - Device Interfaces

//------------------------------------------------------------------------------
// gemv
//------------------------------------------------------------------------------
void gemv(
    blas::Op trans,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const* dA, device_blas_int ldda,
    float const* dx, device_blas_int incdx,
    float beta,
    float*       dy, device_blas_int incdy,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void gemv(
    blas::Op trans,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const* dA, device_blas_int ldda,
    double const* dx, device_blas_int incdx,
    double beta,
    double*       dy, device_blas_int incdy,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void gemv(
    blas::Op trans,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const* dA, device_blas_int ldda,
    std::complex<float> const* dx, device_blas_int incdx,
    std::complex<float> beta,
    std::complex<float>*       dy, device_blas_int incdy,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void gemv(
    blas::Op trans,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const* dA, device_blas_int ldda,
    std::complex<double> const* dx, device_blas_int incdx,
    std::complex<double> beta,
    std::complex<double>*       dy, device_blas_int incdy,
    blas::Queue& queue );

//==============================================================================
// Level 3 BLAS - Device Interfaces

//------------------------------------------------------------------------------
void gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    float alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float beta,
    float       *dC, device_blas_int lddc,
    blas::Queue& queue );

void gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    double alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double beta,
    double       *dC, device_blas_int lddc,
    blas::Queue& queue );

void gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float> beta,
    std::complex<float>       *dC, device_blas_int lddc,
    blas::Queue& queue );

void gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double> beta,
    std::complex<double>       *dC, device_blas_int lddc,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const *dA, device_blas_int ldda,
    float       *dB, device_blas_int lddb,
    blas::Queue& queue );

void trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const *dA, device_blas_int ldda,
    double       *dB, device_blas_int lddb,
    blas::Queue& queue );

void trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>       *dB, device_blas_int lddb,
    blas::Queue& queue );

void trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>       *dB, device_blas_int lddb,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void trmm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const *dA, device_blas_int ldda,
    float       *dB, device_blas_int lddb,
    blas::Queue& queue );

void trmm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const *dA, device_blas_int ldda,
    double       *dB, device_blas_int lddb,
    blas::Queue& queue );

void trmm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>       *dB, device_blas_int lddb,
    blas::Queue& queue );

void trmm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>       *dB, device_blas_int lddb,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void hemm(
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc,
    blas::Queue& queue );

void hemm(
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void symm(
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    float  alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float  beta,
    float* dC, device_blas_int lddc,
    blas::Queue& queue );

void symm(
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    double  alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double  beta,
    double* dC, device_blas_int lddc,
    blas::Queue& queue );

void symm(
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc,
    blas::Queue& queue );

void symm(
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void herk(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    float alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    float  beta,
    std::complex<float>* dC, device_blas_int lddc,
    blas::Queue& queue );

void herk(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    double alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    double  beta,
    std::complex<double>* dC, device_blas_int lddc,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void syrk(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    float alpha,
    float const *dA, device_blas_int ldda,
    float  beta,
    float* dC, device_blas_int lddc,
    blas::Queue& queue );

void syrk(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    double alpha,
    double const *dA, device_blas_int ldda,
    double  beta,
    double* dC, device_blas_int lddc,
    blas::Queue& queue );

void syrk(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc,
    blas::Queue& queue );

void syrk(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void her2k(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    float  beta,
    std::complex<float>* dC, device_blas_int lddc,
    blas::Queue& queue );

void her2k(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    double  beta,
    std::complex<double>* dC, device_blas_int lddc,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void syr2k(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    float  alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float  beta,
    float* dC, device_blas_int lddc,
    blas::Queue& queue );

void syr2k(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    double  alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double  beta,
    double* dC, device_blas_int lddc,
    blas::Queue& queue );

void syr2k(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc,
    blas::Queue& queue );

void syr2k(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc,
    blas::Queue& queue );

//------------------------------------------------------------------------------
// batch gemm
void batch_gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    float alpha,
    float const * const * dAarray, device_blas_int ldda,
    float const * const * dBarray, device_blas_int lddb,
    float beta,
    float** dCarray, device_blas_int lddc,
    device_blas_int batch_size,
    blas::Queue& queue );

void batch_gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    double alpha,
    double const * const * dAarray, device_blas_int ldda,
    double const * const * dBarray, device_blas_int lddb,
    double beta,
    double** dCarray, device_blas_int lddc,
    device_blas_int batch_size,
    blas::Queue& queue );

void batch_gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<float> alpha,
    std::complex<float> const * const * dAarray, device_blas_int ldda,
    std::complex<float> const * const * dBarray, device_blas_int lddb,
    std::complex<float> beta,
    std::complex<float>** dCarray, device_blas_int lddc,
    device_blas_int batch_size,
    blas::Queue& queue );

void batch_gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<double> alpha,
    std::complex<double> const * const * dAarray, device_blas_int ldda,
    std::complex<double> const * const * dBarray, device_blas_int lddb,
    std::complex<double> beta,
    std::complex<double>** dCarray, device_blas_int lddc,
    device_blas_int batch_size,
    blas::Queue& queue );

//------------------------------------------------------------------------------
// batch trsm
void batch_trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const * const * dAarray, device_blas_int ldda,
    float const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size,
    blas::Queue& queue );

void batch_trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const * const * dAarray, device_blas_int ldda,
    double const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size,
    blas::Queue& queue );

void batch_trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const * const * dAarray, device_blas_int ldda,
    std::complex<float> const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size,
    blas::Queue& queue );

void batch_trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const * const * dAarray, device_blas_int ldda,
    std::complex<double> const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size,
    blas::Queue& queue );

}  // namespace internal
}  // namespace blas

#endif        //  #ifndef BLAS_DEVICE_NAMES_HH
