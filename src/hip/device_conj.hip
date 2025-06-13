#include "blas/device.hh"
#include <hip/hip_complex.h>

#if defined(BLAS_HAVE_ROCBLAS)

namespace blas {

// Overload conj for rocblas_float_complex
__device__ inline rocblas_float_complex conj(rocblas_float_complex z) {
    return hipConjf(z);
}

// Overload conj for rocblas_double_complex
__device__ inline rocblas_double_complex conj(rocblas_double_complex z) {
    return hipConj(z);
}

template <typename TS, typename TD>
__global__ void conj_kernel(
    int64_t n,
    TS const* src, int64_t inc_src,
    TD* dst, int64_t inc_dst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[ i*inc_dst ] = conj( src[ i*inc_src ] );
}

template <typename TS, typename TD>
void conj(
    int64_t n,
    TS const* src, int64_t inc_src,
    TD* dst, int64_t inc_dst,
    blas::Queue& queue )
{
    if (n <= 0) {
        return;
    }

    const int BlockSize = 128;

    int64_t n_threads = std::min( int64_t( BlockSize ), n );
    int64_t n_blocks = 1 + ((n - 1) / n_threads);

    blas_dev_call(
        hipSetDevice( queue.device() ) );

    conj_kernel<<<n_blocks, n_threads, 0, queue.stream()>>>(
        n, src, inc_src, dst, inc_dst );

    blas_dev_call(
        hipGetLastError() );
}

} // namespace blas

#endif // BLAS_HAVE_ROCBLAS
