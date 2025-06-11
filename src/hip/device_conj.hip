#include "blas/device.hh"
#include <hip/hip_complex.h>

#if defined(BLAS_HAVE_ROCBLAS)

namespace blas {

__device__ std::complex<float> conj_convert(
    std::complex<float> z)
{
    hipFloatComplex res = hipConjf(*(hipFloatComplex*) &z);
    return *(std::complex<float>*) &res;
}

__device__ std::complex<double> conj_convert(
    std::complex<double> z)
{
    hipDoubleComplex res = hipConj(*(hipDoubleComplex*) &z);
    return *(std::complex<double>*) &res;
}

template <typename TS, typename TD>
__global__ void conj_kernel(
    int64_t n,
    TS const* src, int64_t inc_src, int64_t i_src,
    TD* dst, int64_t inc_dst, int64_t i_dst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[ i*inc_dst + i_dst ] = conj_convert( src[ i*inc_src + i_src ] );
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
    int64_t n_blocks = ceildiv(n, n_threads);

    int64_t i_src = (inc_src > 0 ? 0 : (1 - n) * inc_src);
    int64_t i_dst = (inc_dst > 0 ? 0 : (1 - n) * inc_dst);

    blas_dev_call(
        hipSetDevice( queue.device() ) );

    conj_kernel<<<n_blocks, n_threads, 0, queue.stream()>>>(
        n, src, inc_src, i_src, dst, inc_dst, i_dst );

    blas_dev_call(
        hipGetLastError() );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template void conj(
    int64_t n,
    std::complex<float> const* src, int64_t inc_src,
    std::complex<float>* dst, int64_t inc_dst,
    blas::Queue& queue);

template void conj(
    int64_t n,
    std::complex<double> const* src, int64_t inc_src,
    std::complex<double>* dst, int64_t inc_dst,
    blas::Queue& queue);

} // namespace blas

#endif // BLAS_HAVE_ROCBLAS
