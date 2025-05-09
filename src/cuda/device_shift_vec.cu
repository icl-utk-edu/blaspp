#include "blas/device.hh"

#if defined(BLAS_HAVE_CUBLAS)

namespace blas {

template <typename scalar_t>
__global__ void shift_vec_kernel(
    int64_t n, scalar_t* v,
    scalar_t c)
{
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        v[ i ] += c;
    }
}

//------------------------------------------------------------------------------
/// Shifts each element of the vector v by a constant value c.
///
/// @param[in] n
///     Number of elements in the vector. n >= 0.
///
/// @param[in,out] v
///     Pointer to the vector of length n.
///     On exit, each element v[i] is updated as v[i] += c.
///
/// @param[in] c
///     Scalar value to be added to each element of v.
///
/// @param[in] queue
///     BLAS++ queue to execute in.
///
template <typename scalar_t>
void shift_vec(
    int64_t n, scalar_t* v,
    scalar_t c,
    blas::Queue& queue)
{
    if (n == 0) {
        return;
    }

    int64_t nthreads = std::min( int64_t( 1024 ), n );

    blas_dev_call(
        cudaSetDevice( queue.device() ) );

    shift_vec_kernel<<<1, nthreads, 0, queue.stream()>>>( n, v, c );

    blas_dev_call(
        cudaGetLastError() );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void shift_vec(
    int64_t n, int64_t* v,
    int64_t c,
    blas::Queue& queue);

template
void shift_vec(
    int64_t n, int* v,
    int c,
    blas::Queue& queue);

} // namespace blas

#endif // BLAS_HAVE_CUBLAS
