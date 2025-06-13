#include "device.hh"

#if defined(BLAS_HAVE_SYCL)

namespace blas {

template <typename TS, typename TD>
void conj(
    int64_t n,
    TS const* src, int64_t inc_src,
    TD* dst, int64_t inc_dst,
    blas::Queue& queue )
{
    using std::conj;

    if (n <= 0) return;
    queue.stream().submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            dst[i * inc_dst] = conj(src[i * inc_src]);
        });
    });
}

} // namespace blas

#endif // BLAS_HAVE_SYCL