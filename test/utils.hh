#ifndef UTILS_HH
#define UTILS_HH

#include "blas.hh"

namespace blas {

template <typename src_t, typename dst_t = src_t>
void copy_matrix(
    int m, int n,
    src_t const* src, int ld_src,
    dst_t*       dst, int ld_dst,
    blas::Queue &queue);

} // namespace blas

#endif // UTILS_HH
