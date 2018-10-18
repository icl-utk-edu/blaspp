#include <limits>
#include <cstring>
#include "batch_common.hh"
#include "blas.hh"

// -----------------------------------------------------------------------------
/// @ingroup symm
void blas::batch::symm(
    blas::Layout                layout,
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<int64_t>    const &m,
    std::vector<int64_t>    const &n,
    std::vector<float >     const &alpha,
    std::vector<float*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<float >     const &beta,
    std::vector<float*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info )
{
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if (info.size() > 0) {
        // perform error checking
        blas::batch::symm_check<float>(
                        layout, side, uplo,
                        m, n,
                        alpha, Aarray, ldda,
                               Barray, lddb,
                        beta,  Carray, lddc,
                        batch, info );
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < batch; ++i) {
        Side side_   = blas::batch::extract<Side>(side, i);
        Uplo uplo_   = blas::batch::extract<Uplo>(uplo, i);
        int64_t m_   = blas::batch::extract<int64_t>(m, i);
        int64_t n_   = blas::batch::extract<int64_t>(n, i);
        int64_t lda_ = blas::batch::extract<int64_t>(ldda, i);
        int64_t ldb_ = blas::batch::extract<int64_t>(lddb, i);
        int64_t ldc_ = blas::batch::extract<int64_t>(lddc, i);
        float alpha_ = blas::batch::extract<float>(alpha, i);
        float beta_  = blas::batch::extract<float>(beta, i);
        float* dA_   = blas::batch::extract<float*>(Aarray, i);
        float* dB_   = blas::batch::extract<float*>(Barray, i);
        float* dC_   = blas::batch::extract<float*>(Carray, i);
        blas::symm(
            layout, side_, uplo_, m_, n_,
            alpha_, dA_, lda_ ,
                    dB_, ldb_ ,
            beta_,  dC_, ldc_ );
    }
}

// -----------------------------------------------------------------------------
/// @ingroup symm
void blas::batch::symm(
    blas::Layout                layout,
    std::vector<blas::Side>  const &side,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<int64_t>     const &m,
    std::vector<int64_t>     const &n,
    std::vector<double >     const &alpha,
    std::vector<double*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<double >     const &beta,
    std::vector<double*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                     std::vector<int64_t>       &info )
{
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if (info.size() > 0) {
        // perform error checking
        blas::batch::symm_check<double>(
                        layout, side, uplo,
                        m, n,
                        alpha, Aarray, ldda,
                               Barray, lddb,
                        beta,  Carray, lddc,
                        batch, info );
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < batch; ++i) {
        Side side_   = blas::batch::extract<Side>(side, i);
        Uplo uplo_   = blas::batch::extract<Uplo>(uplo, i);
        int64_t m_   = blas::batch::extract<int64_t>(m, i);
        int64_t n_   = blas::batch::extract<int64_t>(n, i);
        int64_t lda_ = blas::batch::extract<int64_t>(ldda, i);
        int64_t ldb_ = blas::batch::extract<int64_t>(lddb, i);
        int64_t ldc_ = blas::batch::extract<int64_t>(lddc, i);
        double alpha_ = blas::batch::extract<double>(alpha, i);
        double beta_  = blas::batch::extract<double>(beta, i);
        double* dA_   = blas::batch::extract<double*>(Aarray, i);
        double* dB_   = blas::batch::extract<double*>(Barray, i);
        double* dC_   = blas::batch::extract<double*>(Carray, i);
        blas::symm(
            layout, side_, uplo_, m_, n_,
            alpha_, dA_, lda_ ,
                    dB_, ldb_ ,
            beta_,  dC_, ldc_ );
    }
}

// -----------------------------------------------------------------------------
/// @ingroup symm
void blas::batch::symm(
    blas::Layout                layout,
    std::vector<blas::Side>  const &side,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<int64_t>     const &m,
    std::vector<int64_t>     const &n,
    std::vector<std::complex<float> >     const &alpha,
    std::vector<std::complex<float>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<float>*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<std::complex<float> >     const &beta,
    std::vector<std::complex<float>*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                                  std::vector<int64_t>       &info )
{
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if (info.size() > 0) {
        // perform error checking
        blas::batch::symm_check<std::complex<float>>(
                        layout, side, uplo,
                        m, n,
                        alpha, Aarray, ldda,
                               Barray, lddb,
                        beta,  Carray, lddc,
                        batch, info );
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < batch; ++i) {
        Side side_   = blas::batch::extract<Side>(side, i);
        Uplo uplo_   = blas::batch::extract<Uplo>(uplo, i);
        int64_t m_   = blas::batch::extract<int64_t>(m, i);
        int64_t n_   = blas::batch::extract<int64_t>(n, i);
        int64_t lda_ = blas::batch::extract<int64_t>(ldda, i);
        int64_t ldb_ = blas::batch::extract<int64_t>(lddb, i);
        int64_t ldc_ = blas::batch::extract<int64_t>(lddc, i);
        std::complex<float> alpha_ = blas::batch::extract<std::complex<float>>(alpha, i);
        std::complex<float> beta_  = blas::batch::extract<std::complex<float>>(beta, i);
        std::complex<float>* dA_   = blas::batch::extract<std::complex<float>*>(Aarray, i);
        std::complex<float>* dB_   = blas::batch::extract<std::complex<float>*>(Barray, i);
        std::complex<float>* dC_   = blas::batch::extract<std::complex<float>*>(Carray, i);
        blas::symm(
            layout, side_, uplo_, m_, n_,
            alpha_, dA_, lda_ ,
                    dB_, ldb_ ,
            beta_,  dC_, ldc_ );
    }
}

// -----------------------------------------------------------------------------
/// @ingroup symm
void blas::batch::symm(
    blas::Layout                layout,
    std::vector<blas::Side>  const &side,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<int64_t>     const &m,
    std::vector<int64_t>     const &n,
    std::vector<std::complex<double> >     const &alpha,
    std::vector<std::complex<double>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<double>*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<std::complex<double> >     const &beta,
    std::vector<std::complex<double>*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                                   std::vector<int64_t>       &info )
{
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if (info.size() > 0) {
        // perform error checking
        blas::batch::symm_check<std::complex<double>>(
                        layout, side, uplo,
                        m, n,
                        alpha, Aarray, ldda,
                               Barray, lddb,
                        beta,  Carray, lddc,
                        batch, info );
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < batch; ++i) {
        Side side_   = blas::batch::extract<Side>(side, i);
        Uplo uplo_   = blas::batch::extract<Uplo>(uplo, i);
        int64_t m_   = blas::batch::extract<int64_t>(m, i);
        int64_t n_   = blas::batch::extract<int64_t>(n, i);
        int64_t lda_ = blas::batch::extract<int64_t>(ldda, i);
        int64_t ldb_ = blas::batch::extract<int64_t>(lddb, i);
        int64_t ldc_ = blas::batch::extract<int64_t>(lddc, i);
        std::complex<double> alpha_ = blas::batch::extract<std::complex<double>>(alpha, i);
        std::complex<double> beta_  = blas::batch::extract<std::complex<double>>(beta, i);
        std::complex<double>* dA_   = blas::batch::extract<std::complex<double>*>(Aarray, i);
        std::complex<double>* dB_   = blas::batch::extract<std::complex<double>*>(Barray, i);
        std::complex<double>* dC_   = blas::batch::extract<std::complex<double>*>(Carray, i);
        blas::symm(
            layout, side_, uplo_, m_, n_,
            alpha_, dA_, lda_ ,
                    dB_, ldb_ ,
            beta_,  dC_, ldc_ );
    }
}
