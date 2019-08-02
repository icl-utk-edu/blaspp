#include <limits>
#include <cstring>
#include "batch_common.hh"
#include "device_blas.hh"

// -----------------------------------------------------------------------------
/// @ingroup syr2k
void blas::batch::syr2k(
    blas::Layout                   layout,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<int64_t>    const &n,
    std::vector<int64_t>    const &k,
    std::vector<float >     const &alpha,
    std::vector<float*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<float >     const &beta,
    std::vector<float*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info,
    blas::Queue &queue )
{
    blas_error_if( layout != Layout::ColMajor && layout != Layout::RowMajor );
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if (info.size() > 0) {
        // perform error checking
        blas::batch::syr2k_check<float>(
                        layout, uplo, trans,
                        n, k,
                        alpha, Aarray, ldda,
                               Barray, lddb,
                        beta,  Carray, lddc,
                        batch, info );
    }

    blas::set_device( queue.device() );

    queue.fork();
    for (size_t i = 0; i < batch; ++i) {
        Uplo uplo_   = blas::batch::extract<Uplo>(uplo, i);
        Op   trans_  = blas::batch::extract<Op>(trans, i);
        int64_t n_   = blas::batch::extract<int64_t>(n, i);
        int64_t k_   = blas::batch::extract<int64_t>(k, i);
        int64_t lda_ = blas::batch::extract<int64_t>(ldda, i);
        int64_t ldb_ = blas::batch::extract<int64_t>(lddb, i);
        int64_t ldc_ = blas::batch::extract<int64_t>(lddc, i);
        float alpha_ = blas::batch::extract<float>(alpha, i);
        float beta_  = blas::batch::extract<float>(beta, i);
        float* dA_   = blas::batch::extract<float*>(Aarray, i);
        float* dB_   = blas::batch::extract<float*>(Barray, i);
        float* dC_   = blas::batch::extract<float*>(Carray, i);
        blas::syr2k(
            layout, uplo_, trans_, n_, k_,
            alpha_, dA_, lda_ ,
                    dB_, ldb_ ,
            beta_,  dC_, ldc_ , queue );
        queue.revolve();
    }
    queue.join();
}

// -----------------------------------------------------------------------------
/// @ingroup syr2k
void blas::batch::syr2k(
    blas::Layout                   layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<double >     const &alpha,
    std::vector<double*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<double >     const &beta,
    std::vector<double*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                     std::vector<int64_t>       &info,
    blas::Queue &queue )
{
    blas_error_if( layout != Layout::ColMajor && layout != Layout::RowMajor );
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if (info.size() > 0) {
        // perform error checking
        blas::batch::syr2k_check<double>(
                        layout, uplo, trans,
                        n, k,
                        alpha, Aarray, ldda,
                               Barray, lddb,
                        beta,  Carray, lddc,
                        batch, info );
    }

    blas::set_device( queue.device() );

    queue.fork();
    for (size_t i = 0; i < batch; ++i) {
        Uplo uplo_   = blas::batch::extract<Uplo>(uplo, i);
        Op   trans_  = blas::batch::extract<Op>(trans, i);
        int64_t n_   = blas::batch::extract<int64_t>(n, i);
        int64_t k_   = blas::batch::extract<int64_t>(k, i);
        int64_t lda_ = blas::batch::extract<int64_t>(ldda, i);
        int64_t ldb_ = blas::batch::extract<int64_t>(lddb, i);
        int64_t ldc_ = blas::batch::extract<int64_t>(lddc, i);
        double alpha_ = blas::batch::extract<double>(alpha, i);
        double beta_  = blas::batch::extract<double>(beta, i);
        double* dA_   = blas::batch::extract<double*>(Aarray, i);
        double* dB_   = blas::batch::extract<double*>(Barray, i);
        double* dC_   = blas::batch::extract<double*>(Carray, i);
        blas::syr2k(
            layout, uplo_, trans_, n_, k_,
            alpha_, dA_, lda_ ,
                    dB_, ldb_ ,
            beta_,  dC_, ldc_ , queue );
        queue.revolve();
    }
    queue.join();
}

// -----------------------------------------------------------------------------
/// @ingroup syr2k
void blas::batch::syr2k(
    blas::Layout                   layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<std::complex<float> > const &alpha,
    std::vector<std::complex<float>*> const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<float>*> const &Barray, std::vector<int64_t> const &lddb,
    std::vector<std::complex<float> > const &beta,
    std::vector<std::complex<float>*> const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch, std::vector<int64_t>       &info,
    blas::Queue &queue )
{
    blas_error_if( layout != Layout::ColMajor && layout != Layout::RowMajor );
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if (info.size() > 0) {
        // perform error checking
        blas::batch::syr2k_check<std::complex<float>>(
                        layout, uplo, trans,
                        n, k,
                        alpha, Aarray, ldda,
                               Barray, lddb,
                        beta,  Carray, lddc,
                        batch, info );
    }

    blas::set_device( queue.device() );

    queue.fork();
    for (size_t i = 0; i < batch; ++i) {
        Uplo uplo_   = blas::batch::extract<Uplo>(uplo, i);
        Op   trans_  = blas::batch::extract<Op>(trans, i);
        int64_t n_   = blas::batch::extract<int64_t>(n, i);
        int64_t k_   = blas::batch::extract<int64_t>(k, i);
        int64_t lda_ = blas::batch::extract<int64_t>(ldda, i);
        int64_t ldb_ = blas::batch::extract<int64_t>(lddb, i);
        int64_t ldc_ = blas::batch::extract<int64_t>(lddc, i);
        std::complex<float> alpha_ = blas::batch::extract<std::complex<float>>(alpha, i);
        std::complex<float> beta_  = blas::batch::extract<std::complex<float>>(beta, i);
        std::complex<float>* dA_   = blas::batch::extract<std::complex<float>*>(Aarray, i);
        std::complex<float>* dB_   = blas::batch::extract<std::complex<float>*>(Barray, i);
        std::complex<float>* dC_   = blas::batch::extract<std::complex<float>*>(Carray, i);
        blas::syr2k(
            layout, uplo_, trans_, n_, k_,
            alpha_, dA_, lda_ ,
                    dB_, ldb_ ,
            beta_,  dC_, ldc_ , queue );
        queue.revolve();
    }
    queue.join();
}

// -----------------------------------------------------------------------------
/// @ingroup syr2k
void blas::batch::syr2k(
    blas::Layout                   layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<std::complex<double> > const &alpha,
    std::vector<std::complex<double>*> const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<double>*> const &Barray, std::vector<int64_t> const &lddb,
    std::vector<std::complex<double> > const &beta,
    std::vector<std::complex<double>*> const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch, std::vector<int64_t>       &info,
    blas::Queue &queue )
{
    blas_error_if( layout != Layout::ColMajor && layout != Layout::RowMajor );
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if (info.size() > 0) {
        // perform error checking
        blas::batch::syr2k_check<std::complex<double>>(
                        layout, uplo, trans,
                        n, k,
                        alpha, Aarray, ldda,
                               Barray, lddb,
                        beta,  Carray, lddc,
                        batch, info );
    }

    blas::set_device( queue.device() );

    queue.fork();
    for (size_t i = 0; i < batch; ++i) {
        Uplo uplo_   = blas::batch::extract<Uplo>(uplo, i);
        Op   trans_  = blas::batch::extract<Op>(trans, i);
        int64_t n_   = blas::batch::extract<int64_t>(n, i);
        int64_t k_   = blas::batch::extract<int64_t>(k, i);
        int64_t lda_ = blas::batch::extract<int64_t>(ldda, i);
        int64_t ldb_ = blas::batch::extract<int64_t>(lddb, i);
        int64_t ldc_ = blas::batch::extract<int64_t>(lddc, i);
        std::complex<double> alpha_ = blas::batch::extract<std::complex<double>>(alpha, i);
        std::complex<double> beta_  = blas::batch::extract<std::complex<double>>(beta, i);
        std::complex<double>* dA_   = blas::batch::extract<std::complex<double>*>(Aarray, i);
        std::complex<double>* dB_   = blas::batch::extract<std::complex<double>*>(Barray, i);
        std::complex<double>* dC_   = blas::batch::extract<std::complex<double>*>(Carray, i);
        blas::syr2k(
            layout, uplo_, trans_, n_, k_,
            alpha_, dA_, lda_ ,
                    dB_, ldb_ ,
            beta_,  dC_, ldc_ , queue );
        queue.revolve();
    }
    queue.join();
}
