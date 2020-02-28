#include <limits>
#include <cstring>
#include "blas/batch_common.hh"
#include "blas/device_blas.hh"

// -----------------------------------------------------------------------------
/// @ingroup gemm
void blas::batch::gemm(
    blas::Layout                 layout,
    std::vector<blas::Op> const &transA,
    std::vector<blas::Op> const &transB,
    std::vector<int64_t>  const &m,
    std::vector<int64_t>  const &n,
    std::vector<int64_t>  const &k,
    std::vector<float >   const &alpha,
    std::vector<float*>   const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float*>   const &Barray, std::vector<int64_t> const &lddb,
    std::vector<float >   const &beta,
    std::vector<float*>   const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                  std::vector<int64_t>       &info,
    blas::Queue &queue )
{
    blas_error_if( layout != Layout::ColMajor && layout != Layout::RowMajor );
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if (info.size() > 0) {
        // perform error checking
        blas::batch::gemm_check<float>( layout, transA, transB,
                                        m, n, k,
                                        alpha, Aarray, ldda,
                                               Barray, lddb,
                                        beta,  Carray, lddc,
                                        batch, info );
    }

    bool fixed_size =   ( transA.size() == 1     &&
                          transB.size() == 1     &&
                          m.size()      == 1     &&
                          n.size()      == 1     &&
                          k.size()      == 1     &&
                          alpha.size()  == 1     &&
                          Aarray.size() == batch &&
                          ldda.size()   == 1     &&
                          Barray.size() == batch &&
                          lddb.size()   == 1     &&
                          beta.size()   == 1     &&
                          Carray.size() == batch &&
                          lddc.size()   == 1 );

    blas::set_device( queue.device() );
    if (fixed_size) {
        // call the vendor routine
        device_trans_t  transA_ = blas::device_trans_const( transA[0] );
        device_trans_t  transB_ = blas::device_trans_const( transB[0] );
        device_blas_int m_      = (device_blas_int) m[0];
        device_blas_int n_      = (device_blas_int) n[0];
        device_blas_int k_      = (device_blas_int) k[0];
        device_blas_int ldda_   = (device_blas_int) ldda[0];
        device_blas_int lddb_   = (device_blas_int) lddb[0];
        device_blas_int lddc_   = (device_blas_int) lddc[0];

        size_t batch_limit = queue.get_batch_limit();
        float **dAarray, **dBarray, **dCarray;
        dAarray = (float**)queue.get_devPtrArray();
        dBarray = dAarray + batch_limit;
        dCarray = dBarray + batch_limit;

        for( size_t ib = 0; ib < batch; ib += batch_limit ) { 
            size_t ibatch = std::min( batch_limit, batch-ib );

            // copy Aarray, Barray, and Carray to device
            device_setvector<float*>(ibatch, (float**)&Aarray[ib], 1, dAarray, 1, queue);
            device_setvector<float*>(ibatch, (float**)&Barray[ib], 1, dBarray, 1, queue);
            device_setvector<float*>(ibatch, (float**)&Carray[ib], 1, dCarray, 1, queue);

            if (layout == Layout::RowMajor) {
                // swap transA <=> transB, m <=> n, B <=> A
                DEVICE_BATCH_sgemm( queue.handle(),
                                    transB_, transA_,
                                    n_, m_, k_,
                                    alpha[0], dBarray, lddb_, dAarray, ldda_,
                                    beta[0],  dCarray, lddc_,
                                    ibatch);
            }
            else {
                DEVICE_BATCH_sgemm( queue.handle(),
                                    transA_, transB_,
                                    m_, n_, k_,
                                    alpha[0], dAarray, ldda_, dBarray, lddb_,
                                    beta[0],  dCarray, lddc_,
                                    ibatch);
            }
        }
    }
    else {
        queue.fork();
        for (size_t i = 0; i < batch; ++i) {
            Op transA_   = blas::batch::extract<Op>(transA, i);
            Op transB_   = blas::batch::extract<Op>(transB, i);
            int64_t m_   = blas::batch::extract<int64_t>(m, i);
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
            blas::gemm(
                layout, transA_, transB_, m_, n_, k_,
                alpha_, dA_, lda_,
                        dB_, ldb_,
                beta_,  dC_, ldc_, queue );
            queue.revolve();
        }
        queue.join();
    }
}

// -----------------------------------------------------------------------------
/// @ingroup gemm
void blas::batch::gemm(
    blas::Layout                 layout,
    std::vector<blas::Op> const &transA,
    std::vector<blas::Op> const &transB,
    std::vector<int64_t>  const &m,
    std::vector<int64_t>  const &n,
    std::vector<int64_t>  const &k,
    std::vector<double >  const &alpha,
    std::vector<double*>  const &Aarray, std::vector<int64_t>  const &ldda,
    std::vector<double*>  const &Barray, std::vector<int64_t>  const &lddb,
    std::vector<double >  const &beta,
    std::vector<double*>  const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                  std::vector<int64_t>       &info,
    blas::Queue &queue )
{
    blas_error_if( layout != Layout::ColMajor && layout != Layout::RowMajor );
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if (info.size() > 0) {
        // perform error checking
        blas::batch::gemm_check<double>( layout, transA, transB,
                                        m, n, k,
                                        alpha, Aarray, ldda,
                                               Barray, lddb,
                                        beta,  Carray, lddc,
                                        batch, info );
    }
    bool fixed_size =   ( transA.size() == 1     &&
                          transB.size() == 1     &&
                          m.size()      == 1     &&
                          n.size()      == 1     &&
                          k.size()      == 1     &&
                          alpha.size()  == 1     &&
                          Aarray.size() == batch &&
                          ldda.size()   == 1     &&
                          Barray.size() == batch &&
                          lddb.size()   == 1     &&
                          beta.size()   == 1     &&
                          Carray.size() == batch &&
                          lddc.size()   == 1 );

    blas::set_device( queue.device() );
    if (fixed_size) {
        // call the vendor routine
        device_trans_t  transA_ = blas::device_trans_const( transA[0] );
        device_trans_t  transB_ = blas::device_trans_const( transB[0] );
        device_blas_int m_      = (device_blas_int) m[0];
        device_blas_int n_      = (device_blas_int) n[0];
        device_blas_int k_      = (device_blas_int) k[0];
        device_blas_int ldda_   = (device_blas_int) ldda[0];
        device_blas_int lddb_   = (device_blas_int) lddb[0];
        device_blas_int lddc_   = (device_blas_int) lddc[0];

        size_t batch_limit = queue.get_batch_limit();
        double **dAarray, **dBarray, **dCarray;
        dAarray = (double**)queue.get_devPtrArray();
        dBarray = dAarray + batch_limit;
        dCarray = dBarray + batch_limit;

        for( size_t ib = 0; ib < batch; ib += batch_limit ) { 
            size_t ibatch = std::min( batch_limit, batch-ib );

            // copy Aarray, Barray, and Carray to device
            device_setvector<double*>(ibatch, (double**)&Aarray[ib], 1, dAarray, 1, queue);
            device_setvector<double*>(ibatch, (double**)&Barray[ib], 1, dBarray, 1, queue);
            device_setvector<double*>(ibatch, (double**)&Carray[ib], 1, dCarray, 1, queue);

            if (layout == Layout::RowMajor) {
                // swap transA <=> transB, m <=> n, B <=> A
                DEVICE_BATCH_dgemm( queue.handle(),
                                    transB_, transA_,
                                    n_, m_, k_,
                                    alpha[0], dBarray, lddb_, dAarray, ldda_,
                                    beta[0],  dCarray, lddc_,
                                    ibatch);
            }
            else {
                DEVICE_BATCH_dgemm( queue.handle(),
                                    transA_, transB_,
                                    m_, n_, k_,
                                    alpha[0], dAarray, ldda_, dBarray, lddb_,
                                    beta[0],  dCarray, lddc_,
                                    ibatch);
            }
        }
    }
    else {
        queue.fork();
        for (size_t i = 0; i < batch; ++i) {
            Op transA_    = blas::batch::extract<Op>(transA, i);
            Op transB_    = blas::batch::extract<Op>(transB, i);
            int64_t m_    = blas::batch::extract<int64_t>(m, i);
            int64_t n_    = blas::batch::extract<int64_t>(n, i);
            int64_t k_    = blas::batch::extract<int64_t>(k, i);
            int64_t lda_  = blas::batch::extract<int64_t>(ldda, i);
            int64_t ldb_  = blas::batch::extract<int64_t>(lddb, i);
            int64_t ldc_  = blas::batch::extract<int64_t>(lddc, i);
            double alpha_ = blas::batch::extract<double>(alpha, i);
            double beta_  = blas::batch::extract<double>(beta, i);
            double* dA_   = blas::batch::extract<double*>(Aarray, i);
            double* dB_   = blas::batch::extract<double*>(Barray, i);
            double* dC_   = blas::batch::extract<double*>(Carray, i);
            blas::gemm(
                layout, transA_, transB_, m_, n_, k_,
                alpha_, dA_, lda_,
                        dB_, ldb_,
                beta_,  dC_, ldc_, queue );
            queue.revolve();
        }
        queue.join();
    }
}

// -----------------------------------------------------------------------------
/// @ingroup gemm
void blas::batch::gemm(
    blas::Layout                 layout,
    std::vector<blas::Op> const &transA,
    std::vector<blas::Op> const &transB,
    std::vector<int64_t>  const &m,
    std::vector<int64_t>  const &n,
    std::vector<int64_t>  const &k,
    std::vector< std::complex<float>  >   const &alpha,
    std::vector< std::complex<float>* >   const &Aarray, std::vector<int64_t> const &ldda,
    std::vector< std::complex<float>* >   const &Barray, std::vector<int64_t> const &lddb,
    std::vector< std::complex<float>  >   const &beta,
    std::vector< std::complex<float>* >   const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                                  std::vector<int64_t>  &info,
    blas::Queue &queue )
{
    blas_error_if( layout != Layout::ColMajor && layout != Layout::RowMajor );
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if (info.size() > 0) {
        // perform error checking
        blas::batch::gemm_check< std::complex<float> >( layout, transA, transB,
                                        m, n, k,
                                        alpha, Aarray, ldda,
                                               Barray, lddb,
                                        beta,  Carray, lddc,
                                        batch, info );
    }

    bool fixed_size =   ( transA.size() == 1     &&
                          transB.size() == 1     &&
                          m.size()      == 1     &&
                          n.size()      == 1     &&
                          k.size()      == 1     &&
                          alpha.size()  == 1     &&
                          Aarray.size() == batch &&
                          ldda.size()   == 1     &&
                          Barray.size() == batch &&
                          lddb.size()   == 1     &&
                          beta.size()   == 1     &&
                          Carray.size() == batch &&
                          lddc.size()   == 1 );
    blas::set_device( queue.device() );

    if (fixed_size) {
        // call the vendor routine
        device_trans_t  transA_ = blas::device_trans_const( transA[0] );
        device_trans_t  transB_ = blas::device_trans_const( transB[0] );
        device_blas_int m_      = (device_blas_int) m[0];
        device_blas_int n_      = (device_blas_int) n[0];
        device_blas_int k_      = (device_blas_int) k[0];
        device_blas_int ldda_   = (device_blas_int) ldda[0];
        device_blas_int lddb_   = (device_blas_int) lddb[0];
        device_blas_int lddc_   = (device_blas_int) lddc[0];

        size_t batch_limit = queue.get_batch_limit();
        std::complex<float> **dAarray, **dBarray, **dCarray;
        dAarray = (std::complex<float> **)queue.get_devPtrArray();
        dBarray = dAarray + batch_limit;
        dCarray = dBarray + batch_limit;

        for( size_t ib = 0; ib < batch; ib += batch_limit ) { 
            size_t ibatch = std::min( batch_limit, batch-ib );

            // copy Aarray, Barray, and Carray to device
            device_setvector< std::complex<float> *>(ibatch, (std::complex<float>**)&Aarray[ib], 1, dAarray, 1, queue);
            device_setvector< std::complex<float> *>(ibatch, (std::complex<float>**)&Barray[ib], 1, dBarray, 1, queue);
            device_setvector< std::complex<float> *>(ibatch, (std::complex<float>**)&Carray[ib], 1, dCarray, 1, queue);

            if (layout == Layout::RowMajor) {
                // swap transA <=> transB, m <=> n, B <=> A
                DEVICE_BATCH_cgemm( queue.handle(),
                                    transB_, transA_,
                                    n_, m_, k_,
                                    alpha[0], dBarray, lddb_, dAarray, ldda_,
                                    beta[0],  dCarray, lddc_,
                                    ibatch);
            }
            else {
                DEVICE_BATCH_cgemm( queue.handle(),
                                    transA_, transB_,
                                    m_, n_, k_,
                                    alpha[0], dAarray, ldda_, dBarray, lddb_,
                                    beta[0],  dCarray, lddc_,
                                    ibatch);
            }
        }
    }
    else {
        queue.fork();
        for (size_t i = 0; i < batch; ++i) {
            Op transA_    = blas::batch::extract<Op>(transA, i);
            Op transB_    = blas::batch::extract<Op>(transB, i);
            int64_t m_    = blas::batch::extract<int64_t>(m, i);
            int64_t n_    = blas::batch::extract<int64_t>(n, i);
            int64_t k_    = blas::batch::extract<int64_t>(k, i);
            int64_t lda_  = blas::batch::extract<int64_t>(ldda, i);
            int64_t ldb_  = blas::batch::extract<int64_t>(lddb, i);
            int64_t ldc_  = blas::batch::extract<int64_t>(lddc, i);
            std::complex<float> alpha_ = blas::batch::extract<std::complex<float> >(alpha, i);
            std::complex<float> beta_  = blas::batch::extract<std::complex<float> >(beta, i);
            std::complex<float>* dA_   = blas::batch::extract<std::complex<float>*>(Aarray, i);
            std::complex<float>* dB_   = blas::batch::extract<std::complex<float>*>(Barray, i);
            std::complex<float>* dC_   = blas::batch::extract<std::complex<float>*>(Carray, i);
            blas::gemm(
                layout, transA_, transB_, m_, n_, k_,
                alpha_, dA_, lda_,
                        dB_, ldb_,
                beta_,  dC_, ldc_, queue );
            queue.revolve();
        }
        queue.join();
    }
}

// -----------------------------------------------------------------------------
/// @ingroup gemm
void blas::batch::gemm(
    blas::Layout                 layout,
    std::vector<blas::Op> const &transA,
    std::vector<blas::Op> const &transB,
    std::vector<int64_t>  const &m,
    std::vector<int64_t>  const &n,
    std::vector<int64_t>  const &k,
    std::vector< std::complex<double>  >   const &alpha,
    std::vector< std::complex<double>* >   const &Aarray, std::vector<int64_t> const &ldda,
    std::vector< std::complex<double>* >   const &Barray, std::vector<int64_t> const &lddb,
    std::vector< std::complex<double>  >   const &beta,
    std::vector< std::complex<double>* >   const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                                   std::vector<int64_t>       &info,
    blas::Queue &queue )
{
    blas_error_if( layout != Layout::ColMajor && layout != Layout::RowMajor );
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if (info.size() > 0) {
        // perform error checking
        blas::batch::gemm_check< std::complex<double> >( layout, transA, transB,
                                        m, n, k,
                                        alpha, Aarray, ldda,
                                               Barray, lddb,
                                        beta,  Carray, lddc,
                                        batch, info );
    }

    bool fixed_size =   ( transA.size() == 1     &&
                          transB.size() == 1     &&
                          m.size()      == 1     &&
                          n.size()      == 1     &&
                          k.size()      == 1     &&
                          alpha.size()  == 1     &&
                          Aarray.size() == batch &&
                          ldda.size()   == 1     &&
                          Barray.size() == batch &&
                          lddb.size()   == 1     &&
                          beta.size()   == 1     &&
                          Carray.size() == batch &&
                          lddc.size()   == 1 );
    blas::set_device( queue.device() );

    if (fixed_size) {
        // call the vendor routine
        device_trans_t  transA_ = blas::device_trans_const( transA[0] );
        device_trans_t  transB_ = blas::device_trans_const( transB[0] );
        device_blas_int m_      = (device_blas_int) m[0];
        device_blas_int n_      = (device_blas_int) n[0];
        device_blas_int k_      = (device_blas_int) k[0];
        device_blas_int ldda_   = (device_blas_int) ldda[0];
        device_blas_int lddb_   = (device_blas_int) lddb[0];
        device_blas_int lddc_   = (device_blas_int) lddc[0];

        size_t batch_limit = queue.get_batch_limit();
        std::complex<double> **dAarray, **dBarray, **dCarray;
        dAarray = (std::complex<double> **)queue.get_devPtrArray();
        dBarray = dAarray + batch_limit;
        dCarray = dBarray + batch_limit;

        for( size_t ib = 0; ib < batch; ib += batch_limit ) { 
            size_t ibatch = std::min( batch_limit, batch-ib );

            // copy Aarray, Barray, and Carray to device
            device_setvector< std::complex<double> *>(ibatch, (std::complex<double>**)&Aarray[ib], 1, dAarray, 1, queue);
            device_setvector< std::complex<double> *>(ibatch, (std::complex<double>**)&Barray[ib], 1, dBarray, 1, queue);
            device_setvector< std::complex<double> *>(ibatch, (std::complex<double>**)&Carray[ib], 1, dCarray, 1, queue);

            if (layout == Layout::RowMajor) {
                // swap transA <=> transB, m <=> n, B <=> A
                DEVICE_BATCH_zgemm( queue.handle(),
                                    transB_, transA_,
                                    n_, m_, k_,
                                    alpha[0], dBarray, lddb_, dAarray, ldda_,
                                    beta[0],  dCarray, lddc_,
                                    ibatch);
            }
            else {
                DEVICE_BATCH_zgemm( queue.handle(),
                                    transA_, transB_,
                                    m_, n_, k_,
                                    alpha[0], dAarray, ldda_, dBarray, lddb_,
                                    beta[0],  dCarray, lddc_,
                                    ibatch);
            }
        }
    }
    else {
        queue.fork();
        for (size_t i = 0; i < batch; ++i) {
            Op transA_    = blas::batch::extract<Op>(transA, i);
            Op transB_    = blas::batch::extract<Op>(transB, i);
            int64_t m_    = blas::batch::extract<int64_t>(m, i);
            int64_t n_    = blas::batch::extract<int64_t>(n, i);
            int64_t k_    = blas::batch::extract<int64_t>(k, i);
            int64_t lda_  = blas::batch::extract<int64_t>(ldda, i);
            int64_t ldb_  = blas::batch::extract<int64_t>(lddb, i);
            int64_t ldc_  = blas::batch::extract<int64_t>(lddc, i);
            std::complex<double> alpha_ = blas::batch::extract<std::complex<double> >(alpha, i);
            std::complex<double> beta_  = blas::batch::extract<std::complex<double> >(beta, i);
            std::complex<double>* dA_   = blas::batch::extract<std::complex<double>*>(Aarray, i);
            std::complex<double>* dB_   = blas::batch::extract<std::complex<double>*>(Barray, i);
            std::complex<double>* dC_   = blas::batch::extract<std::complex<double>*>(Carray, i);
            blas::gemm(
                layout, transA_, transB_, m_, n_, k_,
                alpha_, dA_, lda_,
                        dB_, ldb_,
                beta_,  dC_, ldc_, queue );
            queue.revolve();
        }
        queue.join();
    }
}

// -----------------------------------------------------------------------------
/// @ingroup gemm
//  an API for group of fixed size (in fact, fixed params)
void blas::batch::gemm(
    blas::Layout                 layout,
    std::vector<blas::Op> const &transA,
    std::vector<blas::Op> const &transB,
    std::vector<int64_t>  const &m,
    std::vector<int64_t>  const &n,
    std::vector<int64_t>  const &k,
    std::vector<float >   const &alpha,
    std::vector<float*>   const &Aarray,     std::vector<int64_t> const &ldda,
    std::vector<float*>   const &Barray,     std::vector<int64_t> const &lddb,
    std::vector<float >   const &beta,
    std::vector<float*>   const &Carray,     std::vector<int64_t> const &lddc,
    std::vector<size_t>   const &group_size, std::vector<int64_t>       &info,
    blas::Queue &queue )
{
    size_t batch_size = 0;
    size_t group_count = group_size.size(); 
    if( group_count == 0 ) return;

    blas_error_if( layout      != Layout::ColMajor && layout      != Layout::RowMajor );
    blas_error_if( info.size() != 0                && info.size() != group_count );

    for(size_t ig = 0; ig < group_count; ig++) {
        batch_size += group_size[ ig ];
    }

    blas_error_if( transA.size() !=  group_count );
    blas_error_if( transB.size() !=  group_count );
    blas_error_if( m.size()      !=  group_count );
    blas_error_if( n.size()      !=  group_count );
    blas_error_if( k.size()      !=  group_count );
    blas_error_if( alpha.size()  !=  group_count );
    blas_error_if( ldda.size()   !=  group_count );
    blas_error_if( lddb.size()   !=  group_count );
    blas_error_if( beta.size()   !=  group_count );
    blas_error_if( lddc.size()   !=  group_count );

    blas_error_if( Aarray.size() !=  batch_size );
    blas_error_if( Barray.size() !=  batch_size );
    blas_error_if( Carray.size() !=  batch_size );

    // assume at least one operation per group
    blas_error_if( batch_size < group_count );

    if (info.size() > 0) {
        // perform error checking
        blas::batch::gemm_check<float>( layout, transA, transB,
                                        m, n, k,
                                        alpha, Aarray, ldda,
                                               Barray, lddb,
                                        beta,  Carray, lddc,
                                        group_count, info );
    }

    // set device
    blas::set_device( queue.device() );

    float **dAarray, **dBarray, **dCarray;
    size_t batch_limit = queue.get_batch_limit();
    size_t processed = 0;

    // if we have one group only, no need to fork
    if( group_count > 1 ) queue.fork();

    for(size_t ig = 0; ig < group_count; ig++) {
        // extract params for the current group
        size_t          batch   = group_size[ ig ];
        device_trans_t  transA_ = blas::device_trans_const( transA[ ig ] );
        device_trans_t  transB_ = blas::device_trans_const( transB[ ig ] );
        device_blas_int m_      = (device_blas_int) m[ ig ];
        device_blas_int n_      = (device_blas_int) n[ ig ];
        device_blas_int k_      = (device_blas_int) k[ ig ];
        device_blas_int ldda_   = (device_blas_int) ldda[ ig ];
        device_blas_int lddb_   = (device_blas_int) lddb[ ig ];
        device_blas_int lddc_   = (device_blas_int) lddc[ ig ];

        // each group is submitted to a different stream using strides of batch_limit
        // first, get the device pointer array for the current stream
        dAarray = (float**)queue.get_devPtrArray();
        dBarray = dAarray + batch_limit;
        dCarray = dBarray + batch_limit;
        for( size_t ib = 0; ib < batch; ib += batch_limit ) { 
            size_t ibatch = std::min( batch_limit, batch-ib );

            // copy Aarray, Barray, and Carray to device
            device_setvector<float*>(ibatch, (float**)&Aarray[ processed+ib ], 1, dAarray, 1, queue);
            device_setvector<float*>(ibatch, (float**)&Barray[ processed+ib ], 1, dBarray, 1, queue);
            device_setvector<float*>(ibatch, (float**)&Carray[ processed+ib ], 1, dCarray, 1, queue);

            if (layout == Layout::RowMajor) {
                // swap transA <=> transB, m <=> n, B <=> A
                DEVICE_BATCH_sgemm( queue.handle(),
                                    transB_, transA_,
                                    n_, m_, k_,
                                    alpha[ig], dBarray, lddb_, dAarray, ldda_,
                                    beta[ig],  dCarray, lddc_,
                                    ibatch);
            }
            else {
                DEVICE_BATCH_sgemm( queue.handle(),
                                    transA_, transB_,
                                    m_, n_, k_,
                                    alpha[ig], dAarray, ldda_, dBarray, lddb_,
                                    beta[ig],  dCarray, lddc_,
                                    ibatch);
            }
        }

        processed += batch; 
        if( group_count > 1 ) queue.revolve();
    }

    if( group_count > 1 ) queue.join();
}

// -----------------------------------------------------------------------------
/// @ingroup gemm
//  an API for group of fixed size (in fact, fixed params)
void blas::batch::gemm(
    blas::Layout                 layout,
    std::vector<blas::Op> const &transA,
    std::vector<blas::Op> const &transB,
    std::vector<int64_t>  const &m,
    std::vector<int64_t>  const &n,
    std::vector<int64_t>  const &k,
    std::vector<double >  const &alpha,
    std::vector<double*>  const &Aarray,     std::vector<int64_t> const &ldda,
    std::vector<double*>  const &Barray,     std::vector<int64_t> const &lddb,
    std::vector<double >  const &beta,
    std::vector<double*>  const &Carray,     std::vector<int64_t> const &lddc,
    std::vector<size_t>   const &group_size, std::vector<int64_t>       &info,
    blas::Queue &queue )
{
    size_t batch_size = 0;
    size_t group_count = group_size.size(); 
    if( group_count == 0 ) return;

    blas_error_if( layout      != Layout::ColMajor && layout      != Layout::RowMajor );
    blas_error_if( info.size() != 0                && info.size() != group_count );

    for(size_t ig = 0; ig < group_count; ig++) {
        batch_size += group_size[ ig ];
    }

    blas_error_if( transA.size() !=  group_count );
    blas_error_if( transB.size() !=  group_count );
    blas_error_if( m.size()      !=  group_count );
    blas_error_if( n.size()      !=  group_count );
    blas_error_if( k.size()      !=  group_count );
    blas_error_if( alpha.size()  !=  group_count );
    blas_error_if( ldda.size()   !=  group_count );
    blas_error_if( lddb.size()   !=  group_count );
    blas_error_if( beta.size()   !=  group_count );
    blas_error_if( lddc.size()   !=  group_count );

    blas_error_if( Aarray.size() !=  batch_size );
    blas_error_if( Barray.size() !=  batch_size );
    blas_error_if( Carray.size() !=  batch_size );

    // assume at least one operation per group
    blas_error_if( batch_size < group_count );

    if (info.size() > 0) {
        // perform error checking
        blas::batch::gemm_check<double>( layout, transA, transB,
                                         m, n, k,
                                         alpha, Aarray, ldda,
                                                Barray, lddb,
                                         beta,  Carray, lddc,
                                         group_count, info );
    }

    // set device
    blas::set_device( queue.device() );

    double **dAarray, **dBarray, **dCarray;
    size_t batch_limit = queue.get_batch_limit();
    size_t processed = 0;

    // if we have one group only, no need to fork
    if( group_count > 1 ) queue.fork();

    for(size_t ig = 0; ig < group_count; ig++) {
        // extract params for the current group
        size_t          batch   = group_size[ ig ];
        device_trans_t  transA_ = blas::device_trans_const( transA[ ig ] );
        device_trans_t  transB_ = blas::device_trans_const( transB[ ig ] );
        device_blas_int m_      = (device_blas_int) m[ ig ];
        device_blas_int n_      = (device_blas_int) n[ ig ];
        device_blas_int k_      = (device_blas_int) k[ ig ];
        device_blas_int ldda_   = (device_blas_int) ldda[ ig ];
        device_blas_int lddb_   = (device_blas_int) lddb[ ig ];
        device_blas_int lddc_   = (device_blas_int) lddc[ ig ];

        // each group is submitted to a different stream using strides of batch_limit
        // first, get the device pointer array for the current stream
        dAarray = (double**)queue.get_devPtrArray();
        dBarray = dAarray + batch_limit;
        dCarray = dBarray + batch_limit;
        for( size_t ib = 0; ib < batch; ib += batch_limit ) { 
            size_t ibatch = std::min( batch_limit, batch-ib );

            // copy Aarray, Barray, and Carray to device
            device_setvector<double*>(ibatch, (double**)&Aarray[ processed+ib ], 1, dAarray, 1, queue);
            device_setvector<double*>(ibatch, (double**)&Barray[ processed+ib ], 1, dBarray, 1, queue);
            device_setvector<double*>(ibatch, (double**)&Carray[ processed+ib ], 1, dCarray, 1, queue);

            if (layout == Layout::RowMajor) {
                // swap transA <=> transB, m <=> n, B <=> A
                DEVICE_BATCH_dgemm( queue.handle(),
                                    transB_, transA_,
                                    n_, m_, k_,
                                    alpha[ig], dBarray, lddb_, dAarray, ldda_,
                                    beta[ig],  dCarray, lddc_,
                                    ibatch);
            }
            else {
                DEVICE_BATCH_dgemm( queue.handle(),
                                    transA_, transB_,
                                    m_, n_, k_,
                                    alpha[ig], dAarray, ldda_, dBarray, lddb_,
                                    beta[ig],  dCarray, lddc_,
                                    ibatch);
            }
        }

        processed += batch; 
        if( group_count > 1 ) queue.revolve();
    }

    if( group_count > 1 ) queue.join();
}

// -----------------------------------------------------------------------------
/// @ingroup gemm
//  an API for group of fixed size (in fact, fixed params)
void blas::batch::gemm(
    blas::Layout                              layout,
    std::vector<blas::Op>              const &transA,
    std::vector<blas::Op>              const &transB,
    std::vector<int64_t>               const &m,
    std::vector<int64_t>               const &n,
    std::vector<int64_t>               const &k,
    std::vector<std::complex<float> >  const &alpha,
    std::vector<std::complex<float>*>  const &Aarray,     std::vector<int64_t> const &ldda,
    std::vector<std::complex<float>*>  const &Barray,     std::vector<int64_t> const &lddb,
    std::vector<std::complex<float> >  const &beta,
    std::vector<std::complex<float>*>  const &Carray,     std::vector<int64_t> const &lddc,
    std::vector<size_t>   const &group_size, std::vector<int64_t>       &info,
    blas::Queue &queue )
{
    size_t batch_size = 0;
    size_t group_count = group_size.size(); 
    if( group_count == 0 ) return;

    blas_error_if( layout      != Layout::ColMajor && layout      != Layout::RowMajor );
    blas_error_if( info.size() != 0                && info.size() != group_count );

    for(size_t ig = 0; ig < group_count; ig++) {
        batch_size += group_size[ ig ];
    }

    blas_error_if( transA.size() !=  group_count );
    blas_error_if( transB.size() !=  group_count );
    blas_error_if( m.size()      !=  group_count );
    blas_error_if( n.size()      !=  group_count );
    blas_error_if( k.size()      !=  group_count );
    blas_error_if( alpha.size()  !=  group_count );
    blas_error_if( ldda.size()   !=  group_count );
    blas_error_if( lddb.size()   !=  group_count );
    blas_error_if( beta.size()   !=  group_count );
    blas_error_if( lddc.size()   !=  group_count );

    blas_error_if( Aarray.size() !=  batch_size );
    blas_error_if( Barray.size() !=  batch_size );
    blas_error_if( Carray.size() !=  batch_size );

    // assume at least one operation per group
    blas_error_if( batch_size < group_count );

    if (info.size() > 0) {
        // perform error checking
        blas::batch::gemm_check<std::complex<float>>( layout, transA, transB,
                                         m, n, k,
                                         alpha, Aarray, ldda,
                                                Barray, lddb,
                                         beta,  Carray, lddc,
                                         group_count, info );
    }

    // set device
    blas::set_device( queue.device() );

    std::complex<float> **dAarray, **dBarray, **dCarray;
    size_t batch_limit = queue.get_batch_limit();
    size_t processed = 0;

    // if we have one group only, no need to fork
    if( group_count > 1 ) queue.fork();

    for(size_t ig = 0; ig < group_count; ig++) {
        // extract params for the current group
        size_t          batch   = group_size[ ig ];
        device_trans_t  transA_ = blas::device_trans_const( transA[ ig ] );
        device_trans_t  transB_ = blas::device_trans_const( transB[ ig ] );
        device_blas_int m_      = (device_blas_int) m[ ig ];
        device_blas_int n_      = (device_blas_int) n[ ig ];
        device_blas_int k_      = (device_blas_int) k[ ig ];
        device_blas_int ldda_   = (device_blas_int) ldda[ ig ];
        device_blas_int lddb_   = (device_blas_int) lddb[ ig ];
        device_blas_int lddc_   = (device_blas_int) lddc[ ig ];

        // each group is submitted to a different stream using strides of batch_limit
        // first, get the device pointer array for the current stream
        dAarray = (std::complex<float>**)queue.get_devPtrArray();
        dBarray = dAarray + batch_limit;
        dCarray = dBarray + batch_limit;
        for( size_t ib = 0; ib < batch; ib += batch_limit ) { 
            size_t ibatch = std::min( batch_limit, batch-ib );

            // copy Aarray, Barray, and Carray to device
            device_setvector<std::complex<float>*>(ibatch, (std::complex<float>**)&Aarray[ processed+ib ], 1, dAarray, 1, queue);
            device_setvector<std::complex<float>*>(ibatch, (std::complex<float>**)&Barray[ processed+ib ], 1, dBarray, 1, queue);
            device_setvector<std::complex<float>*>(ibatch, (std::complex<float>**)&Carray[ processed+ib ], 1, dCarray, 1, queue);

            if (layout == Layout::RowMajor) {
                // swap transA <=> transB, m <=> n, B <=> A
                DEVICE_BATCH_cgemm( queue.handle(),
                                    transB_, transA_,
                                    n_, m_, k_,
                                    alpha[ig], dBarray, lddb_, dAarray, ldda_,
                                    beta[ig],  dCarray, lddc_,
                                    ibatch);
            }
            else {
                DEVICE_BATCH_cgemm( queue.handle(),
                                    transA_, transB_,
                                    m_, n_, k_,
                                    alpha[ig], dAarray, ldda_, dBarray, lddb_,
                                    beta[ig],  dCarray, lddc_,
                                    ibatch);
            }
        }

        processed += batch; 
        if( group_count > 1 ) queue.revolve();
    }

    if( group_count > 1 ) queue.join();
}

// -----------------------------------------------------------------------------
/// @ingroup gemm
//  an API for group of fixed size (in fact, fixed params)
void blas::batch::gemm(
    blas::Layout                              layout,
    std::vector<blas::Op>              const &transA,
    std::vector<blas::Op>              const &transB,
    std::vector<int64_t>               const &m,
    std::vector<int64_t>               const &n,
    std::vector<int64_t>               const &k,
    std::vector<std::complex<double> >  const &alpha,
    std::vector<std::complex<double>*>  const &Aarray,     std::vector<int64_t> const &ldda,
    std::vector<std::complex<double>*>  const &Barray,     std::vector<int64_t> const &lddb,
    std::vector<std::complex<double> >  const &beta,
    std::vector<std::complex<double>*>  const &Carray,     std::vector<int64_t> const &lddc,
    std::vector<size_t>   const &group_size, std::vector<int64_t>       &info,
    blas::Queue &queue )
{
    size_t batch_size = 0;
    size_t group_count = group_size.size(); 
    if( group_count == 0 ) return;

    blas_error_if( layout      != Layout::ColMajor && layout      != Layout::RowMajor );
    blas_error_if( info.size() != 0                && info.size() != group_count );

    for(size_t ig = 0; ig < group_count; ig++) {
        batch_size += group_size[ ig ];
    }

    blas_error_if( transA.size() !=  group_count );
    blas_error_if( transB.size() !=  group_count );
    blas_error_if( m.size()      !=  group_count );
    blas_error_if( n.size()      !=  group_count );
    blas_error_if( k.size()      !=  group_count );
    blas_error_if( alpha.size()  !=  group_count );
    blas_error_if( ldda.size()   !=  group_count );
    blas_error_if( lddb.size()   !=  group_count );
    blas_error_if( beta.size()   !=  group_count );
    blas_error_if( lddc.size()   !=  group_count );

    blas_error_if( Aarray.size() !=  batch_size );
    blas_error_if( Barray.size() !=  batch_size );
    blas_error_if( Carray.size() !=  batch_size );

    // assume at least one operation per group
    blas_error_if( batch_size < group_count );

    if (info.size() > 0) {
        // perform error checking
        blas::batch::gemm_check<std::complex<double>>( layout, transA, transB,
                                         m, n, k,
                                         alpha, Aarray, ldda,
                                                Barray, lddb,
                                         beta,  Carray, lddc,
                                         group_count, info );
    }

    // set device
    blas::set_device( queue.device() );

    std::complex<double> **dAarray, **dBarray, **dCarray;
    size_t batch_limit = queue.get_batch_limit();
    size_t processed = 0;

    // if we have one group only, no need to fork
    if( group_count > 1 ) queue.fork();

    for(size_t ig = 0; ig < group_count; ig++) {
        // extract params for the current group
        size_t          batch   = group_size[ ig ];
        device_trans_t  transA_ = blas::device_trans_const( transA[ ig ] );
        device_trans_t  transB_ = blas::device_trans_const( transB[ ig ] );
        device_blas_int m_      = (device_blas_int) m[ ig ];
        device_blas_int n_      = (device_blas_int) n[ ig ];
        device_blas_int k_      = (device_blas_int) k[ ig ];
        device_blas_int ldda_   = (device_blas_int) ldda[ ig ];
        device_blas_int lddb_   = (device_blas_int) lddb[ ig ];
        device_blas_int lddc_   = (device_blas_int) lddc[ ig ];

        // each group is submitted to a different stream using strides of batch_limit
        // first, get the device pointer array for the current stream
        dAarray = (std::complex<double>**)queue.get_devPtrArray();
        dBarray = dAarray + batch_limit;
        dCarray = dBarray + batch_limit;
        for( size_t ib = 0; ib < batch; ib += batch_limit ) { 
            size_t ibatch = std::min( batch_limit, batch-ib );

            // copy Aarray, Barray, and Carray to device
            device_setvector<std::complex<double>*>(ibatch, (std::complex<double>**)&Aarray[ processed+ib ], 1, dAarray, 1, queue);
            device_setvector<std::complex<double>*>(ibatch, (std::complex<double>**)&Barray[ processed+ib ], 1, dBarray, 1, queue);
            device_setvector<std::complex<double>*>(ibatch, (std::complex<double>**)&Carray[ processed+ib ], 1, dCarray, 1, queue);

            if (layout == Layout::RowMajor) {
                // swap transA <=> transB, m <=> n, B <=> A
                DEVICE_BATCH_zgemm( queue.handle(),
                                    transB_, transA_,
                                    n_, m_, k_,
                                    alpha[ig], dBarray, lddb_, dAarray, ldda_,
                                    beta[ig],  dCarray, lddc_,
                                    ibatch);
            }
            else {
                DEVICE_BATCH_zgemm( queue.handle(),
                                    transA_, transB_,
                                    m_, n_, k_,
                                    alpha[ig], dAarray, ldda_, dBarray, lddb_,
                                    beta[ig],  dCarray, lddc_,
                                    ibatch);
            }
        }

        processed += batch; 
        if( group_count > 1 ) queue.revolve();
    }

    if( group_count > 1 ) queue.join();
}
