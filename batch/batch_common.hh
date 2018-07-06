#ifndef BATCH_COMMON_HH
#define BATCH_COMMON_HH

#include "blas_util.hh"
#include <vector>

namespace blas{

namespace batch{

template<typename T>
T extract(std::vector<T> const &ivector, const int64_t index){
    return (ivector.size() == 1) ? ivector[0] : ivector[index];
}

// -----------------------------------------------------------------------------
// batch gemm check
template<typename T>
void gemm_check(
        std::vector<blas::Op> const &transA, 
        std::vector<blas::Op> const &transB, 
        std::vector<int64_t>  const &m, 
        std::vector<int64_t>  const &n, 
        std::vector<int64_t>  const &k, 
        std::vector<T >       const &alpha, 
        std::vector<T*>       const &A, std::vector<int64_t> const &lda, 
        std::vector<T*>       const &B, std::vector<int64_t> const &ldb, 
        std::vector<T >       const &beta, 
        std::vector<T*>       const &C, std::vector<int64_t> const &ldc, 
        const size_t batchCount, std::vector<int64_t> &info)
{
    // size error checking
    blas_error_if( (transA.size() != 1 && transA.size() != batchCount) );
    blas_error_if( (transB.size() != 1 && transB.size() != batchCount) );

    blas_error_if( (m.size() != 1 && m.size() != batchCount) );
    blas_error_if( (n.size() != 1 && n.size() != batchCount) );
    blas_error_if( (k.size() != 1 && k.size() != batchCount) );

    blas_error_if( (alpha.size() != 1 && alpha.size() != batchCount) );
    blas_error_if( (beta.size()  != 1 && beta.size()  != batchCount) );

    blas_error_if( (lda.size() != 1 && lda.size() != batchCount) );
    blas_error_if( (ldb.size() != 1 && ldb.size() != batchCount) );
    blas_error_if( (ldc.size() != 1 && ldc.size() != batchCount) );

    blas_error_if( (A.size() != 1 && A.size() != batchCount) );
    blas_error_if( (B.size() != 1 && B.size() != batchCount) );
    blas_error_if( (C.size() != batchCount) );

    blas_error_if( A.size() == 1 && (m.size() > 1 || k.size() > 1 || lda.size() > 1) );
    blas_error_if( B.size() == 1 && (k.size() > 1 || n.size() > 1 || ldb.size() > 1) );
    blas_error_if( C.size() == 1 && 
               (transA.size() > 1 || transB.size() > 1 || 
                m.size()      > 1 || n.size()      > 1 || k.size()   > 1 ||
                alpha.size()  > 1 || beta.size()   > 1 || 
                lda.size()    > 1 || ldb.size()    > 1 || ldc.size() > 1 || 
                A.size()      > 1 || B.size()      > 1      
                ) 
             );

    if(info.size() == 1){
        /* argument based error reporting */
        int64_t linfo;
        
        // transA
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < transA.size(); i++){
            linfo += (transA[i] != Op::NoTrans && 
                      transA[i] != Op::Trans   && 
                      transA[i] != Op::ConjTrans
                      ) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -1 : 0;
        blas_error_if( linfo > 0 );
        
        // transB
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < transB.size(); i++){
            linfo += (transB[i] != Op::NoTrans && 
                      transB[i] != Op::Trans   && 
                      transB[i] != Op::ConjTrans
                     ) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -2 : 0;
        blas_error_if( linfo > 0 );
        
        // m
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < m.size(); i++){
            linfo += (m[i] < 0) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -3 : 0;
        blas_error_if( linfo > 0 );
        
        // n
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < n.size(); i++){
            linfo += (n[i] < 0) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -4 : 0;
        blas_error_if( linfo > 0 );
        
        // k
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < k.size(); i++){
            linfo += (k[i] < 0) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -5 : 0;
        blas_error_if( linfo > 0 );

        // lda
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < A.size(); i++){
            Op trans_  = extract<Op>(transA, i);
            int64_t nrowA_ = (trans_ == Op::NoTrans) ? 
                             extract<int64_t>(m, i) : extract<int64_t>(k, i);
            int64_t lda_   = extract<int64_t>(lda, i);
            linfo += (lda_ < nrowA_) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -8 : 0;
        blas_error_if( linfo > 0 );

        // ldb
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < B.size(); i++){
            Op trans_  = extract<Op>(transB, i);
            int64_t nrowB_ = (trans_ == Op::NoTrans) ? 
                             extract<int64_t>(k, i) : extract<int64_t>(n, i);
            int64_t ldb_   = extract<int64_t>(ldb, i);
            linfo += (ldb_ < nrowB_) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -10 : 0;
        blas_error_if( linfo > 0 );
        
        // ldc
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < C.size(); i++){
            int64_t m_   = extract<int64_t>(m, i);
            int64_t ldc_ = extract<int64_t>(ldc, i);
            linfo += (ldc_ < m_) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -13 : 0;
        blas_error_if( linfo > 0 );
    }
    else{
        /* problem based eror reporting */
        #pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < batchCount; i++){
            Op transA_ = extract<Op>(transA, i);
            Op transB_ = extract<Op>(transB, i);
        
            int64_t m_ = extract<int64_t>(m, i); 
            int64_t n_ = extract<int64_t>(n, i);
            int64_t k_ = extract<int64_t>(k, i);

            int64_t lda_ = extract<int64_t>(lda, i);
            int64_t ldb_ = extract<int64_t>(ldb, i);
            int64_t ldc_ = extract<int64_t>(ldc, i);
            
            int64_t nrowA_ = (transA_ == Op::NoTrans) ? m_ : k_;
            int64_t nrowB_ = (transB_ == Op::NoTrans) ? k_ : n_;
            
            info[i] = 0;
            if(transA_ != Op::NoTrans && 
               transA_ != Op::Trans   && 
               transA_ != Op::ConjTrans) {
                info[i] = -1;
            }
            else if(transB_ != Op::NoTrans && 
                    transB_ != Op::Trans   && 
                    transB_ != Op::ConjTrans) {
                info[i] = -2;
            }
            else if(m_ < 0) info[i] = -3;
            else if(n_ < 0) info[i] = -4;
            else if(k_ < 0) info[i] = -5;
            else if(lda_ < nrowA_) info[i] = -8;
            else if(ldb_ < nrowB_) info[i] = -10;
            else if(ldc_ < m_    ) info[i] = -13;
        }
        
        int64_t info_ = 0;
        #pragma omp parallel for reduction(+:info_)
        for(size_t i = 0; i < batchCount; i++){
            info_ += info[i];
        }
        blas_error_if( info_ != 0 );
    }
}

// -----------------------------------------------------------------------------
// batch trsm check
template<typename T>
void trsm_check(
        std::vector<blas::Side> const &side, 
        std::vector<blas::Uplo> const &uplo, 
        std::vector<blas::Op>   const &trans, 
        std::vector<blas::Diag> const &diag, 
        std::vector<int64_t>    const &m, 
        std::vector<int64_t>    const &n, 
        std::vector<T>          const &alpha, 
        std::vector<T*>         const &A, std::vector<int64_t> const &lda, 
        std::vector<T*>         const &B, std::vector<int64_t> const &ldb, 
        const size_t batchCount, std::vector<int64_t> &info)
{
    // size error checking
    blas_error_if( (side.size()  != 1 && side.size()  != batchCount) );
    blas_error_if( (uplo.size()  != 1 && uplo.size()  != batchCount) );
    blas_error_if( (trans.size() != 1 && trans.size() != batchCount) );
    blas_error_if( (diag.size()  != 1 && diag.size()  != batchCount) );

    blas_error_if( (m.size() != 1 && m.size() != batchCount) );
    blas_error_if( (n.size() != 1 && n.size() != batchCount) );

    blas_error_if( (A.size() != 1 && A.size() != batchCount) );
    blas_error_if(  B.size() != batchCount );

    blas_error_if( (lda.size() != 1 && lda.size() != batchCount) );
    blas_error_if( (ldb.size() != 1 && ldb.size() != batchCount) );

    blas_error_if( (alpha.size() != 1 && alpha.size() != batchCount) );

    blas_error_if( A.size() == 1 && ( lda.size()  > 1                         || 
                                      side.size() > 1                         || 
                                     (side[0] == Side::Left  && m.size() > 1) || 
                                     (side[0] == Side::Right && n.size() > 1) ));
    blas_error_if( B.size() == 1 && ( side.size()  > 1 || uplo.size() > 1 || 
                                      trans.size() > 1 || diag.size() > 1 ||
                                      m.size()     > 1 || n.size()    > 1 || 
                                      alpha.size() > 1 || A.size()    > 1 || 
                                      lda.size()   > 1 || ldb.size()  > 1 )); 
    
    if(info.size() == 1){
        /* argument based error reporting */
        int64_t linfo;
        
        // side
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < side.size(); i++){
            linfo += (side[i] != Side::Left  && 
                      side[i] != Side::Right 
                      ) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -1 : 0;
        blas_error_if( linfo > 0 );
        
        // uplo
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < uplo.size(); i++){
            linfo += (uplo[i] != Uplo::Lower  && 
                      uplo[i] != Uplo::Upper
                      ) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -2 : 0;
        blas_error_if( linfo > 0 );

        // trans
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < trans.size(); i++){
            linfo += (trans[i] != Op::NoTrans && 
                      trans[i] != Op::Trans   && 
                      trans[i] != Op::ConjTrans
                     ) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -3 : 0;
        blas_error_if( linfo > 0 );
        
        // diag
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < trans.size(); i++){
            linfo += (diag[i] != Diag::NonUnit && 
                      diag[i] != Diag::Unit  
                     ) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -4 : 0;
        blas_error_if( linfo > 0 );

        // m
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < m.size(); i++){
            linfo += (m[i] < 0) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -5 : 0;
        blas_error_if( linfo > 0 );

        // n
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < n.size(); i++){
            linfo += (n[i] < 0) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -6 : 0;
        blas_error_if( linfo > 0 );
        
        // lda
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < A.size(); i++){
            Side    side_  = extract<Side>(side, i);
            int64_t lda_   = extract<int64_t>(lda, i);
            int64_t nrowA_ = (side_ == Side::Left) ? extract<int64_t>(m, i) : extract<int64_t>(n, i);
            linfo += (lda_ < nrowA_) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -9 : 0;
        blas_error_if( linfo > 0 );

        // ldb
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < B.size(); i++){
            int64_t m_   = extract<int64_t>(m, i);
            int64_t ldb_ = extract<int64_t>(ldb, i);
            linfo += (ldb_ < m_) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -11 : 0;
        blas_error_if( linfo > 0 );
    }
    else{
        /* problem based eror reporting */
        #pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < batchCount; i++){
            Side  side_ = extract<Side>(side , i);
            Uplo  uplo_ = extract<Uplo>(uplo , i);
            Op   trans_ = extract<Op  >(trans, i);
            Diag  diag_ = extract<Diag>(diag , i);

            int64_t m_ = extract<int64_t>(m, i); 
            int64_t n_ = extract<int64_t>(n, i);

            int64_t lda_ = extract<int64_t>(lda, i);
            int64_t ldb_ = extract<int64_t>(ldb, i);

            int64_t nrowA_ = (side_ == Side::Left) ? m_ : n_;

            info[i] = 0;
            if(side_ != Side::Left && side_ != Side::Right) {
                info[i] = -1;
            }
            else if(uplo_ != Uplo::Lower && uplo_ != Uplo::Upper) {
                info[i] = -2;
            }
            else if(trans_ != Op::NoTrans && trans_ != Op::Trans && trans_ != Op::ConjTrans){
                info[i] = -3;
            }
            else if( diag_ != Diag::NonUnit && diag_ != Diag::Unit){
                info[i] = -4;
            }
            else if(m_ < 0) info[i] = -5;
            else if(n_ < 0) info[i] = -6;
            else if(lda_ < nrowA_) info[i] = -9;
            else if(ldb_ < m_ ) info[i] = -11;
        }
        
        int64_t info_ = 0;
        #pragma omp parallel for reduction(+:info_)
        for(size_t i = 0; i < batchCount; i++){
            info_ += info[i];
        }
        blas_error_if( info_ != 0 );
    }
}

// -----------------------------------------------------------------------------
// batch trmm check
template<typename T>
void trmm_check(
        std::vector<blas::Side> const &side,
        std::vector<blas::Uplo> const &uplo, 
        std::vector<blas::Op>   const &trans, 
        std::vector<blas::Diag> const &diag, 
        std::vector<int64_t>    const &m, 
        std::vector<int64_t>    const &n, 
        std::vector<T>          const &alpha, 
        std::vector<T*>         const &A, std::vector<int64_t> const &lda, 
        std::vector<T*>         const &B, std::vector<int64_t> const &ldb, 
        const size_t batchCount, std::vector<int64_t> &info)
{
    // size error checking
    blas_error_if( (side.size()  != 1 && side.size()  != batchCount) );
    blas_error_if( (uplo.size()  != 1 && uplo.size()  != batchCount) );
    blas_error_if( (trans.size() != 1 && trans.size() != batchCount) );
    blas_error_if( (diag.size()  != 1 && diag.size()  != batchCount) );

    blas_error_if( (m.size() != 1 && m.size() != batchCount) );
    blas_error_if( (n.size() != 1 && n.size() != batchCount) );

    blas_error_if( (A.size() != 1 && A.size() != batchCount) );
    blas_error_if(  B.size() != batchCount );

    blas_error_if( (lda.size() != 1 && lda.size() != batchCount) );
    blas_error_if( (ldb.size() != 1 && ldb.size() != batchCount) );

    blas_error_if( (alpha.size() != 1 && alpha.size() != batchCount) );

    blas_error_if( A.size() == 1 && ( lda.size()  > 1                         || 
                                      side.size() > 1                         || 
                                     (side[0] == Side::Left  && m.size() > 1) || 
                                     (side[0] == Side::Right && n.size() > 1) ));
    blas_error_if( B.size() == 1 && ( side.size()  > 1 || uplo.size() > 1 || 
                                      trans.size() > 1 || diag.size() > 1 ||
                                      m.size()     > 1 || n.size()    > 1 || 
                                      alpha.size() > 1 || A.size()    > 1 || 
                                      lda.size()   > 1 || ldb.size()  > 1 )); 
    
    if(info.size() == 1){
        /* argument based error reporting */
        int64_t linfo;

        // side
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < side.size(); i++){
            linfo += (side[i] != Side::Left  && 
                      side[i] != Side::Right 
                      ) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -1 : 0;
        blas_error_if( linfo > 0 );

        // uplo
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < uplo.size(); i++){
            linfo += (uplo[i] != Uplo::Lower  && 
                      uplo[i] != Uplo::Upper
                      ) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -2 : 0;
        blas_error_if( linfo > 0 );

        // trans
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < trans.size(); i++){
            linfo += (trans[i] != Op::NoTrans && 
                      trans[i] != Op::Trans   && 
                      trans[i] != Op::ConjTrans
                     ) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -3 : 0;
        blas_error_if( linfo > 0 );

        // diag
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < trans.size(); i++){
            linfo += (diag[i] != Diag::NonUnit && 
                      diag[i] != Diag::Unit  
                     ) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -4 : 0;
        blas_error_if( linfo > 0 );

        // m
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < m.size(); i++){
            linfo += (m[i] < 0) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -5 : 0;
        blas_error_if( linfo > 0 );

        // n
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < n.size(); i++){
            linfo += (n[i] < 0) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -6 : 0;
        blas_error_if( linfo > 0 );

        // lda
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < A.size(); i++){
            Side side_     = extract<Side>(side, i);
            int64_t lda_   = extract<int64_t>(lda, i);
            int64_t nrowA_ = (side_ == Side::Left) ? extract<int64_t>(m, i) : extract<int64_t>(n, i);
            linfo += (lda_ < nrowA_) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -9 : 0;
        blas_error_if( linfo > 0 );

        // ldb
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < B.size(); i++){
            int64_t m_   = extract<int64_t>(m, i);
            int64_t ldb_ = extract<int64_t>(ldb, i);
            linfo += (ldb_ < m_) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -11 : 0;
        blas_error_if( linfo > 0 );
        
    }
    else{
        /* problem based eror reporting */
        #pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < batchCount; i++){
            Side  side_ = extract<Side>(side , i);
            Uplo  uplo_ = extract<Uplo>(uplo , i);
            Op   trans_ = extract<Op  >(trans, i);
            Diag  diag_ = extract<Diag>(diag , i);

            int64_t m_ = extract<int64_t>(m, i); 
            int64_t n_ = extract<int64_t>(n, i);

            int64_t lda_ = extract<int64_t>(lda, i);
            int64_t ldb_ = extract<int64_t>(ldb, i);

            int64_t nrowA_ = (side_ == Side::Left) ? m_ : n_;

            info[i] = 0;
            if(side_ != Side::Left && side_ != Side::Right) {
                info[i] = -1;
            }
            else if(uplo_ != Uplo::Lower && uplo_ != Uplo::Upper) {
                info[i] = -2;
            }
            else if(trans_ != Op::NoTrans && trans_ != Op::Trans && trans_ != Op::ConjTrans){
                info[i] = -3;
            }
            else if( diag_ != Diag::NonUnit && diag_ != Diag::Unit){
                info[i] = -4;
            }
            else if(m_ < 0) info[i] = -5;
            else if(n_ < 0) info[i] = -6;
            else if(lda_ < nrowA_) info[i] = -9;
            else if(ldb_ < m_ ) info[i] = -11;
        }

        int64_t info_ = 0;
        #pragma omp parallel for reduction(+:info_)
        for(size_t i = 0; i < batchCount; i++){
            info_ += info[i];
        }
        blas_error_if( info_ != 0 );
    }
}

// -----------------------------------------------------------------------------
// batch hemm check
template<typename T>
void hemm_check(
        std::vector<blas::Side> const &side,
        std::vector<blas::Uplo> const &uplo, 
        std::vector<int64_t>    const &m, 
        std::vector<int64_t>    const &n, 
        std::vector<T>          const &alpha, 
        std::vector<T*>         const &A, std::vector<int64_t> const &lda, 
        std::vector<T*>         const &B, std::vector<int64_t> const &ldb, 
        std::vector<T>          const &beta, 
        std::vector<T*>         const &C, std::vector<int64_t> const &ldc, 
        const size_t batchCount, std::vector<int64_t> &info)
{
    // size error checking
    blas_error_if( (side.size()  != 1 && side.size()  != batchCount) );
    blas_error_if( (uplo.size()  != 1 && uplo.size()  != batchCount) );

    blas_error_if( (m.size() != 1 && m.size() != batchCount) );
    blas_error_if( (n.size() != 1 && n.size() != batchCount) );

    blas_error_if( (A.size() != 1 && A.size() != batchCount) );
    blas_error_if( (B.size() != 1 && B.size() != batchCount) );
    blas_error_if(  C.size() != batchCount );

    blas_error_if( (lda.size() != 1 && lda.size() != batchCount) );
    blas_error_if( (ldb.size() != 1 && ldb.size() != batchCount) );
    blas_error_if( (ldc.size() != 1 && ldc.size() != batchCount) );

    blas_error_if( (alpha.size() != 1 && alpha.size() != batchCount) );
    blas_error_if( (beta.size()  != 1 && beta.size()  != batchCount) );

    blas_error_if( A.size() == 1 && 
                  (lda.size()  > 1                          || 
                   side.size() > 1                          || 
                   (side[0] == Side::Left  && m.size() > 1) || 
                   (side[0] == Side::Right && n.size() > 1) ));

    blas_error_if( B.size() == 1 && 
                  (m.size()   > 1 || 
                   n.size()   > 1 || 
                   ldb.size() > 1 ));

    blas_error_if( C.size() == 1 && 
                  (side.size()  > 1 || 
                   uplo.size()  > 1 || 
                   m.size()     > 1 || 
                   n.size()     > 1 || 
                   alpha.size() > 1 || 
                   A.size()     > 1 || 
                   lda.size()   > 1 || 
                   B.size()     > 1 ||
                   ldb.size()   > 1 || 
                   beta.size()  > 1 || 
                   ldc.size()   > 1 ));
    
    if(info.size() == 1){
        /* argument based error reporting */
        int64_t linfo;

        // side
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < side.size(); i++){
            linfo += (side[i] != Side::Left  && 
                      side[i] != Side::Right 
                      ) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -1 : 0;
        blas_error_if( linfo > 0 );

        // uplo
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < uplo.size(); i++){
            linfo += (uplo[i] != Uplo::Lower  && 
                      uplo[i] != Uplo::Upper
                      ) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -2 : 0;
        blas_error_if( linfo > 0 );

        // m
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < m.size(); i++){
            linfo += (m[i] < 0) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -3 : 0;
        blas_error_if( linfo > 0 );

        // n
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < n.size(); i++){
            linfo += (n[i] < 0) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -4 : 0;
        blas_error_if( linfo > 0 );

        // lda
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < A.size(); i++){
            Side side_     = extract<Side>(side, i);
            int64_t lda_   = extract<int64_t>(lda, i);
            int64_t nrowA_ = (side_ == Side::Left) ? extract<int64_t>(m, i) : extract<int64_t>(n, i);
            linfo += (lda_ < nrowA_) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -7 : 0;
        blas_error_if( linfo > 0 );

        // ldb
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < B.size(); i++){
            int64_t m_   = extract<int64_t>(m, i);
            int64_t ldb_ = extract<int64_t>(ldb, i);
            linfo += (ldb_ < m_) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -9 : 0;
        blas_error_if( linfo > 0 );
        
        // ldc
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < C.size(); i++){
            int64_t m_   = extract<int64_t>(m, i);
            int64_t ldc_ = extract<int64_t>(ldc, i);
            linfo += (ldc_ < m_) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -12 : 0;
        blas_error_if( linfo > 0 );
        
    }
    else{
        /* problem based eror reporting */
        #pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < batchCount; i++){
            Side  side_ = extract<Side>(side , i);
            Uplo  uplo_ = extract<Uplo>(uplo , i);

            int64_t m_ = extract<int64_t>(m, i); 
            int64_t n_ = extract<int64_t>(n, i);

            int64_t lda_ = extract<int64_t>(lda, i);
            int64_t ldb_ = extract<int64_t>(ldb, i);
            int64_t ldc_ = extract<int64_t>(ldc, i);

            int64_t nrowA_ = (side_ == Side::Left) ? m_ : n_;

            info[i] = 0;
            if(side_ != Side::Left && side_ != Side::Right) {
                info[i] = -1;
            }
            else if(uplo_ != Uplo::Lower && uplo_ != Uplo::Upper) {
                info[i] = -2;
            }
            else if(m_ < 0) info[i] = -3;
            else if(n_ < 0) info[i] = -4;
            else if(lda_ < nrowA_) info[i] = -7;
            else if(ldb_ < m_ ) info[i] = -9;
            else if(ldc_ < m_ ) info[i] = -12;
        }

        int64_t info_ = 0;
        #pragma omp parallel for reduction(+:info_)
        for(size_t i = 0; i < batchCount; i++){
            info_ += info[i];
        }
        blas_error_if( info_ != 0 );
    }
}

// -----------------------------------------------------------------------------
// batch herk check
template<typename T, typename scalarT>
void herk_check(
        std::vector<blas::Uplo> const &uplo, 
        std::vector<blas::Op>   const &trans, 
        std::vector<int64_t>    const &n, 
        std::vector<int64_t>    const &k, 
        std::vector<scalarT>    const &alpha, 
        std::vector<T*>         const &A, std::vector<int64_t> const &lda, 
        std::vector<scalarT>    const &beta, 
        std::vector<T*>         const &C, std::vector<int64_t> const &ldc, 
        const size_t batchCount, std::vector<int64_t> &info)
{
    // size error checking
    blas_error_if( (uplo.size()  != 1 && uplo.size()  != batchCount) );
    blas_error_if( (trans.size() != 1 && trans.size() != batchCount) );

    blas_error_if( (n.size() != 1 && n.size() != batchCount) );
    blas_error_if( (k.size() != 1 && k.size() != batchCount) );

    blas_error_if( (A.size() != 1 && A.size() != batchCount) );
    blas_error_if(  C.size() != batchCount );

    blas_error_if( (lda.size() != 1 && lda.size() != batchCount) );
    blas_error_if( (ldc.size() != 1 && ldc.size() != batchCount) );

    blas_error_if( (alpha.size() != 1 && alpha.size() != batchCount) );
    blas_error_if( (beta.size()  != 1 && beta.size()  != batchCount) );

    blas_error_if( A.size() == 1 && 
                  (lda.size()    > 1                  || 
                   n.size()      > 1                  ||
                   k.size()      > 1                  ||
                   (trans.size() > 1 && n[0] != k[0]) ));

    blas_error_if( C.size() == 1 && 
                  (uplo.size()  > 1 || 
                   trans.size() > 1 || 
                   n.size()     > 1 || 
                   k.size()     > 1 || 
                   alpha.size() > 1 || 
                   A.size()     > 1 || 
                   lda.size()   > 1 || 
                   beta.size()  > 1 || 
                   ldc.size()   > 1 ));

    if(info.size() == 1){
        /* argument based error reporting */
        int64_t linfo;

        // uplo
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < uplo.size(); i++){
            linfo += (uplo[i] != Uplo::Lower  && 
                      uplo[i] != Uplo::Upper
                      ) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -1 : 0;
        blas_error_if( linfo > 0 );

        // trans
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < trans.size(); i++){
            linfo += (trans[i] != Op::NoTrans  && 
                      trans[i] != Op::ConjTrans 
                      ) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -2 : 0;
        blas_error_if( linfo > 0 );

        // n
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < n.size(); i++){
            linfo += (n[i] < 0) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -3 : 0;
        blas_error_if( linfo > 0 );

        // k
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < k.size(); i++){
            linfo += (k[i] < 0) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -4 : 0;
        blas_error_if( linfo > 0 );

        // lda
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < A.size(); i++){
            Op trans_      = extract<Op>(trans, i);
            int64_t lda_   = extract<int64_t>(lda, i);
            int64_t nrowA_ = (trans_ == Op::NoTrans) ? extract<int64_t>(n, i) : extract<int64_t>(k, i);
            linfo += (lda_ < nrowA_) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -7 : 0;
        blas_error_if( linfo > 0 );

        // ldc
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < C.size(); i++){
            int64_t n_   = extract<int64_t>(n, i);
            int64_t ldc_ = extract<int64_t>(ldc, i);
            linfo += (ldc_ < n_) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -10 : 0;
        blas_error_if( linfo > 0 );
    }
    else{
        /* problem based eror reporting */
        #pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < batchCount; i++){
            Uplo  uplo_ = extract<Uplo>(uplo , i);
            Op   trans_ = extract<Op>(trans , i);

            int64_t n_ = extract<int64_t>(n, i); 
            int64_t k_ = extract<int64_t>(k, i);

            int64_t lda_ = extract<int64_t>(lda, i);
            int64_t ldc_ = extract<int64_t>(ldc, i);

            int64_t nrowA_ = (trans_ == Op::NoTrans) ? n_ : k_;

            info[i] = 0;
            if(uplo_ != Uplo::Lower && uplo_ != Uplo::Upper) {
                info[i] = -1;
            }
            else if(trans_ != Op::NoTrans && trans_ != Op::ConjTrans) {
                info[i] = -2;
            }
            else if(n_ < 0) info[i] = -3;
            else if(k_ < 0) info[i] = -4;
            else if(lda_ < nrowA_) info[i] = -7;
            else if(ldc_ < n_ ) info[i] = -10;
        }

        int64_t info_ = 0;
        #pragma omp parallel for reduction(+:info_)
        for(size_t i = 0; i < batchCount; i++){
            info_ += info[i];
        }
        blas_error_if( info_ != 0 );
    }
}

// -----------------------------------------------------------------------------
// batch hemm check
template<typename T>
void symm_check(
        std::vector<blas::Side> const &side,
        std::vector<blas::Uplo> const &uplo, 
        std::vector<int64_t>    const &m, 
        std::vector<int64_t>    const &n, 
        std::vector<T>          const &alpha, 
        std::vector<T*>         const &A, std::vector<int64_t> const &lda, 
        std::vector<T*>         const &B, std::vector<int64_t> const &ldb, 
        std::vector<T>          const &beta, 
        std::vector<T*>         const &C, std::vector<int64_t> const &ldc, 
        const size_t batchCount, std::vector<int64_t> &info)
{
    hemm_check(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc, batchCount, info);
}

// -----------------------------------------------------------------------------
// batch syrk check
template<typename T>
void syrk_check(
        std::vector<blas::Uplo> const &uplo, 
        std::vector<blas::Op>   const &trans, 
        std::vector<int64_t>    const &n, 
        std::vector<int64_t>    const &k, 
        std::vector<T>          const &alpha, 
        std::vector<T*>         const &A, std::vector<int64_t> const &lda, 
        std::vector<T>          const &beta, 
        std::vector<T*>         const &C, std::vector<int64_t> const &ldc, 
        const size_t batchCount, std::vector<int64_t> &info)
{
    // size error checking
    blas_error_if( (uplo.size()  != 1 && uplo.size()  != batchCount) );
    blas_error_if( (trans.size() != 1 && trans.size() != batchCount) );

    blas_error_if( (n.size() != 1 && n.size() != batchCount) );
    blas_error_if( (k.size() != 1 && k.size() != batchCount) );

    blas_error_if( (A.size() != 1 && A.size() != batchCount) );
    blas_error_if(  C.size() != batchCount );

    blas_error_if( (lda.size() != 1 && lda.size() != batchCount) );
    blas_error_if( (ldc.size() != 1 && ldc.size() != batchCount) );

    blas_error_if( (alpha.size() != 1 && alpha.size() != batchCount) );
    blas_error_if( (beta.size()  != 1 && beta.size()  != batchCount) );

    blas_error_if( A.size() == 1 && 
                  (lda.size()    > 1                  || 
                   n.size()      > 1                  ||
                   k.size()      > 1                  ||
                   (trans.size() > 1 && n[0] != k[0]) ));

    blas_error_if( C.size() == 1 && 
                  (uplo.size()  > 1 || 
                   trans.size() > 1 || 
                   n.size()     > 1 || 
                   k.size()     > 1 || 
                   alpha.size() > 1 || 
                   A.size()     > 1 || 
                   lda.size()   > 1 || 
                   beta.size()  > 1 || 
                   ldc.size()   > 1 ));

    if(info.size() == 1){
        /* argument based error reporting */
        int64_t linfo;

        // uplo
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < uplo.size(); i++){
            linfo += (uplo[i] != Uplo::Lower  && 
                      uplo[i] != Uplo::Upper
                      ) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -1 : 0;
        blas_error_if( linfo > 0 );

        // trans
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < trans.size(); i++){
            linfo += (trans[i] != Op::NoTrans  && 
                      trans[i] != Op::Trans 
                      ) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -2 : 0;
        blas_error_if( linfo > 0 );

        // n
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < n.size(); i++){
            linfo += (n[i] < 0) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -3 : 0;
        blas_error_if( linfo > 0 );

        // k
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < k.size(); i++){
            linfo += (k[i] < 0) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -4 : 0;
        blas_error_if( linfo > 0 );

        // lda
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < A.size(); i++){
            Op trans_      = extract<Op>(trans, i);
            int64_t lda_   = extract<int64_t>(lda, i);
            int64_t nrowA_ = (trans_ == Op::NoTrans) ? extract<int64_t>(n, i) : extract<int64_t>(k, i);
            linfo += (lda_ < nrowA_) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -7 : 0;
        blas_error_if( linfo > 0 );

        // ldc
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < C.size(); i++){
            int64_t n_   = extract<int64_t>(n, i);
            int64_t ldc_ = extract<int64_t>(ldc, i);
            linfo += (ldc_ < n_) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -10 : 0;
        blas_error_if( linfo > 0 );
    }
    else{
        /* problem based eror reporting */
        #pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < batchCount; i++){
            Uplo  uplo_ = extract<Uplo>(uplo , i);
            Op   trans_ = extract<Op>(trans , i);

            int64_t n_ = extract<int64_t>(n, i); 
            int64_t k_ = extract<int64_t>(k, i);

            int64_t lda_ = extract<int64_t>(lda, i);
            int64_t ldc_ = extract<int64_t>(ldc, i);

            int64_t nrowA_ = (trans_ == Op::NoTrans) ? n_ : k_;

            info[i] = 0;
            if(uplo_ != Uplo::Lower && uplo_ != Uplo::Upper) {
                info[i] = -1;
            }
            else if(trans_ != Op::NoTrans && trans_ != Op::Trans) {
                info[i] = -2;
            }
            else if(n_ < 0) info[i] = -3;
            else if(k_ < 0) info[i] = -4;
            else if(lda_ < nrowA_) info[i] = -7;
            else if(ldc_ < n_ ) info[i] = -10;
        }

        int64_t info_ = 0;
        #pragma omp parallel for reduction(+:info_)
        for(size_t i = 0; i < batchCount; i++){
            info_ += info[i];
        }
        blas_error_if( info_ != 0 );
    }
}

// -----------------------------------------------------------------------------
// batch her2k check
template<typename T, typename scalarT>
void her2k_check(
        std::vector<blas::Uplo> const &uplo, 
        std::vector<blas::Op>   const &trans, 
        std::vector<int64_t>    const &n, 
        std::vector<int64_t>    const &k, 
        std::vector<T>          const &alpha, 
        std::vector<T*>         const &A, std::vector<int64_t> const &lda, 
        std::vector<T*>         const &B, std::vector<int64_t> const &ldb, 
        std::vector<scalarT>    const &beta, 
        std::vector<T*>         const &C, std::vector<int64_t> const &ldc, 
        const size_t batchCount, std::vector<int64_t> &info)
{
    // size error checking
    blas_error_if( (uplo.size()  != 1 && uplo.size()  != batchCount) );
    blas_error_if( (trans.size() != 1 && trans.size() != batchCount) );

    blas_error_if( (n.size() != 1 && n.size() != batchCount) );
    blas_error_if( (k.size() != 1 && k.size() != batchCount) );

    blas_error_if( (A.size() != 1 && A.size() != batchCount) );
    blas_error_if( (B.size() != 1 && B.size() != batchCount) );
    blas_error_if(  C.size() != batchCount );

    blas_error_if( (lda.size() != 1 && lda.size() != batchCount) );
    blas_error_if( (ldb.size() != 1 && ldb.size() != batchCount) );
    blas_error_if( (ldc.size() != 1 && ldc.size() != batchCount) );

    blas_error_if( (alpha.size() != 1 && alpha.size() != batchCount) );
    blas_error_if( (beta.size()  != 1 && beta.size()  != batchCount) );

    blas_error_if( A.size() == 1 && 
                  (lda.size()    > 1                  || 
                   n.size()      > 1                  ||
                   k.size()      > 1                  ||
                   (trans.size() > 1 && n[0] != k[0]) ));

    blas_error_if( B.size() == 1 && 
                  (ldb.size()    > 1                  || 
                   n.size()      > 1                  ||
                   k.size()      > 1                  ||
                   (trans.size() > 1 && n[0] != k[0]) ));

    blas_error_if( C.size() == 1 && 
                  (uplo.size()  > 1 || 
                   trans.size() > 1 || 
                   n.size()     > 1 || 
                   k.size()     > 1 || 
                   alpha.size() > 1 || 
                   A.size()     > 1 || 
                   lda.size()   > 1 || 
                   B.size()     > 1 || 
                   ldb.size()   > 1 || 
                   beta.size()  > 1 || 
                   ldc.size()   > 1 ));

    if(info.size() == 1){
        /* argument based error reporting */
        int64_t linfo;

        // uplo
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < uplo.size(); i++){
            linfo += (uplo[i] != Uplo::Lower  && 
                      uplo[i] != Uplo::Upper
                      ) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -1 : 0;
        blas_error_if( linfo > 0 );

        // trans
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < trans.size(); i++){
            linfo += (trans[i] != Op::NoTrans  && 
                      trans[i] != Op::ConjTrans 
                      ) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -2 : 0;
        blas_error_if( linfo > 0 );

        // n
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < n.size(); i++){
            linfo += (n[i] < 0) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -3 : 0;
        blas_error_if( linfo > 0 );

        // k
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < k.size(); i++){
            linfo += (k[i] < 0) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -4 : 0;
        blas_error_if( linfo > 0 );

        // lda
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < A.size(); i++){
            Op trans_      = extract<Op>(trans, i);
            int64_t lda_   = extract<int64_t>(lda, i);
            int64_t nrowA_ = (trans_ == Op::NoTrans) ? extract<int64_t>(n, i) : extract<int64_t>(k, i);
            linfo += (lda_ < nrowA_) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -7 : 0;
        blas_error_if( linfo > 0 );

        // ldb
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < B.size(); i++){
            Op trans_      = extract<Op>(trans, i);
            int64_t ldb_   = extract<int64_t>(ldb, i);
            int64_t nrowB_ = (trans_ == Op::NoTrans) ? extract<int64_t>(n, i) : extract<int64_t>(k, i);
            linfo += (ldb_ < nrowB_) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -9 : 0;
        blas_error_if( linfo > 0 );

        // ldc
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < C.size(); i++){
            int64_t n_   = extract<int64_t>(n, i);
            int64_t ldc_ = extract<int64_t>(ldc, i);
            linfo += (ldc_ < n_) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -12 : 0;
        blas_error_if( linfo > 0 );
    }
    else{
        /* problem based eror reporting */
        #pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < batchCount; i++){
            Uplo  uplo_ = extract<Uplo>(uplo , i);
            Op   trans_ = extract<Op>(trans , i);

            int64_t n_ = extract<int64_t>(n, i); 
            int64_t k_ = extract<int64_t>(k, i);

            int64_t lda_ = extract<int64_t>(lda, i);
            int64_t ldb_ = extract<int64_t>(ldb, i);
            int64_t ldc_ = extract<int64_t>(ldc, i);

            int64_t nrowA_ = (trans_ == Op::NoTrans) ? n_ : k_;
            int64_t nrowB_ = (trans_ == Op::NoTrans) ? n_ : k_;

            info[i] = 0;
            if(uplo_ != Uplo::Lower && uplo_ != Uplo::Upper) {
                info[i] = -1;
            }
            else if(trans_ != Op::NoTrans && trans_ != Op::ConjTrans) {
                info[i] = -2;
            }
            else if(n_ < 0) info[i] = -3;
            else if(k_ < 0) info[i] = -4;
            else if(lda_ < nrowA_) info[i] = -7;
            else if(ldb_ < nrowB_) info[i] = -9;
            else if(ldc_ < n_ ) info[i] = -12;
        }

        int64_t info_ = 0;
        #pragma omp parallel for reduction(+:info_)
        for(size_t i = 0; i < batchCount; i++){
            info_ += info[i];
        }
        blas_error_if( info_ != 0 );
    }
}

// -----------------------------------------------------------------------------
// batch syr2k check
template<typename T>
void syr2k_check(
        std::vector<blas::Uplo> const &uplo, 
        std::vector<blas::Op>   const &trans, 
        std::vector<int64_t>    const &n, 
        std::vector<int64_t>    const &k, 
        std::vector<T>          const &alpha, 
        std::vector<T*>         const &A, std::vector<int64_t> const &lda, 
        std::vector<T*>         const &B, std::vector<int64_t> const &ldb, 
        std::vector<T>          const &beta, 
        std::vector<T*>         const &C, std::vector<int64_t> const &ldc, 
        const size_t batchCount, std::vector<int64_t> &info)
{
    // size error checking
    blas_error_if( (uplo.size()  != 1 && uplo.size()  != batchCount) );
    blas_error_if( (trans.size() != 1 && trans.size() != batchCount) );

    blas_error_if( (n.size() != 1 && n.size() != batchCount) );
    blas_error_if( (k.size() != 1 && k.size() != batchCount) );

    blas_error_if( (A.size() != 1 && A.size() != batchCount) );
    blas_error_if( (B.size() != 1 && B.size() != batchCount) );
    blas_error_if(  C.size() != batchCount );

    blas_error_if( (lda.size() != 1 && lda.size() != batchCount) );
    blas_error_if( (ldb.size() != 1 && ldb.size() != batchCount) );
    blas_error_if( (ldc.size() != 1 && ldc.size() != batchCount) );

    blas_error_if( (alpha.size() != 1 && alpha.size() != batchCount) );
    blas_error_if( (beta.size()  != 1 && beta.size()  != batchCount) );

    blas_error_if( A.size() == 1 && 
                  (lda.size()    > 1                  || 
                   n.size()      > 1                  ||
                   k.size()      > 1                  ||
                   (trans.size() > 1 && n[0] != k[0]) ));

    blas_error_if( B.size() == 1 && 
                  (ldb.size()    > 1                  || 
                   n.size()      > 1                  ||
                   k.size()      > 1                  ||
                   (trans.size() > 1 && n[0] != k[0]) ));

    blas_error_if( C.size() == 1 && 
                  (uplo.size()  > 1 || 
                   trans.size() > 1 || 
                   n.size()     > 1 || 
                   k.size()     > 1 || 
                   alpha.size() > 1 || 
                   A.size()     > 1 || 
                   lda.size()   > 1 || 
                   B.size()     > 1 || 
                   ldb.size()   > 1 || 
                   beta.size()  > 1 || 
                   ldc.size()   > 1 ));

    if(info.size() == 1){
        /* argument based error reporting */
        int64_t linfo;

        // uplo
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < uplo.size(); i++){
            linfo += (uplo[i] != Uplo::Lower  && 
                      uplo[i] != Uplo::Upper
                      ) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -1 : 0;
        blas_error_if( linfo > 0 );

        // trans
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < trans.size(); i++){
            linfo += (trans[i] != Op::NoTrans  && 
                      trans[i] != Op::Trans 
                      ) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -2 : 0;
        blas_error_if( linfo > 0 );

        // n
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < n.size(); i++){
            linfo += (n[i] < 0) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -3 : 0;
        blas_error_if( linfo > 0 );

        // k
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < k.size(); i++){
            linfo += (k[i] < 0) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -4 : 0;
        blas_error_if( linfo > 0 );

        // lda
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < A.size(); i++){
            Op trans_      = extract<Op>(trans, i);
            int64_t lda_   = extract<int64_t>(lda, i);
            int64_t nrowA_ = (trans_ == Op::NoTrans) ? extract<int64_t>(n, i) : extract<int64_t>(k, i);
            linfo += (lda_ < nrowA_) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -7 : 0;
        blas_error_if( linfo > 0 );

        // ldb
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < B.size(); i++){
            Op trans_      = extract<Op>(trans, i);
            int64_t ldb_   = extract<int64_t>(ldb, i);
            int64_t nrowB_ = (trans_ == Op::NoTrans) ? extract<int64_t>(n, i) : extract<int64_t>(k, i);
            linfo += (ldb_ < nrowB_) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -9 : 0;
        blas_error_if( linfo > 0 );

        // ldc
        linfo = 0;
        #pragma omp parallel for reduction(+:linfo)
        for(size_t i = 0; i < C.size(); i++){
            int64_t n_   = extract<int64_t>(n, i);
            int64_t ldc_ = extract<int64_t>(ldc, i);
            linfo += (ldc_ < n_) ? 1 : 0;
        }
        info[0] = (linfo > 0) ? -12 : 0;
        blas_error_if( linfo > 0 );
    }
    else{
        /* problem based eror reporting */
        #pragma omp parallel for schedule(dynamic)
        for(size_t i = 0; i < batchCount; i++){
            Uplo  uplo_ = extract<Uplo>(uplo , i);
            Op   trans_ = extract<Op>(trans , i);

            int64_t n_ = extract<int64_t>(n, i); 
            int64_t k_ = extract<int64_t>(k, i);

            int64_t lda_ = extract<int64_t>(lda, i);
            int64_t ldb_ = extract<int64_t>(ldb, i);
            int64_t ldc_ = extract<int64_t>(ldc, i);

            int64_t nrowA_ = (trans_ == Op::NoTrans) ? n_ : k_;
            int64_t nrowB_ = (trans_ == Op::NoTrans) ? n_ : k_;

            info[i] = 0;
            if(uplo_ != Uplo::Lower && uplo_ != Uplo::Upper) {
                info[i] = -1;
            }
            else if(trans_ != Op::NoTrans && trans_ != Op::Trans) {
                info[i] = -2;
            }
            else if(n_ < 0) info[i] = -3;
            else if(k_ < 0) info[i] = -4;
            else if(lda_ < nrowA_) info[i] = -7;
            else if(ldb_ < nrowB_) info[i] = -9;
            else if(ldc_ < n_ ) info[i] = -12;
        }

        int64_t info_ = 0;
        #pragma omp parallel for reduction(+:info_)
        for(size_t i = 0; i < batchCount; i++){
            info_ += info[i];
        }
        blas_error_if( info_ != 0 );
    }
}

}        // namespace batch
}        // namespace blas

#endif    // BATCH_COMMON_HH
