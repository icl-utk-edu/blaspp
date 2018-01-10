#include "test.hh"
#include "blas_util.hh"

void test_error( Params& params, bool run )
{
    typedef long long lld;

    int64_t m = params.dim.m();
    int64_t n = params.dim.n();

    if (! run) {
        printf( "test error checks the internal error routines:\n"
                "if m == 100, checks: blas_error_if( m == n );\n"
                "if m == 200, checks: blas_error_if_msg( m == n, \"m %%d == n %%d\", m, n );\n"
                "if m == 300, checks: assert( m != n );\n\n" );
        return;
    }

    if (m == 100) {
        blas_error_if( m == n );
    }
    else if (m == 200) {
        blas_error_if_msg( m == n, "m %lld == n %lld", (lld) m, (lld) n );
    }
    else if (m == 300) {
        assert( m != n );
    }
}
