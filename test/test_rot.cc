#include "rot.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_rot_work( Params& params, bool run )
{
    int64_t n = 100;
    int64_t incx = 1;
    int64_t incy = 1;
    T *x = new T[n];
    T *y = new T[n];
    typename blas::traits<T>::norm_t c = 1;
    T s = 0;

    blas::rot( n, x, incx, y, incy, c, s );

    delete[] x;
    delete[] y;
}

// -----------------------------------------------------------------------------
void test_rot( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_rot_work< int >( params, run );  // todo: generic implementation
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_rot_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_rot_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_rot_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_rot_work< std::complex<double> >( params, run );
            break;
    }
}
