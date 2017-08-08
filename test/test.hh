#ifndef TEST_HH
#define TEST_HH

// -----------------------------------------------------------------------------
// Level 1 BLAS

void test_asum();
void test_axpy();
void test_copy();
void test_dot();
void test_iamax();
void test_nrm2();
void test_rot();
void test_rotg();
void test_rotm();
void test_rotmg();
void test_scal();
void test_swap();

// -----------------------------------------------------------------------------
// Level 2 BLAS

void test_gemv();
void test_symv();
void test_hemv();

#endif        //  #ifndef TEST_HH
