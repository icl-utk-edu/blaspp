#!/bin/tcsh

setenv opts     ""

setenv type     "--type s,d,c,z"
setenv real     "--type s,d"
setenv complex  "--type c,z"

setenv n        "--dim 25:100:25"
setenv mn       "${n} --dim 25x50,50x25 --dim 25x100,100x25"
setenv mnk      "${mn} --dim 25x50x25,25x25x50 --dim 25x100x25,25x25x100"

setenv incx_pos "--incx 1,2"
setenv incx     "--incx 1,2,-1,-2"

setenv incy_pos "--incy 1,2"
setenv incy     "--incy 1,2,-1,-2"

setenv layout   "--layout c,r"
setenv side     "--side l,r"
setenv uplo     "--uplo l,u"
setenv trans    "--trans n,t,c"
setenv trans_nt "--trans n,t"
setenv trans_nc "--trans n,c"
setenv transA   "--transA n,t,c"
setenv transB   "--transB n,t,c"
setenv diag     "--diag n,u"
setenv align    "--align 32"

# smaller set for debugging
#setenv opts    "--verbose 2"
#setenv n       "--dim 10"
#setenv mn      "${n} --dim 10x15,15x10"
#setenv mnk     "${mn} --dim 10x15x10,10x10x15"
#setenv incx    "--incx 1,-1"
#setenv incy    "--incx 1,-1"

set echo

echo "-------------------- Level 1 BLAS"
./test asum  ${opts} ${type} ${n} ${incx_pos}     > asum.txt
./test axpy  ${opts} ${type} ${n} ${incx} ${incy} > axpy.txt
./test copy  ${opts} ${type} ${n} ${incx} ${incy} > copy.txt
./test dot   ${opts} ${type} ${n} ${incx} ${incy} > dot.txt
./test dotu  ${opts} ${type} ${n} ${incx} ${incy} > dotu.txt
./test iamax ${opts} ${type} ${n} ${incx_pos}     > iamax.txt
./test nrm2  ${opts} ${type} ${n} ${incx_pos}     > nrm2.txt
./test rot   ${opts} ${real} ${n} ${incx} ${incy} > rot_sd.txt  # todo: complex
./test rotm  ${opts} ${real} ${n} ${incx} ${incy} > rotm.txt
./test scal  ${opts} ${type} ${n} ${incx_pos}     > scal.txt
./test swap  ${opts} ${type} ${n} ${incx} ${incy} > swap.txt

echo "-------------------- Level 2 BLAS"
./test gemv  ${opts} ${type} ${layout} ${align} ${trans} ${mn} ${incx} ${incy} > gemv.txt
./test ger   ${opts} ${type} ${layout} ${align}          ${mn} ${incx} ${incy} > ger.txt
./test geru  ${opts} ${type} ${layout} ${align}          ${mn} ${incx} ${incy} > geru.txt
./test hemv  ${opts} ${type} ${layout} ${align} ${uplo}  ${n}  ${incx} ${incy} > hemv.txt
./test her   ${opts} ${type} ${layout} ${align} ${uplo}  ${n}  ${incx}         > her.txt
./test her2  ${opts} ${type} ${layout} ${align} ${uplo}  ${n}  ${incx} ${incy} > her2.txt
./test symv  ${opts} ${type} ${layout} ${align} ${uplo}  ${n}  ${incx} ${incy} > symv.txt
./test syr   ${opts} ${type} ${layout} ${align} ${uplo}  ${n}  ${incx}         > syr.txt
./test syr2  ${opts} ${type} ${layout} ${align} ${uplo}  ${n}  ${incx} ${incy} > syr2.txt
./test trmv  ${opts} ${type} ${layout} ${align} ${uplo} ${trans} ${diag} ${n} ${incx} > trmv.txt
./test trsv  ${opts} ${type} ${layout} ${align} ${uplo} ${trans} ${diag} ${n} ${incx} > trsv.txt

echo "-------------------- Level 3 BLAS"
./test gemm  ${opts} ${type} ${layout} ${align} ${transA} ${transB} ${mnk} > gemm.txt
./test hemm  ${opts} ${type} ${layout} ${align} ${side}   ${uplo}   ${mn}  > hemm.txt
./test symm  ${opts} ${type} ${layout} ${align} ${side}   ${uplo}   ${mn}  > symm.txt

./test herk  ${opts} ${real}    ${layout} ${align} ${uplo} ${trans}    ${mn} > herk_sd.txt
./test herk  ${opts} ${complex} ${layout} ${align} ${uplo} ${trans_nc} ${mn} > herk_cz.txt
./test syrk  ${opts} ${real}    ${layout} ${align} ${uplo} ${trans}    ${mn} > syrk_sd.txt
./test syrk  ${opts} ${complex} ${layout} ${align} ${uplo} ${trans_nt} ${mn} > syrk_cz.txt
./test her2k ${opts} ${real}    ${layout} ${align} ${uplo} ${trans}    ${mn} > her2k_sd.txt
./test her2k ${opts} ${complex} ${layout} ${align} ${uplo} ${trans_nc} ${mn} > her2k_cz.txt
./test syr2k ${opts} ${real}    ${layout} ${align} ${uplo} ${trans}    ${mn} > syr2k_sd.txt
./test syr2k ${opts} ${complex} ${layout} ${align} ${uplo} ${trans_nt} ${mn} > syr2k_cz.txt

unset echo
