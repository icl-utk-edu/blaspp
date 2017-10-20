pwd = ${shell pwd}

CXX      = g++
LDFLAGS  = -fPIC -fopenmp
CXXFLAGS = -fPIC -fopenmp -MMD -Wall -pedantic -std=c++11
CPPFLAGS = -I../../libtest -I../include -I/opt/intel/mkl/include \
           -I${CBLASDIR} \
           -DFORTRAN_LOWER \
           -DHAVE_MACOS_ACCELERATE \
           -DBLAS_COMPLEX_RETURN_ARGUMENT \
           -DMKL_Complex8="std::complex<float>" \
           -DMKL_Complex16="std::complex<double>"
LIBS     = -L../../libtest -Wl,-rpath,${pwd}/../../libtest -ltest \
           -framework Accelerate

CXXFLAGS += -Wno-unused-local-typedefs -Wno-unused-but-set-variable
CXXFLAGS += -Werror
#CXXFLAGS += -Wconversion

src := ${wildcard *.cc}
obj := ${addsuffix .o, ${basename ${src}}}
dep := ${addsuffix .d, ${basename ${src}}}

.PHONY: clean test_headers

test: ${obj}
	${CXX} ${LDFLAGS} -o $@ $^ ${LIBS}

%.o: %.cc
	${CXX} ${CXXFLAGS} ${CPPFLAGS} -c -o $@ $<

%.i: %.cc
	${CXX} ${CXXFLAGS} ${CPPFLAGS} -E -o $@ $<

clean:
	-${RM} *.o *.d test gch/*.gch

-include ${dep}

# --------------------
# compile headers to verify self-sufficiency
header_gch = ${addprefix gch/, ${addsuffix .gch, ${wildcard *.hh}}}

test_headers: ${header_gch}

gch/%.hh.gch: %.hh | gch
	${CXX} ${CXXFLAGS} ${CPPFLAGS} -c -o $@ $<

gch:
	mkdir $@
