pwd = $(shell pwd)

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

src := $(wildcard *.cc)
obj := $(addsuffix .o, $(basename ${src}))
dep := $(addsuffix .d, $(basename ${src}))

test: ${obj}
	${CXX} ${LDFLAGS} -o $@ $^ ${LIBS}

%.o: %.cc
	${CXX} ${CXXFLAGS} ${CPPFLAGS} -c -o $@ $<

%.i: %.cc
	${CXX} ${CXXFLAGS} ${CPPFLAGS} -E -o $@ $<

clean:
	-rm -f *.o *.d test

-include ${dep}
