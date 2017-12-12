include make.inc

# defaults if not defined in make.inc
CXX      ?= g++

LDFLAGS  ?= -fPIC -fopenmp
CXXFLAGS ?= -fPIC -fopenmp -MMD -std=c++11 -pedantic \
            -Wall -Wno-unused-local-typedefs -Wno-unused-but-set-variable \
            -I${CBLASDIR}
#CXXFLAGS += -Werror
#CXXFLAGS += -Wconversion

LIBS     ?= -lblas

# ------------------------------------------------------------------------------
# BLAS++ specific flags
pwd = ${shell pwd}
libtest_path = ${realpath ${pwd}/../libtest}
libtest_src  = ${wildcard ${libtest_path}/*.cc} \
               ${wildcard ${libtest_path}/*.hh}
libtest_so   = ${libtest_path}/libtest.so

BLASPP_FLAGS = -I../libtest \
               -Iinclude

BLASPP_LIBS  = -L../libtest -Wl,-rpath,${libtest_path} -ltest

# ------------------------------------------------------------------------------
# files
test_src = ${wildcard test/*.cc}
test_obj = ${addsuffix .o, ${basename ${test_src}}}
test_dep = ${addsuffix .d, ${basename ${test_src}}}

# ------------------------------------------------------------------------------
# rules
.PHONY: default all include test clean test_headers

default: test

all: test

# default rules for subdirectories
include: test_headers

test: test/test

test/test: ${test_obj} ${libtest_so}
	${CXX} ${LDFLAGS} -o $@ ${test_obj} ${BLASPP_LIBS} ${LIBS}

${libtest_so}: ${libtest_src}
	cd ${libtest_path} && ${MAKE}

%.o: %.cc
	${CXX} ${CXXFLAGS} ${BLASPP_FLAGS} -c -o $@ $<

%.i: %.cc
	${CXX} ${CXXFLAGS} ${BLASPP_FLAGS} -E -o $@ $<

clean: include/clean test/clean

include/clean:
	-${RM} gch/include/*.gch

test/clean:
	-${RM} test/test test/*.{o,d} gch/test/*.gch

-include ${test_dep}

# ------------------------------------------------------------------------------
# precompile headers to verify self-sufficiency
headers     = ${wildcard include/*.h include/*.hh test/*.hh}
headers_gch = ${addprefix gch/, ${addsuffix .gch, ${headers}}}

test_headers: ${headers_gch}

gch/include/%.h.gch: include/%.h | gch/include
	${CXX} ${CXXFLAGS} ${BLASPP_FLAGS} -c -o $@ $<

gch/include/%.hh.gch: include/%.hh | gch/include
	${CXX} ${CXXFLAGS} ${BLASPP_FLAGS} -c -o $@ $<

gch/test/%.hh.gch: test/%.hh | gch/test
	${CXX} ${CXXFLAGS} ${BLASPP_FLAGS} -c -o $@ $<

# make directories
gch/include: | gch
gch/test:    | gch
gch/include gch/test gch:
	mkdir $@
