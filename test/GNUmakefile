include make.inc

# defaults if not defined in make.inc
CXX      ?= g++

LDFLAGS  ?= -fPIC -fopenmp
CXXFLAGS ?= -fPIC -fopenmp -MMD -std=c++11 -pedantic \
            -Wall -Wno-unused-local-typedefs -Wno-unused-but-set-variable
#CXXFLAGS += -Werror
#CXXFLAGS += -Wconversion

CPPFLAGS ?= -I${CBLASDIR}

LIBS     ?= -lblas


# --------------------
# SLATE specific flags
pwd = ${shell pwd}

INC = -I../../libtest -I../include

LIBS += -L../../libtest -Wl,-rpath,${pwd}/../../libtest -ltest


# --------------------
# files
src := ${wildcard *.cc}
obj := ${addsuffix .o, ${basename ${src}}}
dep := ${addsuffix .d, ${basename ${src}}}


# --------------------
# rules
.PHONY: clean test_headers

test: ${obj}
	${CXX} ${LDFLAGS} -o $@ $^ ${LIBS}

%.o: %.cc
	${CXX} ${CXXFLAGS} ${CPPFLAGS} ${INC} -c -o $@ $<

%.i: %.cc
	${CXX} ${CXXFLAGS} ${CPPFLAGS} ${INC} -E -o $@ $<

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
