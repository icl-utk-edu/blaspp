CXX = g++
CXXFLAGS = -I../include -I/opt/intel/mkl/include -MMD
CXXFLAGS += -DFORTRAN_LOWER
CXXFLAGS += -DMKL_Complex8="std::complex<float>" \
            -DMKL_Complex16="std::complex<double>"
LIBS     = -framework Accelerate
CXXFLAGS += -Wconversion

src := $(wildcard *.cc)
obj := $(addsuffix .o, $(basename $(src)))
dep := $(addsuffix .d, $(basename $(src)))

test: $(obj)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

%.i: %.cc
	$(CXX) $(CXXFLAGS) -E -o $@ $<

clean:
	-rm -f *.o *.d test

-include $(dep)
