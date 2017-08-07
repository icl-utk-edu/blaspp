CXX = g++
CXXFLAGS = -Wconversion

header_gch = $(addprefix gch/, $(addsuffix .gch, $(wildcard *.hh)))

test_headers: $(header_gch)

gch/%.hh.gch: %.hh | gch
	$(CXX) $(CXXFLAGS) -c -o $@ $<

gch:
	mkdir $@

clean:
	-rm -f gch/*.gch
