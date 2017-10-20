CXX = g++
CXXFLAGS = -Wconversion

.PHONY: test_headers clean docs

# --------------------
# compile headers to verify self-sufficiency
header_gch = ${addprefix gch/, ${addsuffix .gch, ${wildcard *.hh}}}

test_headers: ${header_gch}

gch/%.hh.gch: %.hh | gch
	${CXX} ${CXXFLAGS} -c -o $@ $<

gch:
	mkdir $@

clean:
	-${RM} gch/*.gch

# --------------------
docs:
	cd .. && ${MAKE} $@
