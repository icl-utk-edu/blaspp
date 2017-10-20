.PHONY: test clean docs

test:
	cd test && ${MAKE}

clean:
	cd include && ${MAKE} $@
	cd test    && ${MAKE} $@

docs:
	doxygen docs/doxygen/doxyfile.conf
