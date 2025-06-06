ISOSPEC := $(shell python -m IsoSpecPy --include)
ISOSPEC_SO := $(shell python -m IsoSpecPy --libpath)

all:
	#g++ -fsanitize=address -Og -g -std=c++20 -I.. test.cpp
	g++ -std=c++20 -I.. -I$(ISOSPEC) -L$(ISOSPEC) -lIsoSpecCppPy -Wl,-rpath,$(ISOSPEC) -Og -g test.cpp
