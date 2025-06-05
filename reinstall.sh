#!/bin/bash

pip uninstall -y massimo_cpp; VERBOSE=1 pip install -v -e .
#pip uninstall -y massimo_cpp; VERBOSE=1 CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Debug" CXXFLAGS="-D_GLIBCXX_DEBUG -O0 -g" pip install -v -e . 
