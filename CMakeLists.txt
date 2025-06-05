cmake_minimum_required(VERSION 3.15)

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -undefined dynamic_lookup")

execute_process(
    COMMAND python -m IsoSpecPy --libpath
    OUTPUT_VARIABLE ISOSPEC_LIBPATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND python -m IsoSpecPy --include
    OUTPUT_VARIABLE ISOSPEC_INCLUDEPATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)


project(massimo_cpp)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind REQUIRED CONFIG)

set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_DEBUG} -D_GLIBCXX_DEBUG -g -O0")

nanobind_add_module(massimo_cpp_ext
SHARED NOMINSIZE
src/massimo_main.cpp)

target_include_directories(massimo_cpp_ext
PRIVATE
    ${ISOSPEC_INCLUDEPATH})


#install(TARGETS nanobind LIBRARY DESTINATION massimo_cpp)
install(TARGETS massimo_cpp_ext LIBRARY DESTINATION massimo_cpp)
