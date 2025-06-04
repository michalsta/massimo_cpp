cmake_minimum_required(VERSION 3.15)

project(massimo_cpp)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind REQUIRED CONFIG)

set(CMAKE_CXX_STANDARD 20)

nanobind_add_module(massimo_cpp_ext
    NB_STATIC NOMINSIZE
    src/massimo_main.cpp)

    target_include_directories(massimo_cpp_ext
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

install(TARGETS massimo_cpp_ext LIBRARY DESTINATION massimo_cpp)
