#include <nanobind/nanobind.h>
#include <iostream>

NB_MODULE(massimo_cpp_ext, m) {
    m.def("hello", [](){ std::cout << "hello!\n"; });
}
