#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
//#include <IsoSpec++/isoSpec++.h>
#include <massimo_cpp/massimo_impl.cpp>


namespace nb = nanobind;

NB_MODULE(massimo_cpp_ext, m) {
    m.def("Massimize", &Massimize,
        nb::arg("inputs"),
        nb::arg("output_dir_path"),
        nb::arg("n_threads") = 1,
        nb::arg("beta_bias") = 5.0,
        "Function to process isotopic data");
    nb::class_<ProblematicInput>(m, "ProblematicInput")
        .def(nb::init<size_t, size_t, double, std::vector<size_t> const &, std::vector<double> const &,
                      std::vector<size_t> const &, std::vector<double> const &,
                      std::vector<size_t> const &, std::vector<double> const &>())
        .def("to_cpp_string", &ProblematicInput::to_cpp_string, "Convert the input data to a C++ string representation");
}
