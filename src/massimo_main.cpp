#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>

//#include <IsoSpec++/isoSpec++.h>
#include "massimo_cpp/massimo_impl.cpp"


namespace nb = nanobind;

enum class IsoSpecBackend {
    Layered,
    Ordered
};

void Massimize_(std::vector<ProblematicInput> &inputs, const std::string &output_dir_path, size_t n_threads, double beta_bias = 5.0, std::optional<uint_fast32_t> seed = std::nullopt, IsoSpecBackend iso_backend = IsoSpecBackend::Layered) {
    switch (iso_backend) {
        case IsoSpecBackend::Layered:
            Massimize<IsoSpec::IsoLayeredGeneratorTemplate<IsoSpec::LayeredMarginal>>(inputs, output_dir_path, n_threads, beta_bias, seed);
            break;
        case IsoSpecBackend::Ordered:
            Massimize<IsoSpec::IsoOrderedGenerator>(inputs, output_dir_path, n_threads, beta_bias, seed);
            break;
        default:
            throw std::invalid_argument("Unsupported IsoSpec backend");
    }
}

void MassimizeOrdered(std::vector<ProblematicInput> &inputs, const std::string &output_dir_path, size_t n_threads, double beta_bias = 5.0, std::optional<uint_fast32_t> seed = std::nullopt) {
    Massimize<IsoSpec::IsoOrderedGenerator>(inputs, output_dir_path, n_threads, beta_bias, seed);
}
void MassimizeLayered(std::vector<ProblematicInput> &inputs, const std::string &output_dir_path, size_t n_threads, double beta_bias = 5.0, std::optional<uint_fast32_t> seed = std::nullopt) {
    Massimize<IsoSpec::IsoLayeredGeneratorTemplate<IsoSpec::LayeredMarginal>>(inputs, output_dir_path, n_threads, beta_bias, seed);
}
NB_MODULE(massimo_cpp_ext, m) {
    m.def("Massimize", &MassimizeLayered,
        nb::arg("inputs"),
        nb::arg("output_dir_path"),
        nb::arg("n_threads") = 1,
        nb::arg("beta_bias") = 5.0,
        nb::arg("seed") = std::nullopt,
//        nb::arg("iso_backend") = IsoSpecBackend::Layered,
        "Function to process isotopic data");
    nb::class_<ProblematicInput>(m, "ProblematicInput")
        .def(nb::init<size_t, size_t, double, std::vector<size_t> const &, std::vector<double> const &,
                      std::vector<size_t> const &, std::vector<double> const &,
                      std::vector<size_t> const &, std::vector<double> const &>())
        .def("to_cpp_string", &ProblematicInput::to_cpp_string, "Convert the input data to a C++ string representation");
    nb::enum_<IsoSpecBackend>(m, "IsoSpecBackend")
        .value("Layered", IsoSpecBackend::Layered)
        .value("Ordered", IsoSpecBackend::Ordered)
        .export_values();
}