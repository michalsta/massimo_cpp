#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
//#include <IsoSpec++/isoSpec++.h>
#include <IsoSpec++/unity-build.cpp>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <array>
#include <cstdlib>



class ProblematicInput
{
public:
    const size_t N;
    const size_t N_minimal;
    const double precision;
    const std::vector<size_t> frame_indices;
    const std::vector<double> frame_probs;
    const std::vector<size_t> scan_indices;
    const std::vector<double> scan_probs;
    const std::vector<size_t> tof_indices;
    const std::vector<double> tof_probs;
    // Constructor to initialize the class with the provided parameters
    // Using initializer list for better performance and clarity
    ProblematicInput(size_t N,
                      size_t N_minimal,
                      double precision,
                      std::vector<size_t> const &frame_indices,
                      std::vector<double> const &frame_probs,
                      std::vector<size_t> const &scan_indices,
                      std::vector<double> const &scan_probs,
                      std::vector<size_t> const &tof_indices,
                      std::vector<double> const &tof_probs)
        : N(N), N_minimal(N_minimal), precision(precision), frame_indices(frame_indices), frame_probs(frame_probs),
          scan_indices(scan_indices), scan_probs(scan_probs),
          tof_indices(tof_indices), tof_probs(tof_probs) 
        {
            if (frame_indices.size() != frame_probs.size() ||
                scan_indices.size() != scan_probs.size() ||
                tof_indices.size() != tof_probs.size()) 
            throw std::invalid_argument("Indices and probabilities vectors must have the same size.");
        };
    
};


void worker(std::atomic<size_t> &n_processed, 
            const std::vector<ProblematicInput> &inputs, 
            std::mutex &out_file_lock, 
            std::fstream &out, 
            size_t thread_id) 
{
    while(n_processed < inputs.size())
    {
        size_t index = n_processed.fetch_add(1);
        if (index >= inputs.size())
            break; // No more inputs to process
        
        const ProblematicInput &input = inputs[index];
        std::array<int, 3> isotopeNumbers = {(int)input.frame_indices.size(), (int)input.scan_indices.size(), (int)input.tof_indices.size()};
        std::array<int, 3> atomCounts = {1,1,1};
        std::array<double*, 3> isotopeMasses = {
            (double*)calloc(input.frame_indices.size(), sizeof(double)),
            (double*)calloc(input.scan_indices.size(), sizeof(double)),
            (double*)calloc(input.tof_indices.size(), sizeof(double))
        };
        std::array<const double*, 3> isotopeProbabilities = {
            input.frame_probs.data(),
            input.scan_probs.data(),
            input.tof_probs.data()
        };

        IsoSpec::Iso iso(3, isotopeNumbers.data(), atomCounts.data(), isotopeMasses.data(), isotopeProbabilities.data());
        IsoSpec::IsoStochasticGenerator generator(std::move(iso), input.N, input.precision);

        // TODO: we can replace std::vector here by preallocated memory of size N
        std::vector<double> masses;
        masses.reserve(input.N_minimal);
        std::vector<double> probabilities;
        probabilities.reserve(input.N_minimal);

        while (generator.advanceToNextConfiguration()) {
            masses.push_back(generator.mass());
            probabilities.push_back(generator.prob());
        }
        // Write results to the output file
        {
            std::lock_guard<std::mutex> lock(out_file_lock);
            //out.write(reinterpret_cast<const char*>(masses.data()), masses.size() * sizeof(double));
            out.write(reinterpret_cast<const char*>(probabilities.data()), probabilities.size() * sizeof(double));
        }



        free(isotopeMasses[0]);
        free(isotopeMasses[1]);
        free(isotopeMasses[2]);
    }
}

void Massimize(const std::vector<ProblematicInput> &inputs, size_t n_threads, const std::string &output_path) 
{
    std::fstream out(output_path, std::ios::out | std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }
    
    std::mutex out_file_lock;
    std::atomic<size_t> n_processed(0);
    std::vector<std::thread> threads;

    for(size_t ii = 0; ii < n_threads; ++ii) {
        threads.emplace_back(worker, std::ref(n_processed), std::ref(inputs), std::ref(out_file_lock), std::ref(out), ii);
    }

    for(auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    out.close();
    std::cout << "Processing completed. Output written to " << output_path << std::endl;
}
        





namespace nb = nanobind;

NB_MODULE(massimo_cpp_ext, m) {
    m.def("hello", [](){ std::cout << "hello!\n"; });
    m.def("Massimize", &Massimize, "Function to process isotopic data");
    nb::class_<ProblematicInput>(m, "ProblematicInput")
        .def(nb::init<size_t, size_t, double, std::vector<size_t> const &, std::vector<double> const &,
                      std::vector<size_t> const &, std::vector<double> const &,
                      std::vector<size_t> const &, std::vector<double> const &>());
}
