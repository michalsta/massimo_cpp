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
#include <algorithm>
#include <memory>
#include <span>
#include <string>
#include <cstdint>
#include <cmath>
#include <filesystem>





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

    std::string to_cpp_string() const {
        std::string result = "ProblematicInput(";
        result += std::to_string(N) + ", ";
        result += std::to_string(N_minimal) + ", ";
        result += std::to_string(precision) + ", ";
        result += "\n{";
        for (size_t i = 0; i < frame_indices.size(); ++i) {
            result += std::to_string(frame_indices[i]);
            if (i < frame_indices.size() - 1) {
                result += ", ";
            }
        }
        result += "}, \n{";
        for (size_t i = 0; i < frame_probs.size(); ++i) {
            result += std::to_string(frame_probs[i]);
            if (i < frame_probs.size() - 1) {
                result += ", ";
            }
        }
        result += "}, \n{";
        for (size_t i = 0; i < scan_indices.size(); ++i) {
            result += std::to_string(scan_indices[i]);
            if (i < scan_indices.size() - 1) {
                result += ", ";
            }
        }
        result += "}, \n{";
        for (size_t i = 0; i < scan_probs.size(); ++i) {
            result += std::to_string(scan_probs[i]);
            if (i < scan_probs.size() - 1) {
                result += ", ";
            }
        }
        result += "}, \n{";
        for (size_t i = 0; i < tof_indices.size(); ++i) {
            result += std::to_string(tof_indices[i]);
            if (i < tof_indices.size() - 1) {
                result += ", ";
            }
        }
        result += "}, \n{";
        for (size_t i = 0; i < tof_probs.size(); ++i) {
            result += std::to_string(tof_probs[i]);
            if (i < tof_probs.size() - 1) {
                result += ", ";
            }
        }
        result += "})";
        return result;
    }
};


uint32_t find_one(const std::span<int>& span)
{
    auto it = std::find(span.begin(), span.end(), 1);
    if (it != span.end()) 
        return static_cast<uint32_t>(std::distance(span.begin(), it));
    else 
        throw std::runtime_error("No '1' found in the span. This shouldn't happen.");
}

void worker(std::atomic<size_t> &n_processed,
            const std::vector<ProblematicInput> &inputs,
            std::mutex &out_file_lock,
            std::array<std::ofstream, 5> &out_files,
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
        std::unique_ptr<int[]> configuration = std::make_unique<int[]>(input.frame_indices.size() + input.scan_indices.size() + input.tof_indices.size());

        IsoSpec::Iso iso(3, isotopeNumbers.data(), atomCounts.data(), isotopeMasses.data(), isotopeProbabilities.data());
        IsoSpec::IsoStochasticGenerator generator(std::move(iso), input.N, input.precision);


        // TODO: we can replace std::vector here by preallocated memory of size N
        std::vector<uint32_t> ClusterIds;
        std::vector<uint32_t> frame_indices;
        std::vector<uint32_t> scan_indices;
        std::vector<uint32_t> tof_indices;
        std::vector<uint32_t> intensity;

        size_t total_confs = input.N;
        while (generator.advanceToNextConfiguration() and total_confs >= input.N_minimal) {
            ClusterIds.push_back(static_cast<uint32_t>(index));
            const size_t curr_intensity = static_cast<size_t>(generator.prob());
            intensity.push_back(curr_intensity);
            total_confs -= curr_intensity;
            generator.get_conf_signature(configuration.get());
            frame_indices.push_back(find_one(std::span<int>(configuration.get(), input.frame_indices.size())));
            scan_indices.push_back(find_one(std::span<int>(configuration.get() + input.frame_indices.size(), input.scan_indices.size())));
            tof_indices.push_back(find_one(std::span<int>(configuration.get() + input.frame_indices.size() + input.scan_indices.size(), input.tof_indices.size())));
        }
        // Write results to the output file
        {
            std::lock_guard<std::mutex> lock(out_file_lock);
            //out.write(reinterpret_cast<const char*>(masses.data()), masses.size() * sizeof(double));
            //out.write(reinterpret_cast<const char*>(probabilities.data()), probabilities.size() * sizeof(double));
            out_files[0].write(reinterpret_cast<const char*>(ClusterIds.data()), ClusterIds.size() * sizeof(uint32_t));
            out_files[1].write(reinterpret_cast<const char*>(frame_indices.data()), frame_indices.size() * sizeof(uint32_t));
            out_files[2].write(reinterpret_cast<const char*>(scan_indices.data()), scan_indices.size() * sizeof(uint32_t));
            out_files[3].write(reinterpret_cast<const char*>(tof_indices.data()), tof_indices.size() * sizeof(uint32_t));
            out_files[4].write(reinterpret_cast<const char*>(intensity.data()), intensity.size() * sizeof(uint32_t));
        }
        free(isotopeMasses[0]);
        free(isotopeMasses[1]);
        free(isotopeMasses[2]);
    }
}

void Massimize(const std::vector<ProblematicInput> &inputs, size_t n_threads, const std::string &output_dir_path)
{
    std::filesystem::path output_dir(output_dir_path);
    std::filesystem::create_directory(output_dir);
    {
        std::ofstream schema_file;
        schema_file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
        schema_file.open(output_dir / "schema.txt");
        schema_file << "uint32 ClusterID" << std::endl;
        schema_file << "uint32 frame" << std::endl;
        schema_file << "uint32 scan" << std::endl;
        schema_file << "uint32 tof" << std::endl;
        schema_file << "uint32 intensity" << std::endl;
    }
    std::array<std::ofstream, 5> out_files;
    for(size_t ii = 0; ii < 5; ++ii)
    {
        out_files[ii].exceptions(std::ofstream::failbit | std::ofstream::badbit);
        out_files[ii].open(output_dir / (std::to_string(ii) + ".bin"), std::ios::binary);
    }

    std::mutex out_file_lock;
    std::atomic<size_t> n_processed(0);
    std::vector<std::thread> threads;

    if(n_threads == 0) {
        worker(n_processed, inputs, out_file_lock, out_files, 0);
        return;
    }


    for(size_t ii = 0; ii < n_threads; ++ii) {
        threads.emplace_back(worker, std::ref(n_processed), std::ref(inputs), std::ref(out_file_lock), std::ref(out_files), ii);
    }

    for(auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}



