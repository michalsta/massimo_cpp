#include <IsoSpec++/isoSpec++.h>
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
#include <optional>
#include <random>

#include "concurrency.hpp"

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
    std::optional<uint_fast32_t> seed;
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
          tof_indices(tof_indices), tof_probs(tof_probs), seed(std::nullopt)
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

    void set_seed(uint_fast32_t new_seed) {
        seed = new_seed;
    }

    std::optional<uint_fast32_t> get_seed() const {
        return seed;
    }
};

class ProblematicOutput
{
public:
    std::vector<uint32_t> ClusterIds;
    std::vector<uint32_t> frame_indices;
    std::vector<uint32_t> scan_indices;
    std::vector<uint32_t> tof_indices;
    std::vector<uint32_t> intensity;
    ProblematicOutput() = default;
    ProblematicOutput(const ProblematicOutput & other) = delete;
    ProblematicOutput(ProblematicOutput && other) noexcept
        : ClusterIds(std::move(other.ClusterIds)), frame_indices(std::move(other.frame_indices)),
          scan_indices(std::move(other.scan_indices)), tof_indices(std::move(other.tof_indices)),
          intensity(std::move(other.intensity)) {};
    ProblematicOutput& operator=(const ProblematicOutput & other) = delete;
    ProblematicOutput& operator=(ProblematicOutput && other) = delete;
    ~ProblematicOutput() = default;
};

uint32_t find_one(const std::span<int>& span)
{
    auto it = std::find(span.begin(), span.end(), 1);
    if (it != span.end())
        return static_cast<uint32_t>(std::distance(span.begin(), it));
    else
        throw std::runtime_error("No '1' found in the span. This shouldn't happen.");
}

void writer_thread(std::array<std::ofstream, 5> &out_files,
                   SynchronizedBuffer<std::unique_ptr<ProblematicOutput>> &output_buffer)
{
    while (true) {
        auto output_opt = output_buffer.pop();
        if (!output_opt.has_value()) {
            break; // Buffer is closed and empty, exit the thread
        }
        const ProblematicOutput &output = *(output_opt.value());
        out_files[0].write(reinterpret_cast<const char*>(output.ClusterIds.data()), output.ClusterIds.size() * sizeof(uint32_t));
        out_files[1].write(reinterpret_cast<const char*>(output.frame_indices.data()), output.frame_indices.size() * sizeof(uint32_t));
        out_files[2].write(reinterpret_cast<const char*>(output.scan_indices.data()), output.scan_indices.size() * sizeof(uint32_t));
        out_files[3].write(reinterpret_cast<const char*>(output.tof_indices.data()), output.tof_indices.size() * sizeof(uint32_t));
        out_files[4].write(reinterpret_cast<const char*>(output.intensity.data()), output.intensity.size() * sizeof(uint32_t));
    }
}

template<typename StochasticGeneratorBackend>
void worker(std::atomic<size_t> &n_processed,
            const std::vector<ProblematicInput> &inputs,
            SynchronizedBuffer<std::unique_ptr<ProblematicOutput>> &output_buffer,
            size_t thread_id,
            double beta_bias)
{
    std::mt19937 rng(std::random_device{}());

    while(n_processed < inputs.size())
    {
        size_t index = n_processed.fetch_add(1);
        if (index >= inputs.size())
            break; // No more inputs to process

        const ProblematicInput &input = inputs[index];

        if(input.get_seed().has_value())
            rng.seed(input.get_seed().value());

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

        constexpr bool compact_confs = std::is_same<StochasticGeneratorBackend, IsoSpec::IsoOrderedGeneratorTemplate<IsoSpec::SingleAtomMarginal<false>>>::value ||
                                        std::is_same<StochasticGeneratorBackend, IsoSpec::IsoLayeredGeneratorTemplate<IsoSpec::SingleAtomMarginal<true>>>::value;

        const size_t conf_size = compact_confs ? 3 : input.frame_indices.size() + input.scan_indices.size() + input.tof_indices.size();
        std::unique_ptr<int[]> configuration = std::make_unique<int[]>(conf_size);

        IsoSpec::Iso iso(3, isotopeNumbers.data(), atomCounts.data(), isotopeMasses.data(), isotopeProbabilities.data());
        IsoSpec::IsoStochasticGeneratorTemplate<StochasticGeneratorBackend> generator(std::move(iso), input.N, input.precision, beta_bias, rng);


        // TODO: we can replace std::vector here by preallocated memory of size N
        std::unique_ptr<ProblematicOutput> output = std::make_unique<ProblematicOutput>();
        std::vector<uint32_t> frame_indices;
        std::vector<uint32_t> scan_indices;
        std::vector<uint32_t> tof_indices;
        std::vector<uint32_t> intensity;
        std::vector<size_t> order;

        size_t total_confs = input.N;
        while (generator.advanceToNextConfiguration() and total_confs >= input.N_minimal) {
            const size_t curr_intensity = generator.count();
            total_confs -= curr_intensity;
            if (curr_intensity < input.N_minimal)
                continue;
            output->ClusterIds.push_back(static_cast<uint32_t>(index));
            intensity.push_back(curr_intensity);
            if constexpr (compact_confs) {
                generator.get_indexes(configuration.get());
                frame_indices.push_back(static_cast<uint32_t>(configuration[0]));
                scan_indices.push_back(static_cast<uint32_t>(configuration[1]));
                tof_indices.push_back(static_cast<uint32_t>(configuration[2]));
            } else {
                generator.get_conf_signature(configuration.get());
                frame_indices.push_back(input.frame_indices[find_one(std::span<int>(configuration.get(), input.frame_indices.size()))]);
                scan_indices.push_back(input.scan_indices[find_one(std::span<int>(configuration.get() + input.frame_indices.size(), input.scan_indices.size()))]);
                tof_indices.push_back(input.tof_indices[find_one(std::span<int>(configuration.get() + input.frame_indices.size() + input.scan_indices.size(), input.tof_indices.size()))]);
            }
            order.push_back(order.size());
        }

        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            if (frame_indices[a] != frame_indices[b])
                return frame_indices[a] < frame_indices[b];
            if (scan_indices[a] != scan_indices[b])
                return scan_indices[a] < scan_indices[b];
            if (tof_indices[a] != tof_indices[b])
                return tof_indices[a] < tof_indices[b];
            return false;
        });

        output->frame_indices.reserve(frame_indices.size());
        output->scan_indices.reserve(scan_indices.size());
        output->tof_indices.reserve(tof_indices.size());
        output->intensity.reserve(intensity.size());

        for (size_t idx : order) {
            output->frame_indices.push_back(frame_indices[idx]);
            output->scan_indices.push_back(scan_indices[idx]);
            output->tof_indices.push_back(tof_indices[idx]);
            output->intensity.push_back(intensity[idx]);
        }

        // Push the output to the synchronized buffer
        output_buffer.push(std::move(output));

        free(isotopeMasses[0]);
        free(isotopeMasses[1]);
        free(isotopeMasses[2]);
    }
}

template<typename StochasticGeneratorBackend>
void Massimize(std::vector<ProblematicInput> &inputs, const std::string &output_dir_path, size_t n_threads, double beta_bias = 5.0, std::optional<uint_fast32_t> seed = std::nullopt)
{
    std::mt19937 rng;
    if (seed.has_value())
        rng.seed(seed.value());
    else
        rng.seed(std::random_device{}());

    for(auto &input : inputs)
        input.set_seed(rng());

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

    SynchronizedBuffer<std::unique_ptr<ProblematicOutput>> out_files_buffer(n_threads * 3 + 10); // TODO: tune the buffer size
    std::atomic<size_t> n_processed(0);
    std::vector<std::thread> threads;
    std::thread writer(writer_thread, std::ref(out_files), std::ref(out_files_buffer));

    if(n_threads == 0) {
        worker<StochasticGeneratorBackend>(n_processed, inputs, out_files_buffer, 0, beta_bias);
        return;
    }


    for(size_t ii = 0; ii < n_threads; ++ii) {
        threads.emplace_back(worker<StochasticGeneratorBackend>, std::ref(n_processed), std::ref(inputs), std::ref(out_files_buffer), ii, beta_bias);
    }

    for(auto &thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    out_files_buffer.close();
    if (writer.joinable()) {
        writer.join();
    }
}



