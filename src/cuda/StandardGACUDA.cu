#include "algorithms/StandardGACUDA.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kBlockSize = 256;

enum class ProblemId : int {
    Sphere = 0,
    Rastrigin = 1
};

inline bool tryParseProblem(const std::string& name, ProblemId& problem) {
    if (name == "Sphere") {
        problem = ProblemId::Sphere;
        return true;
    }
    if (name == "Rastrigin") {
        problem = ProblemId::Rastrigin;
        return true;
    }
    return false;
}

inline int divUp(int n, int d) {
    return (n + d - 1) / d;
}

__device__ inline double evaluateIndividual(const double* genes, int dimensions, ProblemId problem) {
    if (problem == ProblemId::Sphere) {
        double sum = 0.0;
        for (int j = 0; j < dimensions; ++j) {
            const double x = genes[j];
            sum += x * x;
        }
        return sum;
    }

    constexpr double A = 10.0;
    constexpr double PI = 3.14159265358979323846;
    double sum = A * static_cast<double>(dimensions);

    for (int j = 0; j < dimensions; ++j) {
        const double x = genes[j];
        sum += x * x - A * cos(2.0 * PI * x);
    }
    return sum;
}

__device__ inline int tournamentSelect(const double* fitness, int populationSize,
                                       int tournamentSize, curandStatePhilox4_32_10_t& state) {
    int winner = static_cast<int>(curand(&state) % static_cast<unsigned int>(populationSize));
    double winnerFitness = fitness[winner];

    for (int i = 1; i < tournamentSize; ++i) {
        const int candidate = static_cast<int>(curand(&state) % static_cast<unsigned int>(populationSize));
        const double candidateFitness = fitness[candidate];
        if (candidateFitness < winnerFitness) {
            winner = candidate;
            winnerFitness = candidateFitness;
        }
    }

    return winner;
}

__global__ void initRng(curandStatePhilox4_32_10_t* states, std::uint64_t seed, int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void initPopulation(double* population, int populationSize, int dimensions,
                               double lowerBound, double upperBound,
                               curandStatePhilox4_32_10_t* states) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize) {
        return;
    }

    curandStatePhilox4_32_10_t localState = states[idx];
    const int base = idx * dimensions;
    const double span = upperBound - lowerBound;

    for (int j = 0; j < dimensions; ++j) {
        const double u = static_cast<double>(curand_uniform_double(&localState));
        population[base + j] = lowerBound + u * span;
    }

    states[idx] = localState;
}

__global__ void evaluatePopulation(const double* population, double* fitness,
                                   int populationSize, int dimensions, ProblemId problem) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize) {
        return;
    }

    const int base = idx * dimensions;
    fitness[idx] = evaluateIndividual(&population[base], dimensions, problem);
}

__global__ void evolvePopulation(const double* population, const double* fitness,
                                 double* nextPopulation, int populationSize, int dimensions,
                                 int elitismOffset, int tournamentSize,
                                 double crossoverRate, double mutationRate,
                                 curandStatePhilox4_32_10_t* states) {
    const int pairIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int firstChildIndex = elitismOffset + pairIndex * 2;

    if (firstChildIndex >= populationSize) {
        return;
    }

    curandStatePhilox4_32_10_t localState = states[pairIndex];

    const int parent1 = tournamentSelect(fitness, populationSize, tournamentSize, localState);
    const int parent2 = tournamentSelect(fitness, populationSize, tournamentSize, localState);

    const bool doCrossover = curand_uniform_double(&localState) < crossoverRate;
    int split = dimensions;
    if (doCrossover && dimensions > 1) {
        split = 1 + static_cast<int>(curand(&localState) % static_cast<unsigned int>(dimensions - 1));
    }

    const int firstBase = firstChildIndex * dimensions;
    const int secondChildIndex = firstChildIndex + 1;
    const int secondBase = secondChildIndex * dimensions;
    const int parent1Base = parent1 * dimensions;
    const int parent2Base = parent2 * dimensions;

    for (int j = 0; j < dimensions; ++j) {
        double child1 = population[parent1Base + j];
        double child2 = population[parent2Base + j];

        if (doCrossover && j >= split) {
            child1 = population[parent2Base + j];
            child2 = population[parent1Base + j];
        }

        if (curand_uniform_double(&localState) < mutationRate) {
            child1 += curand_normal_double(&localState) * mutationRate;
        }

        nextPopulation[firstBase + j] = child1;

        if (secondChildIndex < populationSize) {
            if (curand_uniform_double(&localState) < mutationRate) {
                child2 += curand_normal_double(&localState) * mutationRate;
            }
            nextPopulation[secondBase + j] = child2;
        }
    }

    states[pairIndex] = localState;
}

inline bool checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << "CUDA error in " << operation << ": " << cudaGetErrorString(status) << std::endl;
    return false;
}

inline bool evaluatePopulationHost(
    const double* dPopulation,
    double* dFitness,
    int populationSize,
    int dimensions,
    const galib::FitnessFunction<double>& fitness_function,
    std::vector<double>& hPopulation,
    std::vector<double>& hFitness,
    const std::size_t genesBytes,
    const std::size_t fitnessBytes
) {
    if (!checkCuda(cudaMemcpy(hPopulation.data(), dPopulation, genesBytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy population D2H (host fitness eval)")) {
        return false;
    }

    for (int i = 0; i < populationSize; ++i) {
        const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(dimensions);
        std::vector<double> genotype(static_cast<std::size_t>(dimensions));
        for (int j = 0; j < dimensions; ++j) {
            genotype[static_cast<std::size_t>(j)] = hPopulation[base + static_cast<std::size_t>(j)];
        }
        hFitness[static_cast<std::size_t>(i)] = fitness_function.evaluate(genotype);
    }

    if (!checkCuda(cudaMemcpy(dFitness, hFitness.data(), fitnessBytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy fitness H2D (host fitness eval)")) {
        return false;
    }

    return true;
}

} // namespace

namespace galib {
namespace cuda {

StandardGACUDA::StandardGACUDA(const StandardGACUDAConfig& config)
    : config_m(config) {}

StandardGACUDA::StandardGACUDA(const StandardGACUDAConfig& config, const FitnessFunction<double>& fitness_function)
    : config_m(config), fitness_function_m(&fitness_function) {}

StandardGACUDAConfig StandardGACUDA::fromConfig(const utils::Config& config) {
    StandardGACUDAConfig cuda_config;
    cuda_config.population_size = config.algorithm.pop_size;
    cuda_config.dimensions = config.problem.dimensions;
    cuda_config.max_generations = config.algorithm.max_generations;
    cuda_config.mutation_rate = config.algorithm.mutation_rate;
    cuda_config.crossover_rate = config.algorithm.crossover_rate;
    cuda_config.tournament_size = config.algorithm.selection.tournament_size;
    cuda_config.lower_bound = config.problem.lower_bound;
    cuda_config.upper_bound = config.problem.upper_bound;
    cuda_config.problem_name = config.problem.name;
    cuda_config.log_file = config.output.log_file;
    return cuda_config;
}

bool StandardGACUDA::run(Population<double>& population) const {
    const int populationSize = static_cast<int>(config_m.population_size);
    const int dimensions = static_cast<int>(config_m.dimensions);

    if (populationSize <= 0 || dimensions <= 0) {
        std::cerr << "Invalid CUDA GA dimensions or population size." << std::endl;
        return false;
    }

    ProblemId problem = ProblemId::Sphere;
    const bool use_device_fitness = tryParseProblem(config_m.problem_name, problem);
    if (!use_device_fitness && fitness_function_m == nullptr) {
        std::cerr << "CUDA backend has no device evaluator for '" << config_m.problem_name
                  << "'. Pass a FitnessFunction to StandardGACUDA to enable generic host fitness evaluation."
                  << std::endl;
        return false;
    }

    const std::size_t genesCount = static_cast<std::size_t>(populationSize) * static_cast<std::size_t>(dimensions);
    const std::size_t genesBytes = genesCount * sizeof(double);
    const std::size_t fitnessBytes = static_cast<std::size_t>(populationSize) * sizeof(double);

    double* dPopulation = nullptr;
    double* dNextPopulation = nullptr;
    double* dFitness = nullptr;
    curandStatePhilox4_32_10_t* dStates = nullptr;

    const int numPairs = (populationSize + 1) / 2;
    const int evolveStates = std::max(numPairs, populationSize);

    if (!checkCuda(cudaMalloc(&dPopulation, genesBytes), "cudaMalloc(dPopulation)") ||
        !checkCuda(cudaMalloc(&dNextPopulation, genesBytes), "cudaMalloc(dNextPopulation)") ||
        !checkCuda(cudaMalloc(&dFitness, fitnessBytes), "cudaMalloc(dFitness)") ||
        !checkCuda(cudaMalloc(&dStates, static_cast<std::size_t>(evolveStates) * sizeof(curandStatePhilox4_32_10_t)),
                   "cudaMalloc(dStates)")) {
        cudaFree(dPopulation);
        cudaFree(dNextPopulation);
        cudaFree(dFitness);
        cudaFree(dStates);
        return false;
    }

    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const std::uint64_t seed = static_cast<std::uint64_t>(now) ^ 0xA5A5A5A5ULL;

    initRng<<<divUp(evolveStates, kBlockSize), kBlockSize>>>(dStates, seed, evolveStates);
    if (!checkCuda(cudaGetLastError(), "initRng launch") || !checkCuda(cudaDeviceSynchronize(), "initRng sync")) {
        cudaFree(dPopulation);
        cudaFree(dNextPopulation);
        cudaFree(dFitness);
        cudaFree(dStates);
        return false;
    }

    initPopulation<<<divUp(populationSize, kBlockSize), kBlockSize>>>(
        dPopulation,
        populationSize,
        dimensions,
        config_m.lower_bound,
        config_m.upper_bound,
        dStates);
    if (!checkCuda(cudaGetLastError(), "initPopulation launch") ||
        !checkCuda(cudaDeviceSynchronize(), "initPopulation sync")) {
        cudaFree(dPopulation);
        cudaFree(dNextPopulation);
        cudaFree(dFitness);
        cudaFree(dStates);
        return false;
    }

    std::ofstream log;
    if (!config_m.log_file.empty()) {
        std::filesystem::path p(config_m.log_file);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }
        log.open(config_m.log_file);
        if (log.is_open()) {
            log << "generation,individual_idx,x,y\n";
        } else {
            std::cerr << "Error: Could not open log file " << config_m.log_file << std::endl;
        }
    }

    std::vector<double> hFitness(static_cast<std::size_t>(populationSize));
    std::vector<double> hPopulation;
    if (log.is_open() || !use_device_fitness) {
        hPopulation.resize(genesCount);
    }

    for (std::size_t generation = 0; generation < config_m.max_generations; ++generation) {
        if (use_device_fitness) {
            evaluatePopulation<<<divUp(populationSize, kBlockSize), kBlockSize>>>(
                dPopulation, dFitness, populationSize, dimensions, problem);
            if (!checkCuda(cudaGetLastError(), "evaluatePopulation launch") ||
                !checkCuda(cudaDeviceSynchronize(), "evaluatePopulation sync")) {
                cudaFree(dPopulation);
                cudaFree(dNextPopulation);
                cudaFree(dFitness);
                cudaFree(dStates);
                return false;
            }
        } else {
            if (!evaluatePopulationHost(
                    dPopulation,
                    dFitness,
                    populationSize,
                    dimensions,
                    *fitness_function_m,
                    hPopulation,
                    hFitness,
                    genesBytes,
                    fitnessBytes)) {
                cudaFree(dPopulation);
                cudaFree(dNextPopulation);
                cudaFree(dFitness);
                cudaFree(dStates);
                return false;
            }
        }

        if (!checkCuda(cudaMemcpy(hFitness.data(), dFitness, fitnessBytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy fitness D2H")) {
            cudaFree(dPopulation);
            cudaFree(dNextPopulation);
            cudaFree(dFitness);
            cudaFree(dStates);
            return false;
        }

        int bestIndex = 0;
        double bestFitness = hFitness[0];
        for (int i = 1; i < populationSize; ++i) {
            if (hFitness[i] < bestFitness) {
                bestFitness = hFitness[i];
                bestIndex = i;
            }
        }

        if (log.is_open()) {
            if (!checkCuda(cudaMemcpy(hPopulation.data(), dPopulation, genesBytes, cudaMemcpyDeviceToHost),
                           "cudaMemcpy population D2H")) {
                cudaFree(dPopulation);
                cudaFree(dNextPopulation);
                cudaFree(dFitness);
                cudaFree(dStates);
                return false;
            }

            for (int i = 0; i < populationSize; ++i) {
                const int base = i * dimensions;
                const double x = hPopulation[base];
                const double y = (dimensions > 1) ? hPopulation[base + 1] : 0.0;
                log << generation << "," << i << "," << x << "," << y << "\n";
            }
        }

        int elitismOffset = 0;
        if (config_m.use_elitism) {
            elitismOffset = 1;
            if (!checkCuda(cudaMemcpy(dNextPopulation,
                                      dPopulation + static_cast<std::size_t>(bestIndex) * dimensions,
                                      static_cast<std::size_t>(dimensions) * sizeof(double),
                                      cudaMemcpyDeviceToDevice),
                           "cudaMemcpy elitism D2D")) {
                cudaFree(dPopulation);
                cudaFree(dNextPopulation);
                cudaFree(dFitness);
                cudaFree(dStates);
                return false;
            }
        }

        const int generated = populationSize - elitismOffset;
        const int pairCount = (generated + 1) / 2;
        evolvePopulation<<<divUp(pairCount, kBlockSize), kBlockSize>>>(
            dPopulation,
            dFitness,
            dNextPopulation,
            populationSize,
            dimensions,
            elitismOffset,
            std::max(1, config_m.tournament_size),
            config_m.crossover_rate,
            config_m.mutation_rate,
            dStates);

        if (!checkCuda(cudaGetLastError(), "evolvePopulation launch") ||
            !checkCuda(cudaDeviceSynchronize(), "evolvePopulation sync")) {
            cudaFree(dPopulation);
            cudaFree(dNextPopulation);
            cudaFree(dFitness);
            cudaFree(dStates);
            return false;
        }

        std::swap(dPopulation, dNextPopulation);

        if ((generation + 1) % 50 == 0 || generation == 0) {
            std::cout << "Generation " << (generation + 1)
                      << " | Best Fitness: " << bestFitness << std::endl;
        }
    }

    if (use_device_fitness) {
        evaluatePopulation<<<divUp(populationSize, kBlockSize), kBlockSize>>>(
            dPopulation, dFitness, populationSize, dimensions, problem);
        if (!checkCuda(cudaGetLastError(), "final evaluatePopulation launch") ||
            !checkCuda(cudaDeviceSynchronize(), "final evaluatePopulation sync")) {
            cudaFree(dPopulation);
            cudaFree(dNextPopulation);
            cudaFree(dFitness);
            cudaFree(dStates);
            return false;
        }
    } else {
        if (!evaluatePopulationHost(
                dPopulation,
                dFitness,
                populationSize,
                dimensions,
                *fitness_function_m,
                hPopulation,
                hFitness,
                genesBytes,
                fitnessBytes)) {
            cudaFree(dPopulation);
            cudaFree(dNextPopulation);
            cudaFree(dFitness);
            cudaFree(dStates);
            return false;
        }
    }

    std::vector<double> finalPopulation(genesCount);
    std::vector<double> finalFitness(static_cast<std::size_t>(populationSize));

    if (!checkCuda(cudaMemcpy(finalPopulation.data(), dPopulation, genesBytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy final population D2H") ||
        !checkCuda(cudaMemcpy(finalFitness.data(), dFitness, fitnessBytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy final fitness D2H")) {
        cudaFree(dPopulation);
        cudaFree(dNextPopulation);
        cudaFree(dFitness);
        cudaFree(dStates);
        return false;
    }

    for (int i = 0; i < populationSize; ++i) {
        std::vector<double>& genotype = population[static_cast<std::size_t>(i)].getGenotype();
        const std::size_t base = static_cast<std::size_t>(i) * static_cast<std::size_t>(dimensions);
        for (int j = 0; j < dimensions; ++j) {
            genotype[static_cast<std::size_t>(j)] = finalPopulation[base + static_cast<std::size_t>(j)];
        }
        population[static_cast<std::size_t>(i)].setFitness(finalFitness[static_cast<std::size_t>(i)]);
    }

    if (log.is_open()) {
        log.close();
    }

    cudaFree(dPopulation);
    cudaFree(dNextPopulation);
    cudaFree(dFitness);
    cudaFree(dStates);

    return true;
}

bool runStandardGACUDA(
    const utils::Config& config,
    Population<double>& population,
    const FitnessFunction<double>& fitness_function
) {
    StandardGACUDA ga(StandardGACUDA::fromConfig(config), fitness_function);
    return ga.run(population);
}

} // namespace cuda
} // namespace galib
