#include "algorithms/StandardGACUDA.h"

#include <cuda_runtime.h>

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
using Scalar = float;

enum class ProblemId : int {
    Sphere    = 0,
    Rastrigin = 1,
    HeavyTrig  = 2
};

inline bool tryParseProblem(const std::string& name, ProblemId& problem) {
    if (name == "Sphere")    { problem = ProblemId::Sphere;    return true; }
    if (name == "Rastrigin") { problem = ProblemId::Rastrigin; return true; }
    if (name == "HeavyTrig") { problem = ProblemId::HeavyTrig;  return true; }
    return false;
}

inline int divUp(int n, int d) { return (n + d - 1) / d; }

// ── Device fitness evaluation ────────────────────────────────────────────────

__device__ inline Scalar evaluateIndividual(const Scalar* genes, int dimensions, ProblemId problem) {
    if (problem == ProblemId::Sphere) {
        Scalar sum = 0.0f;
        for (int j = 0; j < dimensions; ++j) {
            const Scalar x = genes[j];
            sum += x * x;
        }
        return sum;
    }

    if (problem == ProblemId::Rastrigin) {
        constexpr Scalar A  = 10.0f;
        constexpr Scalar PI = 3.14159265358979323846f;
        Scalar sum = A * static_cast<Scalar>(dimensions);
        for (int j = 0; j < dimensions; ++j) {
            const Scalar x = genes[j];
            sum += x * x - A * cosf(2.0f * PI * x);
        }
        return sum;
    }

    if (problem == ProblemId::HeavyTrig) {
        constexpr std::size_t repeat_count = 64;
        Scalar sum = 0.0f;
        for (int j = 0; j < dimensions; ++j) {
            const Scalar x = genes[j];
            Scalar state = x;
            for (std::size_t r = 1; r <= repeat_count; ++r) {
                const Scalar factor = static_cast<Scalar>(r);
                const Scalar angle = factor * 0.017f + x * 0.031f;
                state = sinf(state + angle)
                      + cosf(state - angle * 0.5f)
                      + expf(-0.001f * state * state)
                      + 0.0005f * state * state;
            }
            sum += state * state + 0.1f * x * x;
        }
        return sum;
    }

    return 0.0f;
}

// ── Stateless RNG (splitmix64 hash) ─────────────────────────────────────────
//
// Replaces per-thread Philox states entirely.
// Benefits:
//   • Zero state memory (no dStates allocation).
//   • Zero initialisation time (eliminates the ~470 ms initRng startup).
//   • Quality sufficient for GA: splitmix64 passes BigCrush.

__device__ inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x  = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x  = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Returns a uniform double in [0, 1).
__device__ inline Scalar uniformHash(std::uint64_t seed) {
    return static_cast<Scalar>(static_cast<double>(splitmix64(seed) >> 11) * (1.0 / static_cast<double>(1ULL << 53)));
}

// Returns a standard-normal sample via Box-Muller.
__device__ inline Scalar normalHash(std::uint64_t seed) {
    const Scalar u1 = fmaxf(uniformHash(seed),                         1e-15f);
    const Scalar u2 =       uniformHash(splitmix64(seed ^ 0x1234567890ABCDEFULL));
    constexpr Scalar TWO_PI = 6.283185307179586476925f;
    return sqrtf(-2.0f * logf(u1)) * cosf(TWO_PI * u2);
}

// Tournament selection using stateless RNG.
// slot = 0 for parent-1, slot = 1 for parent-2 — ensures independent draws.
__device__ inline int tournamentSelectSL(const Scalar* fitness, int populationSize,
                                         int tournamentSize,
                                         std::uint64_t seed, int slot) {
    // First candidate
    std::uint64_t h0 = splitmix64(seed ^ (static_cast<std::uint64_t>(slot) << 32));
    int winner       = static_cast<int>(h0 % static_cast<std::uint64_t>(populationSize));
    Scalar winnerFit = fitness[winner];

    for (int k = 1; k < tournamentSize; ++k) {
        std::uint64_t hk = splitmix64(seed ^ (static_cast<std::uint64_t>(slot) << 32)
                                           ^ static_cast<std::uint64_t>(k));
        int candidate       = static_cast<int>(hk % static_cast<std::uint64_t>(populationSize));
        Scalar candidateFit = fitness[candidate];
        if (candidateFit < winnerFit) {
            winner    = candidate;
            winnerFit = candidateFit;
        }
    }
    return winner;
}

// ── Kernels ──────────────────────────────────────────────────────────────────

// Fills population with uniform random values — stateless (no curand states).
__global__ void initPopulation(Scalar* population, int populationSize, int dimensions,
                               Scalar lowerBound, Scalar upperBound,
                               std::uint64_t baseSeed) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize) return;

    const int    base = idx * dimensions;
    const Scalar span = upperBound - lowerBound;

    for (int j = 0; j < dimensions; ++j) {
        // Unique seed per (individual, gene): mix in individual and gene indices.
        const std::uint64_t seed =
            splitmix64(baseSeed ^ splitmix64(static_cast<std::uint64_t>(idx) * 100003ULL
                                             + static_cast<std::uint64_t>(j)));
        population[base + j] = lowerBound + uniformHash(seed) * span;
    }
}

// One thread per individual: evaluates device fitness.
__global__ void evaluatePopulation(const Scalar* population, Scalar* fitness,
                                   int populationSize, int dimensions, ProblemId problem) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize) return;
    fitness[idx] = evaluateIndividual(&population[idx * dimensions], dimensions, problem);
}

// ── Two-phase evolution ───────────────────────────────────────────────────────
//
// Phase A  precomputePairs  (pop/2 threads, 1 per pair)
//   Runs tournament selection and decides the crossover split for every parent
//   pair.  Writes results to small dPairs / dSplits arrays.
//   RNG cost per thread: tournamentSize×2 + 2 draws — cheap.
//
// Phase B  evolveGenes  (pop × dim threads, 1 per gene)
//   Applies crossover using the precomputed pair data and performs per-gene
//   mutation with stateless RNG.  Launching one thread per gene saturates the
//   GPU even for small populations: e.g. pop=50 000, dim=100 → 5 M threads.
//
// Together these replace the old single-phase evolvePopulation which launched
// only pop/2 threads and forced each thread to loop over all D genes serially.

// Phase A: select parents and compute crossover split point.
// dPairs[i]  = {parent1_index, parent2_index}
// dSplits[i] = crossover split in [1, dim-1], or -1 for no crossover.
__global__ void precomputePairs(const Scalar* fitness,
                                int2*         pairs,
                                int*          splits,
                                int populationSize, int dimensions,
                                int elitismOffset, int tournamentSize,
                                Scalar crossoverRate,
                                std::uint64_t baseSeed, std::uint64_t generation) {
    const int pairIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (elitismOffset + pairIndex * 2 >= populationSize) return;

    // Per-(generation, pair) seed.
    const std::uint64_t seed =
        splitmix64(baseSeed ^ splitmix64(generation * 65537ULL
                                         + static_cast<std::uint64_t>(pairIndex)));

    const int p1 = tournamentSelectSL(fitness, populationSize, tournamentSize, seed, 0);
    const int p2 = tournamentSelectSL(fitness, populationSize, tournamentSize, seed, 1);
    pairs[pairIndex] = make_int2(p1, p2);

    int split = -1;
    if (dimensions > 1 && uniformHash(seed ^ 0xFEEDBEEFULL) < crossoverRate) {
        split = 1 + static_cast<int>(
            splitmix64(seed ^ 0xCAFEBABEULL) % static_cast<std::uint64_t>(dimensions - 1));
    }
    splits[pairIndex] = split;
}

// Phase B: one thread per (child, gene) — fully parallel gene evolution.
__global__ void evolveGenes(const Scalar* population, Scalar* nextPopulation,
                            const int2* pairs, const int* splits,
                            int populationSize, int dimensions,
                            int elitismOffset, Scalar mutationRate,
                            std::uint64_t baseSeed, std::uint64_t generation) {
    const int gid            = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalChildGenes = (populationSize - elitismOffset) * dimensions;
    if (gid >= totalChildGenes) return;

    const int childLocal  = gid / dimensions;   // 0-based child index (elitism slot excluded)
    const int gene        = gid % dimensions;
    const int pairIndex   = childLocal / 2;
    const int childInPair = childLocal & 1;      // 0 = first child, 1 = second child

    const int2 pair  = pairs[pairIndex];
    const int  split = splits[pairIndex];

    // Crossover: child-0 inherits p1 before split, child-1 inherits p2 before split.
    const bool useP1 = (split < 0) ? (childInPair == 0)
                                   : ((childInPair == 0) ? (gene < split) : (gene >= split));

    const int parentIdx = useP1 ? pair.x : pair.y;
    Scalar    geneVal   = population[parentIdx * dimensions + gene];

    // Stateless mutation: unique seed per (generation, childGlobal, gene).
    const int  childGlobal = elitismOffset + childLocal;
    const std::uint64_t mutSeed =
        splitmix64(baseSeed ^ splitmix64(
            generation * static_cast<std::uint64_t>(populationSize)
                       * static_cast<std::uint64_t>(dimensions)
          + static_cast<std::uint64_t>(childGlobal) * static_cast<std::uint64_t>(dimensions)
          + static_cast<std::uint64_t>(gene)));

    if (uniformHash(mutSeed) < mutationRate) {
        geneVal += normalHash(mutSeed ^ 0xDEADBEEFCAFEBABEULL) * mutationRate;
    }

    nextPopulation[childGlobal * dimensions + gene] = geneVal;
}

// Parallel reduction: each block finds its local minimum fitness and index.
// A CPU pass over the (numBlocks) partial results finishes the reduction.
// This replaces the full D2H fitness copy + CPU linear scan done every generation.
__global__ void findBestBlock(const Scalar* fitness, int populationSize,
                              Scalar* partialFitness, int* partialIndex) {
    extern __shared__ char smem[];
    Scalar* sFit = reinterpret_cast<Scalar*>(smem);
    int*    sIdx = reinterpret_cast<int*>(smem + blockDim.x * sizeof(Scalar));

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    sFit[tid] = (gid < populationSize) ? fitness[gid] : 1e30f;
    sIdx[tid] = (gid < populationSize) ? gid          : -1;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && sFit[tid + stride] < sFit[tid]) {
            sFit[tid] = sFit[tid + stride];
            sIdx[tid] = sIdx[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partialFitness[blockIdx.x] = sFit[0];
        partialIndex[blockIdx.x]   = sIdx[0];
    }
}

// Finalize min reduction on device in a single block so host does not need
// to synchronize/copy partial arrays each generation.
__global__ void finalizeBest(const Scalar* partialFitness, const int* partialIndex,
                             int partialCount,
                             Scalar* bestFitness, int* bestIndex) {
    extern __shared__ char smem[];
    Scalar* sFit = reinterpret_cast<Scalar*>(smem);
    int*    sIdx = reinterpret_cast<int*>(smem + blockDim.x * sizeof(Scalar));

    const int tid = threadIdx.x;

    Scalar localBest = 1e30f;
    int localIndex = -1;
    for (int i = tid; i < partialCount; i += blockDim.x) {
        const Scalar f = partialFitness[i];
        if (f < localBest) {
            localBest = f;
            localIndex = partialIndex[i];
        }
    }

    sFit[tid] = localBest;
    sIdx[tid] = localIndex;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && sFit[tid + stride] < sFit[tid]) {
            sFit[tid] = sFit[tid + stride];
            sIdx[tid] = sIdx[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *bestFitness = sFit[0];
        *bestIndex = sIdx[0];
    }
}

// Copy elite individual from current population to next population slot 0.
__global__ void copyElite(const Scalar* population, Scalar* nextPopulation,
                          const int* bestIndex, int dimensions) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= dimensions) {
        return;
    }
    const int idx = *bestIndex;
    nextPopulation[j] = population[idx * dimensions + j];
}

// ── Host helpers ─────────────────────────────────────────────────────────────

inline bool checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    std::cerr << "CUDA error in " << operation << ": "
              << cudaGetErrorString(status) << std::endl;
    return false;
}

inline bool evaluatePopulationHost(
    const Scalar* dPopulation, Scalar* dFitness,
        int populationSize, int dimensions,
        const galib::FitnessFunction<double>& fitness_function,
    std::vector<Scalar>& hPopulation, std::vector<Scalar>& hFitness,
        std::size_t genesBytes, std::size_t fitnessBytes) {
    if (!checkCuda(cudaMemcpy(hPopulation.data(), dPopulation, genesBytes,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy population D2H (host fitness eval)"))
        return false;

    for (int i = 0; i < populationSize; ++i) {
        const std::size_t base = static_cast<std::size_t>(i)
                               * static_cast<std::size_t>(dimensions);
        std::vector<double> genotype(static_cast<std::size_t>(dimensions));
        for (int j = 0; j < dimensions; ++j)
            genotype[static_cast<std::size_t>(j)] = static_cast<double>(hPopulation[base + static_cast<std::size_t>(j)]);
        hFitness[static_cast<std::size_t>(i)] = static_cast<Scalar>(fitness_function.evaluate(genotype));
    }

    if (!checkCuda(cudaMemcpy(dFitness, hFitness.data(), fitnessBytes,
                              cudaMemcpyHostToDevice),
                   "cudaMemcpy fitness H2D (host fitness eval)"))
        return false;

    return true;
}

} // namespace

namespace galib {
namespace cuda {

StandardGACUDA::StandardGACUDA(const StandardGACUDAConfig& config)
    : config_m(config) {}

StandardGACUDA::StandardGACUDA(const StandardGACUDAConfig& config,
                               const FitnessFunction<double>& fitness_function)
    : config_m(config), fitness_function_m(&fitness_function) {}

StandardGACUDAConfig StandardGACUDA::fromConfig(const utils::Config& config) {
    StandardGACUDAConfig c;
    c.population_size  = config.algorithm.pop_size;
    c.dimensions       = config.problem.dimensions;
    c.max_generations  = config.algorithm.max_generations;
    c.mutation_rate    = config.algorithm.mutation_rate;
    c.crossover_rate   = config.algorithm.crossover_rate;
    c.tournament_size  = config.algorithm.selection.tournament_size;
    c.lower_bound      = config.problem.lower_bound;
    c.upper_bound      = config.problem.upper_bound;
    c.problem_name     = config.problem.name;
    c.log_file         = config.output.log_file;
    return c;
}

bool StandardGACUDA::run(Population<double>& population) const {
    const int populationSize = static_cast<int>(config_m.population_size);
    const int dimensions     = static_cast<int>(config_m.dimensions);

    if (populationSize <= 0 || dimensions <= 0) {
        std::cerr << "Invalid CUDA GA dimensions or population size." << std::endl;
        return false;
    }

    ProblemId problem = ProblemId::Sphere;
    const bool use_device_fitness = tryParseProblem(config_m.problem_name, problem);
    if (!use_device_fitness && fitness_function_m == nullptr) {
        std::cerr << "CUDA backend has no device evaluator for '" << config_m.problem_name
                  << "'. Pass a FitnessFunction to StandardGACUDA." << std::endl;
        return false;
    }

    const std::size_t genesCount   = static_cast<std::size_t>(populationSize)
                                   * static_cast<std::size_t>(dimensions);
    const std::size_t genesBytes   = genesCount * sizeof(Scalar);
    const std::size_t fitnessBytes = static_cast<std::size_t>(populationSize) * sizeof(Scalar);

    const int numEvalBlocks          = divUp(populationSize, kBlockSize);
    const std::size_t partialFitnessBytes = static_cast<std::size_t>(numEvalBlocks) * sizeof(Scalar);
    const std::size_t partialIndexBytes   = static_cast<std::size_t>(numEvalBlocks) * sizeof(int);

    const int numPairs = (populationSize + 1) / 2;

    // ── Device buffers ────────────────────────────────────────────────────────
    Scalar* dPopulation     = nullptr;
    Scalar* dNextPopulation = nullptr;
    Scalar* dFitness        = nullptr;
    Scalar* dPartialFitness = nullptr;
    int*    dPartialIndex   = nullptr;
    Scalar* dBestFitness    = nullptr;
    int*    dBestIndex      = nullptr;
    int2*   dPairs          = nullptr;  // parent index pairs  (numPairs entries)
    int*    dSplits         = nullptr;  // crossover split pts (numPairs entries)

    if (!checkCuda(cudaMalloc(&dPopulation,     genesBytes),          "cudaMalloc(dPopulation)")     ||
        !checkCuda(cudaMalloc(&dNextPopulation, genesBytes),          "cudaMalloc(dNextPopulation)") ||
        !checkCuda(cudaMalloc(&dFitness,        fitnessBytes),        "cudaMalloc(dFitness)")        ||
        !checkCuda(cudaMalloc(&dPartialFitness, partialFitnessBytes), "cudaMalloc(dPartialFitness)") ||
        !checkCuda(cudaMalloc(&dPartialIndex,   partialIndexBytes),   "cudaMalloc(dPartialIndex)")   ||
        !checkCuda(cudaMalloc(&dBestFitness,    sizeof(Scalar)),       "cudaMalloc(dBestFitness)")    ||
        !checkCuda(cudaMalloc(&dBestIndex,      sizeof(int)),          "cudaMalloc(dBestIndex)")      ||
        !checkCuda(cudaMalloc(&dPairs,  static_cast<std::size_t>(numPairs) * sizeof(int2)), "cudaMalloc(dPairs)")  ||
        !checkCuda(cudaMalloc(&dSplits, static_cast<std::size_t>(numPairs) * sizeof(int)),  "cudaMalloc(dSplits)")) {
        cudaFree(dPopulation); cudaFree(dNextPopulation); cudaFree(dFitness);
        cudaFree(dPartialFitness); cudaFree(dPartialIndex);
        cudaFree(dBestFitness); cudaFree(dBestIndex);
        cudaFree(dPairs); cudaFree(dSplits);
        return false;
    }

    auto cleanup = [&]() {
        cudaFree(dPopulation); cudaFree(dNextPopulation); cudaFree(dFitness);
        cudaFree(dPartialFitness); cudaFree(dPartialIndex);
        cudaFree(dBestFitness); cudaFree(dBestIndex);
        cudaFree(dPairs); cudaFree(dSplits);
    };

    // ── Initialise population (stateless — no initRng required) ─────────────
    const auto     now     = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const std::uint64_t seed = static_cast<std::uint64_t>(now) ^ 0xA5A5A5A5ULL;

    initPopulation<<<divUp(populationSize, kBlockSize), kBlockSize>>>(
        dPopulation, populationSize, dimensions,
        static_cast<Scalar>(config_m.lower_bound), static_cast<Scalar>(config_m.upper_bound), seed);
    if (!checkCuda(cudaGetLastError(),          "initPopulation launch") ||
        !checkCuda(cudaDeviceSynchronize(),     "initPopulation sync")) {
        cleanup(); return false;
    }

    // ── Logging setup ─────────────────────────────────────────────────────────
    std::ofstream log;
    if (!config_m.log_file.empty()) {
        std::filesystem::path p(config_m.log_file);
        if (p.has_parent_path())
            std::filesystem::create_directories(p.parent_path());
        log.open(config_m.log_file);
        if (log.is_open())
            log << "generation,individual_idx,x,y\n";
        else
            std::cerr << "Error: Could not open log file " << config_m.log_file << std::endl;
    }

    std::vector<Scalar> hFitness;
    std::vector<Scalar> hPopulation;
    if (!use_device_fitness) {
        hFitness.resize(static_cast<std::size_t>(populationSize));
        hPopulation.resize(genesCount);
    } else if (log.is_open()) {
        hPopulation.resize(genesCount);
    }

    const std::size_t   smemSize = static_cast<std::size_t>(kBlockSize) * (sizeof(Scalar) + sizeof(int));

    // ── Main generation loop ──────────────────────────────────────────────────
    for (std::size_t generation = 0; generation < config_m.max_generations; ++generation) {

        // ── EVALUATE ────────────────────────────────────────────────────────
        if (use_device_fitness) {
            evaluatePopulation<<<divUp(populationSize, kBlockSize), kBlockSize>>>(
                dPopulation, dFitness, populationSize, dimensions, problem);
            if (!checkCuda(cudaGetLastError(), "evaluatePopulation launch")) {
                cleanup(); return false;
            }
            // Stream ordering: findBestBlock (next) waits for evaluatePopulation.
        } else {
            if (!evaluatePopulationHost(dPopulation, dFitness, populationSize, dimensions,
                                        *fitness_function_m, hPopulation, hFitness,
                                        genesBytes, fitnessBytes)) {
                cleanup(); return false;
            }
        }

        // ── FIND BEST (on-device reduction) ─────────────────────────────────
        //  Replaces full D2H + CPU scan.  Transfers only ~numEvalBlocks entries.
        findBestBlock<<<numEvalBlocks, kBlockSize, smemSize>>>(
            dFitness, populationSize, dPartialFitness, dPartialIndex);
        if (!checkCuda(cudaGetLastError(), "findBestBlock launch")) {
            cleanup(); return false;
        }

        finalizeBest<<<1, kBlockSize, smemSize>>>(
            dPartialFitness, dPartialIndex, numEvalBlocks, dBestFitness, dBestIndex);
        if (!checkCuda(cudaGetLastError(), "finalizeBest launch")) {
            cleanup(); return false;
        }

        // ── LOGGING ─────────────────────────────────────────────────────────
        if (log.is_open()) {
            if (!checkCuda(cudaMemcpy(hPopulation.data(), dPopulation,
                                      genesBytes, cudaMemcpyDeviceToHost),
                           "cudaMemcpy population D2H")) {
                cleanup(); return false;
            }
            for (int i = 0; i < populationSize; ++i) {
                const int    base = i * dimensions;
                const double x    = hPopulation[static_cast<std::size_t>(base)];
                const double y    = (dimensions > 1)
                                  ? hPopulation[static_cast<std::size_t>(base + 1)] : 0.0;
                log << generation << "," << i << "," << x << "," << y << "\n";
            }
        }

        // ── ELITISM ─────────────────────────────────────────────────────────
        int elitismOffset = 0;
        if (config_m.use_elitism) {
            elitismOffset = 1;
            copyElite<<<divUp(dimensions, kBlockSize), kBlockSize>>>(
                dPopulation, dNextPopulation, dBestIndex, dimensions);
            if (!checkCuda(cudaGetLastError(), "copyElite launch")) {
                cleanup(); return false;
            }
        }

        // ── EVOLVE (two-phase) ───────────────────────────────────────────────
        const int generated       = populationSize - elitismOffset;
        const int pairCount       = (generated + 1) / 2;
        const int totalChildGenes = generated * dimensions;

        // Phase A: tournament selection + crossover planning (pop/2 threads).
        precomputePairs<<<divUp(pairCount, kBlockSize), kBlockSize>>>(
            dFitness, dPairs, dSplits,
            populationSize, dimensions, elitismOffset,
            std::max(1, config_m.tournament_size),
            config_m.crossover_rate,
            seed, static_cast<std::uint64_t>(generation));
        if (!checkCuda(cudaGetLastError(), "precomputePairs launch")) {
            cleanup(); return false;
        }

        // Phase B: gene-level crossover + mutation (pop × dim threads).
        // Launches ~pop*dim threads — fully saturates the GPU for large problems.
        evolveGenes<<<divUp(totalChildGenes, kBlockSize), kBlockSize>>>(
            dPopulation, dNextPopulation,
            dPairs, dSplits,
            populationSize, dimensions, elitismOffset,
            config_m.mutation_rate,
            seed, static_cast<std::uint64_t>(generation));
        if (!checkCuda(cudaGetLastError(), "evolveGenes launch")) {
            cleanup(); return false;
        }
        // No explicit sync: next generation's D2H partial copy acts as barrier.

        std::swap(dPopulation, dNextPopulation);

        if ((generation + 1) % 50 == 0 || generation == 0) {
            std::cout << "Generation " << (generation + 1)
                      << " | CUDA generation completed" << std::endl;
        }
    }

    // ── FINAL EVALUATION ──────────────────────────────────────────────────────
    if (use_device_fitness) {
        evaluatePopulation<<<divUp(populationSize, kBlockSize), kBlockSize>>>(
            dPopulation, dFitness, populationSize, dimensions, problem);
        if (!checkCuda(cudaGetLastError(),      "final evaluatePopulation launch") ||
            !checkCuda(cudaDeviceSynchronize(), "final evaluatePopulation sync")) {
            cleanup(); return false;
        }
    } else {
        if (!evaluatePopulationHost(dPopulation, dFitness, populationSize, dimensions,
                                    *fitness_function_m, hPopulation, hFitness,
                                    genesBytes, fitnessBytes)) {
            cleanup(); return false;
        }
    }

    std::vector<Scalar> finalPopulation(genesCount);
    std::vector<Scalar> finalFitness(static_cast<std::size_t>(populationSize));

    if (!checkCuda(cudaMemcpy(finalPopulation.data(), dPopulation,
                              genesBytes, cudaMemcpyDeviceToHost),
                   "final population D2H") ||
        !checkCuda(cudaMemcpy(finalFitness.data(), dFitness,
                              fitnessBytes, cudaMemcpyDeviceToHost),
                   "final fitness D2H")) {
        cleanup(); return false;
    }

    for (int i = 0; i < populationSize; ++i) {
        std::vector<double>& genotype =
            population[static_cast<std::size_t>(i)].getGenotype();
        const std::size_t base = static_cast<std::size_t>(i)
                               * static_cast<std::size_t>(dimensions);
        for (int j = 0; j < dimensions; ++j)
            genotype[static_cast<std::size_t>(j)] =
                static_cast<double>(finalPopulation[base + static_cast<std::size_t>(j)]);
        population[static_cast<std::size_t>(i)].setFitness(
            static_cast<double>(finalFitness[static_cast<std::size_t>(i)]));
    }

    if (log.is_open()) log.close();
    cleanup();
    return true;
}

bool runStandardGACUDA(const utils::Config& config, Population<double>& population,
                       const FitnessFunction<double>& fitness_function) {
    StandardGACUDA ga(StandardGACUDA::fromConfig(config), fitness_function);
    return ga.run(population);
}

} // namespace cuda
} // namespace galib
