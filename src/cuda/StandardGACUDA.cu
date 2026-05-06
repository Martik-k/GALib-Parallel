// GPU kernels for StandardGACUDA.
// These device functions are reserved for future GPU-accelerated evaluation.
// The StandardGACUDA class interface is defined in:
//   include/algorithms/standard/StandardGACUDA.h

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace {

using Scalar = float;
constexpr int kBlockSize = 256;

inline int divUp(int n, int d) { return (n + d - 1) / d; }

enum class ProblemId : int {
    Sphere    = 0,
    Rastrigin = 1,
    HeavyTrig  = 2
};

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
        constexpr int repeat_count = 64;
        Scalar sum = 0.0f;
        for (int j = 0; j < dimensions; ++j) {
            const Scalar x = genes[j];
            Scalar state = x;
            for (int r = 1; r <= repeat_count; ++r) {
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

// ── Stateless RNG (splitmix64) ───────────────────────────────────────────────

__device__ inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x  = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x  = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ inline Scalar uniformHash(uint64_t seed) {
    return static_cast<Scalar>(static_cast<double>(splitmix64(seed) >> 11) * (1.0 / (double)(1ULL << 53)));
}

__device__ inline Scalar normalHash(uint64_t seed) {
    const Scalar u1 = fmaxf(uniformHash(seed), 1e-15f);
    const Scalar u2 = uniformHash(splitmix64(seed ^ 0x1234567890ABCDEFULL));
    constexpr Scalar TWO_PI = 6.283185307179586476925f;
    return sqrtf(-2.0f * logf(u1)) * cosf(TWO_PI * u2);
}

__device__ inline int tournamentSelectSL(const Scalar* fitness, int populationSize,
                                         int tournamentSize, uint64_t seed, int slot) {
    uint64_t h0 = splitmix64(seed ^ (static_cast<uint64_t>(slot) << 32));
    int winner   = static_cast<int>(h0 % static_cast<uint64_t>(populationSize));
    Scalar wFit  = fitness[winner];
    for (int k = 1; k < tournamentSize; ++k) {
        uint64_t hk = splitmix64(seed ^ (static_cast<uint64_t>(slot) << 32)
                                       ^ static_cast<uint64_t>(k));
        int candidate   = static_cast<int>(hk % static_cast<uint64_t>(populationSize));
        Scalar cFit     = fitness[candidate];
        if (cFit < wFit) { winner = candidate; wFit = cFit; }
    }
    return winner;
}

// ── Kernels ──────────────────────────────────────────────────────────────────

__global__ void initPopulation(Scalar* population, int populationSize, int dimensions,
                               Scalar lowerBound, Scalar upperBound, uint64_t baseSeed) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize) return;
    const int    base = idx * dimensions;
    const Scalar span = upperBound - lowerBound;
    for (int j = 0; j < dimensions; ++j) {
        const uint64_t seed = splitmix64(baseSeed ^ splitmix64(
            static_cast<uint64_t>(idx) * 100003ULL + static_cast<uint64_t>(j)));
        population[base + j] = lowerBound + uniformHash(seed) * span;
    }
}

__global__ void evaluatePopulation(const Scalar* population, Scalar* fitness,
                                   int populationSize, int dimensions, ProblemId problem) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize) return;
    fitness[idx] = evaluateIndividual(&population[idx * dimensions], dimensions, problem);
}

__global__ void precomputePairs(const Scalar* fitness, int2* pairs, int* splits,
                                int populationSize, int dimensions,
                                int elitismOffset, int tournamentSize,
                                Scalar crossoverRate, uint64_t baseSeed, uint64_t generation) {
    const int pairIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (elitismOffset + pairIndex * 2 >= populationSize) return;
    const uint64_t seed = splitmix64(baseSeed ^ splitmix64(
        generation * 65537ULL + static_cast<uint64_t>(pairIndex)));
    const int p1 = tournamentSelectSL(fitness, populationSize, tournamentSize, seed, 0);
    const int p2 = tournamentSelectSL(fitness, populationSize, tournamentSize, seed, 1);
    pairs[pairIndex] = make_int2(p1, p2);
    int split = -1;
    if (dimensions > 1 && uniformHash(seed ^ 0xFEEDBEEFULL) < crossoverRate) {
        split = 1 + static_cast<int>(
            splitmix64(seed ^ 0xCAFEBABEULL) % static_cast<uint64_t>(dimensions - 1));
    }
    splits[pairIndex] = split;
}

__global__ void evolveGenes(const Scalar* population, Scalar* nextPopulation,
                            const int2* pairs, const int* splits,
                            int populationSize, int dimensions,
                            int elitismOffset, Scalar mutationRate, Scalar sigma,
                            uint64_t baseSeed, uint64_t generation) {
    const int gid            = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalChildGenes = (populationSize - elitismOffset) * dimensions;
    if (gid >= totalChildGenes) return;
    const int childLocal  = gid / dimensions;
    const int gene        = gid % dimensions;
    const int pairIndex   = childLocal / 2;
    const int childInPair = childLocal & 1;
    const int2 pair  = pairs[pairIndex];
    const int  split = splits[pairIndex];
    const bool useP1 = (split < 0) ? (childInPair == 0)
                                   : ((childInPair == 0) ? (gene < split) : (gene >= split));
    const int  parentIdx = useP1 ? pair.x : pair.y;
    Scalar     geneVal   = population[parentIdx * dimensions + gene];
    const int  childGlobal = elitismOffset + childLocal;
    const uint64_t mutSeed = splitmix64(baseSeed ^ splitmix64(
        generation * static_cast<uint64_t>(populationSize) * static_cast<uint64_t>(dimensions)
        + static_cast<uint64_t>(childGlobal) * static_cast<uint64_t>(dimensions)
        + static_cast<uint64_t>(gene)));
    if (uniformHash(mutSeed) < mutationRate)
        geneVal += normalHash(mutSeed ^ 0xDEADBEEFCAFEBABEULL) * sigma;
    nextPopulation[childGlobal * dimensions + gene] = geneVal;
}

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
    if (tid == 0) { partialFitness[blockIdx.x] = sFit[0]; partialIndex[blockIdx.x] = sIdx[0]; }
}

__global__ void finalizeBest(const Scalar* partialFitness, const int* partialIndex,
                             int partialCount, Scalar* bestFitness, int* bestIndex) {
    extern __shared__ char smem[];
    Scalar* sFit = reinterpret_cast<Scalar*>(smem);
    int*    sIdx = reinterpret_cast<int*>(smem + blockDim.x * sizeof(Scalar));
    const int tid = threadIdx.x;
    Scalar localBest = 1e30f; int localIndex = -1;
    for (int i = tid; i < partialCount; i += blockDim.x) {
        const Scalar f = partialFitness[i];
        if (f < localBest) { localBest = f; localIndex = partialIndex[i]; }
    }
    sFit[tid] = localBest; sIdx[tid] = localIndex;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride && sFit[tid + stride] < sFit[tid]) {
            sFit[tid] = sFit[tid + stride]; sIdx[tid] = sIdx[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) { *bestFitness = sFit[0]; *bestIndex = sIdx[0]; }
}

__global__ void copyElite(const Scalar* population, Scalar* nextPopulation,
                          const int* bestIndex, int dimensions) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= dimensions) return;
    nextPopulation[j] = population[(*bestIndex) * dimensions + j];
}

} // namespace

// ── Full GPU evolution bridge ─────────────────────────────────────────────────

#include "algorithms/standard/CUDAEvaluator.h"
#include <chrono>
#include <iostream>
#include <omp.h>
#include <vector>

namespace galib::cuda::internal {

static bool tryParseProblemId(const std::string& name, ProblemId& out) {
    if (name == "Sphere")    { out = ProblemId::Sphere;    return true; }
    if (name == "Rastrigin") { out = ProblemId::Rastrigin; return true; }
    if (name == "HeavyTrig") { out = ProblemId::HeavyTrig; return true; }
    return false;
}

// Copy population from device, evaluate via CPU callback, upload fitness to device.
static bool cpuEvaluateAndUpload(
    const float* d_pop, float* d_fitness,
    int pop_size, int dims,
    std::vector<float>& h_pop_tmp, std::vector<float>& h_fit_tmp,
    const std::function<double(const float*, int)>& cb)
{
    const std::size_t genes_bytes   = static_cast<std::size_t>(pop_size) * dims * sizeof(float);
    const std::size_t fitness_bytes = static_cast<std::size_t>(pop_size) * sizeof(float);

    if (cudaMemcpy(h_pop_tmp.data(), d_pop, genes_bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < pop_size; ++i)
        h_fit_tmp[static_cast<std::size_t>(i)] =
            static_cast<float>(cb(&h_pop_tmp[static_cast<std::size_t>(i) * dims], dims));

    return cudaMemcpy(d_fitness, h_fit_tmp.data(), fitness_bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

bool runCUDAEvolution(CUDARunParams& params,
                      std::vector<float>& h_genes,
                      std::vector<float>& h_fitness) {
    const int P = params.pop_size;
    const int D = params.dims;

    ProblemId problem = ProblemId::Sphere;
    const bool use_gpu_fitness = tryParseProblemId(params.problem_name, problem);

    const std::size_t genes_bytes   = static_cast<std::size_t>(P) * D * sizeof(float);
    const std::size_t fitness_bytes = static_cast<std::size_t>(P) * sizeof(float);
    const int eval_blocks           = divUp(P, kBlockSize);
    const int pair_count            = (P + 1) / 2;
    const std::size_t smem          = static_cast<std::size_t>(kBlockSize) * (sizeof(float) + sizeof(int));

    // ── Device buffers ────────────────────────────────────────────────────────
    float* d_pop          = nullptr;
    float* d_next_pop     = nullptr;
    float* d_fitness      = nullptr;
    int2*  d_pairs        = nullptr;
    int*   d_splits       = nullptr;
    float* d_best_fit     = nullptr;
    int*   d_best_idx     = nullptr;
    float* d_partial_fit  = nullptr;
    int*   d_partial_idx  = nullptr;

    const std::size_t partial_bytes_f = static_cast<std::size_t>(eval_blocks) * sizeof(float);
    const std::size_t partial_bytes_i = static_cast<std::size_t>(eval_blocks) * sizeof(int);

    bool alloc_ok =
        cudaMalloc(&d_pop,         genes_bytes)                                   == cudaSuccess &&
        cudaMalloc(&d_next_pop,    genes_bytes)                                   == cudaSuccess &&
        cudaMalloc(&d_fitness,     fitness_bytes)                                 == cudaSuccess &&
        cudaMalloc(&d_pairs,       static_cast<std::size_t>(pair_count) * sizeof(int2)) == cudaSuccess &&
        cudaMalloc(&d_splits,      static_cast<std::size_t>(pair_count) * sizeof(int))  == cudaSuccess &&
        cudaMalloc(&d_best_fit,    sizeof(float))                                 == cudaSuccess &&
        cudaMalloc(&d_best_idx,    sizeof(int))                                   == cudaSuccess &&
        cudaMalloc(&d_partial_fit, partial_bytes_f)                               == cudaSuccess &&
        cudaMalloc(&d_partial_idx, partial_bytes_i)                               == cudaSuccess;

    auto cleanup = [&] {
        cudaFree(d_pop); cudaFree(d_next_pop); cudaFree(d_fitness);
        cudaFree(d_pairs); cudaFree(d_splits);
        cudaFree(d_best_fit); cudaFree(d_best_idx);
        cudaFree(d_partial_fit); cudaFree(d_partial_idx);
    };

    if (!alloc_ok) {
        std::cerr << "[GALib CUDA] Device allocation failed.\n";
        cleanup(); return false;
    }

    if (cudaMemcpy(d_pop, h_genes.data(), genes_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "[GALib CUDA] Initial H2D upload failed.\n";
        cleanup(); return false;
    }

    // CPU scratch (only allocated when fitness runs on CPU)
    std::vector<float> h_pop_tmp(use_gpu_fitness ? 0 : static_cast<std::size_t>(P) * D);
    std::vector<float> h_fit_tmp(use_gpu_fitness ? 0 : static_cast<std::size_t>(P));

    const uint64_t base_seed = static_cast<uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()) ^ 0xA5A5A5A5ULL;

    // ── Initial evaluation ────────────────────────────────────────────────────
    if (use_gpu_fitness) {
        evaluatePopulation<<<eval_blocks, kBlockSize>>>(d_pop, d_fitness, P, D, problem);
        if (cudaDeviceSynchronize() != cudaSuccess) { cleanup(); return false; }
    } else {
        if (!cpuEvaluateAndUpload(d_pop, d_fitness, P, D, h_pop_tmp, h_fit_tmp, params.fitness_callback))
            { cleanup(); return false; }
    }

    // ── Generation loop ───────────────────────────────────────────────────────
    for (int gen = 0; gen < params.max_generations; ++gen) {

        // Find best individual (for elitism + progress)
        findBestBlock<<<eval_blocks, kBlockSize, smem>>>(d_fitness, P, d_partial_fit, d_partial_idx);
        finalizeBest <<<1,           kBlockSize, smem>>>(d_partial_fit, d_partial_idx, eval_blocks,
                                                          d_best_fit, d_best_idx);
        if (cudaDeviceSynchronize() != cudaSuccess) { cleanup(); return false; }

        if (params.progress_callback) {
            float best_f = 0.0f;
            cudaMemcpy(&best_f, d_best_fit, sizeof(float), cudaMemcpyDeviceToHost);
            params.progress_callback(gen, static_cast<double>(best_f));
        }

        // Elitism: copy best to slot 0 of next population
        int elitism_offset = 0;
        if (params.use_elitism) {
            elitism_offset = 1;
            copyElite<<<divUp(D, kBlockSize), kBlockSize>>>(d_pop, d_next_pop, d_best_idx, D);
        }

        // Phase A: tournament selection + crossover planning (one thread per pair)
        const int generated  = P - elitism_offset;
        const int pair_cnt   = (generated + 1) / 2;
        precomputePairs<<<divUp(pair_cnt, kBlockSize), kBlockSize>>>(
            d_fitness, d_pairs, d_splits,
            P, D, elitism_offset,
            params.tournament_size,
            params.crossover_rate,
            base_seed, static_cast<uint64_t>(gen));

        // Phase B: crossover + mutation (one thread per gene)
        evolveGenes<<<divUp(generated * D, kBlockSize), kBlockSize>>>(
            d_pop, d_next_pop, d_pairs, d_splits,
            P, D, elitism_offset,
            params.mutation_rate, params.sigma,
            base_seed, static_cast<uint64_t>(gen));

        if (cudaDeviceSynchronize() != cudaSuccess) { cleanup(); return false; }

        std::swap(d_pop, d_next_pop);

        // Evaluate new population
        if (use_gpu_fitness) {
            evaluatePopulation<<<eval_blocks, kBlockSize>>>(d_pop, d_fitness, P, D, problem);
            if (cudaDeviceSynchronize() != cudaSuccess) { cleanup(); return false; }
        } else {
            if (!cpuEvaluateAndUpload(d_pop, d_fitness, P, D, h_pop_tmp, h_fit_tmp, params.fitness_callback))
                { cleanup(); return false; }
        }
    }

    // ── Copy results back to host ─────────────────────────────────────────────
    h_genes.resize(static_cast<std::size_t>(P) * D);
    h_fitness.resize(static_cast<std::size_t>(P));
    const bool ok =
        cudaMemcpy(h_genes.data(),   d_pop,     genes_bytes,   cudaMemcpyDeviceToHost) == cudaSuccess &&
        cudaMemcpy(h_fitness.data(), d_fitness, fitness_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;

    cleanup();
    return ok;
}

} // namespace galib::cuda::internal
