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
                            int elitismOffset, Scalar mutationRate,
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
        geneVal += normalHash(mutSeed ^ 0xDEADBEEFCAFEBABEULL) * mutationRate;
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
