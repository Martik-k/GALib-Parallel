#include <chrono>
#include <iostream>
#include <memory>

#include "algorithms/standard/StandardGA.h"
#include "benchmarks/RastriginFunction.h"
#include "core/Population.h"
#include "operators/crossover/SinglePointCrossover.h"
#include "operators/mutation/GaussianMutation.h"
#include "operators/selection/TournamentSelection.h"

#ifdef GALIB_WITH_CUDA
#include "algorithms/standard/StandardGACUDA.h"
#endif

using namespace galib;

static constexpr std::size_t NUM_GENES       = 10;
static constexpr std::size_t POP_SIZE        = 100;
static constexpr std::size_t MAX_GENERATIONS = 200;
static constexpr double      MUTATION_RATE   = 0.05;
static constexpr double      CROSSOVER_RATE  = 0.8;
static constexpr double      LOWER_BOUND     = -5.12;
static constexpr double      UPPER_BOUND     =  5.12;

static auto makeOperators() {
    struct Ops {
        std::unique_ptr<Selection<double>>  selection;
        std::unique_ptr<Mutation<double>>   mutation;
        std::unique_ptr<Crossover<double>>  crossover;
    };
    return Ops{
        std::make_unique<TournamentSelection<double>>(3),
        std::make_unique<GaussianMutation<double>>(0.1),
        std::make_unique<SinglePointCrossover<double>>()
    };
}

static void runTest(const std::string& label,
                    Algorithm<double>& algo,
                    benchmark::RastriginFunction<double>& fitness_fn) {
    Population<double> population(POP_SIZE, NUM_GENES);
    population.initialize(LOWER_BOUND, UPPER_BOUND);

    std::cout << "\n=== " << label << " ===\n";
    auto t0 = std::chrono::steady_clock::now();
    algo.run(population);
    auto t1 = std::chrono::steady_clock::now();

    const auto& best = population.getBestIndividual();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Best fitness : " << best.getFitness() << "\n";
    std::cout << "Best genotype: " << best << "\n";
    std::cout << "Wall time    : " << ms << " ms\n";

    double expected_min = 0.0;
    bool   converged    = best.getFitness() < 5.0;
    std::cout << "Result       : " << (converged ? "PASS (converged near optimum)" : "WARN (did not converge)") << "\n";
}

int main() {
    benchmark::RastriginFunction<double> fitness_fn(NUM_GENES, LOWER_BOUND, UPPER_BOUND);

    std::cout << "Fitness function : Rastrigin (" << NUM_GENES << "D)\n";
    std::cout << "Population size  : " << POP_SIZE << "\n";
    std::cout << "Max generations  : " << MAX_GENERATIONS << "\n";

    // ── StandardGA ───────────────────────────────────────────────────────────
    {
        auto ops = makeOperators();
        StandardGA<double> ga(
            fitness_fn,
            std::move(ops.selection),
            std::move(ops.mutation),
            std::move(ops.crossover),
            MUTATION_RATE, CROSSOVER_RATE, MAX_GENERATIONS, /*elitism=*/true
        );
        runTest("StandardGA (CPU)", ga, fitness_fn);
    }

#ifdef GALIB_WITH_CUDA
    // ── StandardGACUDA ───────────────────────────────────────────────────────
    {
        auto ops = makeOperators();
        cuda::StandardGACUDA<double> ga_cuda(
            fitness_fn,
            std::move(ops.selection),
            std::move(ops.mutation),
            std::move(ops.crossover),
            MUTATION_RATE, CROSSOVER_RATE, MAX_GENERATIONS, /*elitism=*/true
        );
        runTest("StandardGACUDA  (CPU ops + CUDA build)", ga_cuda, fitness_fn);
    }
#else
    std::cout << "\n=== StandardGACUDA === SKIPPED (not built with GALIB_WITH_CUDA)\n";
#endif

    return 0;
}
