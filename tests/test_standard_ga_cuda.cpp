#include <gtest/gtest.h>

#include <filesystem>
#include <limits>
#include <memory>
#include <vector>

#include "algorithms/standard/StandardGA.h"
#include "benchmarks/RastriginFunction.h"
#include "benchmarks/SphereFunction.h"
#include "core/FitnessFunction.h"
#include "core/Population.h"
#include "operators/crossover/SinglePointCrossover.h"
#include "operators/mutation/GaussianMutation.h"
#include "operators/selection/TournamentSelection.h"
#include "utils/FunctionalFitness.h"

#ifdef GALIB_WITH_CUDA
#include "algorithms/standard/StandardGACUDA.h"
#endif

using namespace galib;
using namespace galib::benchmark;

// ─── Shared operator bundle ────────────────────────────────────────────────

struct Operators {
    std::unique_ptr<Selection<double>>  sel;
    std::unique_ptr<Mutation<double>>   mu;
    std::unique_ptr<Crossover<double>>  cs;
};

static Operators makeOperators() {
    return {
        std::make_unique<TournamentSelection<double>>(3),
        std::make_unique<GaussianMutation<double>>(0.1),
        std::make_unique<SinglePointCrossover<double>>()
    };
}

// ─── Test fixture ──────────────────────────────────────────────────────────

class StandardGACUDATest : public ::testing::Test {
protected:
    static constexpr std::size_t NUM_GENES = 10;
    static constexpr std::size_t POP_SIZE  = 100;
    static constexpr double LB = -5.12;
    static constexpr double UB =  5.12;

    static Population<double> makePopulation(std::size_t pop_size = POP_SIZE) {
        Population<double> pop(pop_size, NUM_GENES);
        pop.initialize(LB, UB);
        return pop;
    }

    // Returns the best fitness across the population without modifying it.
    static double bestFitnessOf(Population<double>& pop, FitnessFunction<double>& ff) {
        double best = std::numeric_limits<double>::max();
        for (std::size_t i = 0; i < pop.size(); ++i)
            best = std::min(best, ff.evaluate(pop[i].getGenotype()));
        return best;
    }

    void TearDown() override {
        if (std::filesystem::exists("test_output.csv"))
            std::filesystem::remove("test_output.csv");
    }
};

// ─── CUDA tests (compiled only when CUDA toolkit is present) ───────────────

#ifdef GALIB_WITH_CUDA

// 1. The constructor accepts valid parameters without throwing.
TEST_F(StandardGACUDATest, ConstructionSucceeds) {
    SphereFunction<double> ff(NUM_GENES, LB, UB);
    auto ops = makeOperators();
    EXPECT_NO_THROW({
        cuda::StandardGACUDA<double> ga(
            ff, std::move(ops.sel), std::move(ops.mu), std::move(ops.cs),
            0.05, 0.8, 10, true
        );
    });
}

// 2. run() completes without crashing when a custom FitnessFunction forces the
//    CPU callback path (name() == "" → no GPU kernel dispatch).
TEST_F(StandardGACUDATest, RunCompletesWithCustomFitnessFunction) {
    FunctionalFitness<double> ff(NUM_GENES, LB, UB,
        [](const std::vector<double>& x) {
            double s = 0.0;
            for (double v : x) s += v * v;
            return s;
        }
    );
    auto ops = makeOperators();
    cuda::StandardGACUDA<double> ga(
        ff, std::move(ops.sel), std::move(ops.mu), std::move(ops.cs),
        0.05, 0.8, 20, true
    );
    auto pop = makePopulation();
    ASSERT_NO_THROW(ga.run(pop));
    EXPECT_EQ(pop.size(), POP_SIZE);
    EXPECT_EQ(pop.getNumGenes(), NUM_GENES);
}

// 3. CPU callback path (custom FitnessFunction) converges: fitness improves
//    over 150 generations on a simple quadratic.
TEST_F(StandardGACUDATest, CPUCallbackFitnessImproves) {
    FunctionalFitness<double> ff(NUM_GENES, LB, UB,
        [](const std::vector<double>& x) {
            double s = 0.0;
            for (double v : x) s += v * v;
            return s;
        }
    );
    auto pop = makePopulation();
    const double initial_best = bestFitnessOf(pop, ff);

    auto ops = makeOperators();
    cuda::StandardGACUDA<double> ga(
        ff, std::move(ops.sel), std::move(ops.mu), std::move(ops.cs),
        0.05, 0.8, 150, true
    );
    ga.run(pop);

    EXPECT_LT(pop.getBestIndividual().getFitness(), initial_best)
        << "Expected strict improvement. Initial best: " << initial_best;
}

// 4. GPU kernel path (Sphere, name()=="Sphere"): converges to near-zero on
//    a 10-D problem with 500 individuals over 300 generations.
TEST_F(StandardGACUDATest, GPUKernelSphereConverges) {
    SphereFunction<double> ff(NUM_GENES, LB, UB);
    auto ops = makeOperators();
    cuda::StandardGACUDA<double> ga(
        ff, std::move(ops.sel), std::move(ops.mu), std::move(ops.cs),
        0.05, 0.8, 300, true
    );
    auto pop = makePopulation(500);
    ga.run(pop);

    EXPECT_LT(pop.getBestIndividual().getFitness(), 0.5)
        << "Sphere did not converge. Best fitness: "
        << pop.getBestIndividual().getFitness();
}

// 5. GPU kernel path (Rastrigin, name()=="Rastrigin"): fitness improves by
//    at least 50 % over the initial random population.
//    Exact convergence is not asserted because Rastrigin is highly multimodal.
TEST_F(StandardGACUDATest, GPUKernelRastriginImproves) {
    RastriginFunction<double> ff(NUM_GENES, LB, UB);
    auto pop = makePopulation(500);
    const double initial_best = bestFitnessOf(pop, ff);

    auto ops = makeOperators();
    cuda::StandardGACUDA<double> ga(
        ff, std::move(ops.sel), std::move(ops.mu), std::move(ops.cs),
        0.05, 0.8, 300, true
    );
    ga.run(pop);

    EXPECT_LT(pop.getBestIndividual().getFitness(), initial_best * 0.5)
        << "Rastrigin did not improve enough. Initial=" << initial_best
        << " Final=" << pop.getBestIndividual().getFitness();
}

// 6. Elitism prevents regression: the best solution found after the run must
//    be strictly better than the best random initial solution.
TEST_F(StandardGACUDATest, ElitismPreventsRegression) {
    FunctionalFitness<double> ff(NUM_GENES, LB, UB,
        [](const std::vector<double>& x) {
            double s = 0.0;
            for (double v : x) s += v * v;
            return s;
        }
    );
    auto pop = makePopulation();
    const double initial_best = bestFitnessOf(pop, ff);

    auto ops = makeOperators();
    cuda::StandardGACUDA<double> ga(
        ff, std::move(ops.sel), std::move(ops.mu), std::move(ops.cs),
        0.05, 0.8, 100, true
    );
    ga.run(pop);

    EXPECT_LT(pop.getBestIndividual().getFitness(), initial_best)
        << "Elitism violated: best fitness degraded. Initial=" << initial_best
        << " Final=" << pop.getBestIndividual().getFitness();
}

// 7. Both StandardGA (CPU) and StandardGACUDA reach a reasonable quality
//    bound on Sphere, confirming the CUDA variant produces valid results.
//    Exact numeric equality is not expected: the two use different RNGs and
//    StandardGACUDA internally converts genes to float32.
TEST_F(StandardGACUDATest, CPUAndCUDABothConvergeOnSphere) {
    SphereFunction<double> ff(NUM_GENES, LB, UB);
    constexpr double    THRESHOLD = 5.0;
    constexpr std::size_t MAX_GEN = 200;

    const double cpu_best = [&] {
        auto ops = makeOperators();
        StandardGA<double> ga(
            ff, std::move(ops.sel), std::move(ops.mu), std::move(ops.cs),
            0.05, 0.8, MAX_GEN, true, /*threads=*/1
        );
        auto pop = makePopulation();
        ga.run(pop);
        return pop.getBestIndividual().getFitness();
    }();

    const double cuda_best = [&] {
        auto ops = makeOperators();
        cuda::StandardGACUDA<double> ga(
            ff, std::move(ops.sel), std::move(ops.mu), std::move(ops.cs),
            0.05, 0.8, MAX_GEN, true
        );
        auto pop = makePopulation();
        ga.run(pop);
        return pop.getBestIndividual().getFitness();
    }();

    EXPECT_LT(cpu_best, THRESHOLD)
        << "StandardGA (CPU) did not converge. Best=" << cpu_best;
    EXPECT_LT(cuda_best, THRESHOLD)
        << "StandardGACUDA did not converge. Best=" << cuda_best;
}

#endif // GALIB_WITH_CUDA

// ─── Always-compiled tests (no CUDA required) ──────────────────────────────

// 8. enableFileLogging() + run() creates a non-empty CSV file.
//    Tested on StandardGA because StandardGACUDA::run() intentionally bypasses
//    notifyLoggers() — the GPU population lives on-device during evolution and
//    syncing it to the CPU each generation would negate GPU throughput.
TEST_F(StandardGACUDATest, FileLoggingCreatesFile) {
    SphereFunction<double> ff(NUM_GENES, LB, UB);
    auto ops = makeOperators();
    StandardGA<double> ga(
        ff, std::move(ops.sel), std::move(ops.mu), std::move(ops.cs),
        0.05, 0.8, /*max_gen=*/10, /*elitism=*/true, /*threads=*/1
    );
    ga.enableFileLogging("test_output.csv", /*interval=*/1);

    auto pop = makePopulation();
    ga.run(pop);

    ASSERT_TRUE(std::filesystem::exists("test_output.csv"))
        << "Log file was not created.";
    EXPECT_GT(std::filesystem::file_size("test_output.csv"), std::uintmax_t{0})
        << "Log file is empty.";
}
