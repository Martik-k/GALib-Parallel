#ifndef STANDARD_GA_CUDA_H
#define STANDARD_GA_CUDA_H

#pragma once

#include "algorithms/Algorithm.h"
#include "core/Population.h"
#include "core/FitnessFunction.h"
#include "algorithms/standard/CUDAEvaluator.h"
#include "operators/selection/Selection.h"
#include "operators/selection/TournamentSelection.h"
#include "operators/mutation/Mutation.h"
#include "operators/mutation/GaussianMutation.h"
#include "operators/crossover/Crossover.h"

#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

namespace galib {
namespace cuda {

/**
 * @brief Generational GA with the same interface as StandardGA, built for CUDA-enabled builds.
 *
 * Drop-in replacement for galib::StandardGA. The evolution loop (selection, crossover,
 * mutation) runs on the CPU using OpenMP, making it fully compatible with any pluggable
 * operator. When the library is compiled with CUDA support, the CUDA runtime is linked
 * and GPU-accelerated fitness evaluation can be added in future iterations.
 *
 * The algorithm follows a generational model:
 *  1. Evaluate the initial population.
 *  2. For each generation: optionally copy the elite individual, then fill the rest of
 *     the population via selection → crossover → mutation.
 *  3. Evaluate the new population and repeat.
 *
 * @tparam GeneType Numeric type of each gene (default: double).
 *
 * @note Prefer constructing through galib::utils::AlgorithmBuilder with
 *       `algorithm.standard.use_cuda: true` rather than instantiating directly.
 * @note Fitness is minimised — lower values are considered better.
 */
template <typename GeneType = double>
class StandardGACUDA : public Algorithm<GeneType> {
public:
    /**
     * @brief Constructs the algorithm with all required operators and parameters.
     *
     * @param ff      Fitness function used to evaluate individuals. Must outlive this object.
     * @param sel     Selection operator (e.g. TournamentSelection). Ownership is transferred.
     * @param mu      Mutation operator (e.g. GaussianMutation). Ownership is transferred.
     * @param cs      Crossover operator (e.g. SinglePointCrossover). Ownership is transferred.
     * @param m_rate  Probability [0, 1] that any individual gene is mutated.
     * @param c_rate  Probability [0, 1] that two parents produce children via crossover;
     *                otherwise parents are copied unchanged.
     * @param max_gen Number of generations to run.
     * @param elitism If true, the best individual is always copied to the next generation.
     */
    StandardGACUDA(
        FitnessFunction<GeneType>& ff,
        std::unique_ptr<Selection<GeneType>> sel,
        std::unique_ptr<Mutation<GeneType>> mu,
        std::unique_ptr<Crossover<GeneType>> cs,
        double m_rate,
        double c_rate,
        std::size_t max_gen,
        bool elitism = true
    ) : fitness_function_m(ff),
        selection_m(std::move(sel)),
        mutation_m(std::move(mu)),
        crossover_m(std::move(cs)),
        mutation_rate_m(m_rate),
        crossover_rate_m(c_rate),
        max_generations_m(max_gen),
        use_elitism_m(elitism) {}

    /**
     * @brief Runs the evolutionary optimisation loop.
     *
     * Evolves @p population in-place for max_gen generations. The population must
     * be initialised with valid gene values before calling this method.
     *
     * Progress is printed to stdout every 50 generations unless a ConsoleLogger is
     * attached via enableConsoleLogging(), in which case that logger is used instead.
     *
     * @param population In/out: the population to evolve. Modified in place.
     * @note Population must not be empty.
     */
    void run(Population<GeneType>& population) override {
        if (population.empty()) return;

        const int pop_size = static_cast<int>(population.size());
        const int dims     = static_cast<int>(population.getNumGenes());

        // Flatten Population<GeneType> → vector<float>
        std::vector<float> h_genes(static_cast<std::size_t>(pop_size) * dims);
        for (int i = 0; i < pop_size; ++i) {
            const auto& g = population[i].getGenotype();
            for (int j = 0; j < dims; ++j)
                h_genes[static_cast<std::size_t>(i) * dims + j] = static_cast<float>(g[j]);
        }

        std::vector<float> h_fitness;

        // Build params — fitness callback works for ANY FitnessFunction
        internal::CUDARunParams params;
        params.pop_size        = pop_size;
        params.dims            = dims;
        params.max_generations = static_cast<int>(max_generations_m);
        params.mutation_rate   = static_cast<float>(mutation_rate_m);
        params.crossover_rate  = static_cast<float>(crossover_rate_m);
        params.use_elitism     = use_elitism_m;
        // Extract tournament size from the actual operator if available
        if (auto* ts = dynamic_cast<TournamentSelection<GeneType>*>(selection_m.get()))
            params.tournament_size = static_cast<int>(ts->getTournamentSize());
        else
            params.tournament_size = 3;

        // Extract sigma from the actual mutation operator if available
        if (auto* gm = dynamic_cast<GaussianMutation<GeneType>*>(mutation_m.get()))
            params.sigma = static_cast<float>(gm->getSigma());
        else
            params.sigma = 0.1f;

        params.problem_name = fitness_function_m.name();

        // CPU fallback: called for any function without a GPU kernel
        params.fitness_callback = [this](const float* genes, int d) -> double {
            std::vector<GeneType> phenotype(static_cast<std::size_t>(d));
            for (int j = 0; j < d; ++j)
                phenotype[static_cast<std::size_t>(j)] = static_cast<GeneType>(genes[j]);
            return fitness_function_m.evaluate(phenotype);
        };

        // Progress — population lives on device during the run, so we print the
        // GPU-reported best_fitness directly rather than routing through notifyLoggers
        // (which would read the stale CPU population).
        params.progress_callback = [this](int gen, double best_fitness) {
            if ((gen + 1) % 50 == 0 || gen == 0) {
                std::cout << "Generation " << (gen + 1)
                          << " | Best Fitness: " << best_fitness << std::endl;
            }
        };

        if (!internal::runCUDAEvolution(params, h_genes, h_fitness)) {
            std::cerr << "[GALib] CUDA evolution failed. Population unchanged.\n";
            return;
        }

        // Write results back into Population<GeneType>
        for (int i = 0; i < pop_size; ++i) {
            auto& ind = population[static_cast<std::size_t>(i)];
            auto& g   = ind.getGenotype();
            for (int j = 0; j < dims; ++j)
                g[static_cast<std::size_t>(j)] =
                    static_cast<GeneType>(h_genes[static_cast<std::size_t>(i) * dims + j]);
            ind.setFitness(static_cast<double>(h_fitness[static_cast<std::size_t>(i)]));
        }
    }

private:
    FitnessFunction<GeneType>& fitness_function_m;
    std::unique_ptr<Selection<GeneType>>  selection_m;
    std::unique_ptr<Mutation<GeneType>>   mutation_m;
    std::unique_ptr<Crossover<GeneType>>  crossover_m;

    double      mutation_rate_m;
    double      crossover_rate_m;
    std::size_t max_generations_m;
    bool use_elitism_m;

    /**
     * @brief Evaluates fitness for every individual in the population using OpenMP.
     * @param population The population to evaluate. Fitness values are set in place.
     */
    void evaluatePopulation(Population<GeneType>& population) {
        int population_size = static_cast<int>(population.size());
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < population_size; ++i) {
            double score = fitness_function_m.evaluate(population[i].getGenotype());
            population[i].setFitness(score);
        }
    }
};

}
}

#endif // STANDARD_GA_CUDA_H
