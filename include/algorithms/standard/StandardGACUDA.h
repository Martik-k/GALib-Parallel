#ifndef STANDARD_GA_CUDA_H
#define STANDARD_GA_CUDA_H

#pragma once

#include "algorithms/Algorithm.h"
#include "core/Population.h"
#include "core/FitnessFunction.h"
#include "operators/selection/Selection.h"
#include "operators/mutation/Mutation.h"
#include "operators/crossover/Crossover.h"

#include <cstddef>
#include <string>
#include <memory>
#include <iostream>
#include <random>
#include <utility>
#include <fstream>
#include <omp.h>
#include <filesystem>

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
     * @brief Enables per-generation CSV logging to a file.
     *
     * When set, every individual's first two gene values (x, y) are written each
     * generation. Useful for 2-D visualisation. Has no effect if called after run().
     *
     * @param filename Path to the output CSV file. Parent directories are created
     *                 automatically if they do not exist.
     */
    void enableLogging(const std::string& filename) override { log_file_m = filename; }

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
        std::ofstream log;
        if (!log_file_m.empty()) {
            std::filesystem::path p(log_file_m);
            if (p.has_parent_path()) {
                std::filesystem::create_directories(p.parent_path());
            }
            log.open(log_file_m);
            if (!log.is_open()) {
                std::cerr << "Error: Could not open log file " << log_file_m << std::endl;
            } else {
                log << "generation,individual_idx,x,y\n";
            }
        }

        if (population.empty()) { return; }

        std::size_t population_size = population.size();
        std::size_t num_genes = population.getNumGenes();

        Population<GeneType> new_population(population_size, num_genes);

        evaluatePopulation(population);

        for (std::size_t generation_idx = 0; generation_idx < max_generations_m; ++generation_idx) {

            if (log.is_open()) {
                for (std::size_t i = 0; i < population.size(); ++i) {
                    log << generation_idx << "," << i << ","
                        << population[i].getGenotype()[0] << ","
                        << population[i].getGenotype()[1] << "\n";
                }
            }

            std::size_t elitism_offset = 0;
            if (use_elitism_m) {
                new_population[0] = population.getBestIndividual();
                elitism_offset = 1;
            }

            #pragma omp parallel for schedule(static)
            for (std::size_t i = elitism_offset; i < population_size; i += 2) {
                thread_local static std::random_device tl_rd;
                thread_local static std::mt19937_64 tl_gen(tl_rd());
                thread_local static std::uniform_real_distribution<double> tl_dist(0.0, 1.0);

                const Individual<GeneType>& parent1 = selection_m->select(population);
                const Individual<GeneType>& parent2 = selection_m->select(population);

                if (tl_dist(tl_gen) < crossover_rate_m) {
                    auto children = crossover_m->crossover(parent1, parent2);
                    new_population[i] = std::move(children.first);
                    if (i + 1 < population_size) {
                        new_population[i + 1] = std::move(children.second);
                    }
                } else {
                    new_population[i] = parent1;
                    if (i + 1 < population_size) {
                        new_population[i + 1] = parent2;
                    }
                }

                mutation_m->mutate(new_population[i], mutation_rate_m);
                if (i + 1 < population_size) {
                    mutation_m->mutate(new_population[i + 1], mutation_rate_m);
                }
            }

            std::swap(population, new_population);

            evaluatePopulation(population);

            this->notifyLoggers(generation_idx, population);
            if (!this->console_logger_m && ((generation_idx + 1) % 50 == 0 || generation_idx == 0)) {
                std::cout << "Generation " << (generation_idx + 1)
                          << " | Best Fitness: " << population.getBestIndividual().getFitness()
                          << std::endl;
            }
        }
        if (log.is_open()) log.close();
    }

private:
    FitnessFunction<GeneType>& fitness_function_m;
    std::unique_ptr<Selection<GeneType>> selection_m;
    std::unique_ptr<Mutation<GeneType>> mutation_m;
    std::unique_ptr<Crossover<GeneType>> crossover_m;

    double mutation_rate_m;
    double crossover_rate_m;
    std::size_t max_generations_m;
    bool use_elitism_m;

    std::string log_file_m = "";

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
