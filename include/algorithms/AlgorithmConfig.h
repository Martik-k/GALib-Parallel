#ifndef ALGORITHM_CONFIG_H
#define ALGORITHM_CONFIG_H

#pragma once

#include <cstddef>

namespace galib {

    /**
     * @brief Base structure for common genetic algorithm configuration parameters.
     * 
     * This POD (Plain Old Data) structure aggregates universal parameters like 
     * mutation rates and generation limits that apply to most GA variants.
     */
    struct AlgorithmConfig {
        /** @brief Maximum number of generations to evolve. */
        std::size_t max_generations = 100;

        /** @brief Probability of mutation occurring for each individual [0.0, 1.0]. */
        double mutation_rate = 0.05;

        /** @brief Probability of crossover between parents [0.0, 1.0]. */
        double crossover_rate = 0.8;

        /** @brief Whether to preserve the best individual across generations. */
        bool use_elitism = true;

        /** @brief Number of OpenMP threads to use. 0 means use system default (omp_get_max_threads). */
        std::size_t num_threads = 0;

        virtual ~AlgorithmConfig() = default;
    };

} // namespace galib

#endif // ALGORITHM_CONFIG_H
