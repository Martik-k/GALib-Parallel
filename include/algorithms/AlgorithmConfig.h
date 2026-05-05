#ifndef ALGORITHM_CONFIG_H
#define ALGORITHM_CONFIG_H

#pragma once

#include <cstddef>

namespace galib {

    /**
     * @brief Base structure for common genetic algorithm configuration parameters.
     */
    struct AlgorithmConfig {
        std::size_t max_generations = 100;
        double mutation_rate = 0.05;
        double crossover_rate = 0.8;
        bool use_elitism = true;

        virtual ~AlgorithmConfig() = default;
    };

} // namespace galib

#endif // ALGORITHM_CONFIG_H
