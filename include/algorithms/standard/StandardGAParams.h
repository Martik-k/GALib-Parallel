#ifndef STANDARD_GA_PARAMS_H
#define STANDARD_GA_PARAMS_H

#pragma once

#include <cstddef>

namespace galib {

    /**
     * @brief Parameters for the Standard (Generational) Genetic Algorithm.
     */
    struct StandardGAParams {
        double mutation_rate = 0.05;
        double crossover_rate = 0.8;
        std::size_t max_generations = 100;
        bool use_elitism = true;
    };

} // namespace galib

#endif // STANDARD_GA_PARAMS_H
