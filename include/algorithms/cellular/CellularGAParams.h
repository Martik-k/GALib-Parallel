#ifndef CELLULAR_GA_PARAMS_H
#define CELLULAR_GA_PARAMS_H

#pragma once

#include <cstddef>

namespace galib {

    /**
     * @brief Parameters for the Cellular Genetic Algorithm.
     */
    struct CellularGAParams {
        double mutation_rate = 0.05;
        double crossover_rate = 0.8;
        std::size_t max_generations = 100;
        
        std::size_t rows = 10;
        std::size_t cols = 10;
        bool use_local_elitism = true;
    };

} // namespace galib

#endif // CELLULAR_GA_PARAMS_H
