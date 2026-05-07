#ifndef CELLULAR_GA_PARAMS_H
#define CELLULAR_GA_PARAMS_H

#pragma once

#include <cstddef>

#include "algorithms/AlgorithmConfig.h"

namespace galib {

/**
 * @brief Configuration parameters specific to the cellular genetic algorithm.
 *
 * Extends the common @ref AlgorithmConfig with behaviour that controls local
 * survivor selection inside each cell neighbourhood.
 */
struct CellularGAParams : public AlgorithmConfig {
    /**
     * @brief Keeps the best individual among the current cell and its offspring.
     *
     * If enabled, the current individual competes with both produced children
     * and the fittest of the three survives into the next generation.
     */
    bool use_local_elitism = true;
};

} // namespace galib

#endif // CELLULAR_GA_PARAMS_H
