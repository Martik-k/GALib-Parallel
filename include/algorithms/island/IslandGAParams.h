#ifndef ISLAND_GA_PARAMS_H
#define ISLAND_GA_PARAMS_H

#pragma once

#include <cstddef>
#include "algorithms/AlgorithmConfig.h"

namespace galib {

    /**
     * @brief Parameters for the Island Model Genetic Algorithm.
     */
    struct IslandGAParams : public AlgorithmConfig {
        std::size_t migration_interval = 50;
        std::size_t migration_size = 5;
        double immigration_quota = 0.1;
        std::size_t buffer_capacity = 5;
    };
}

#endif // ISLAND_GA_PARAMS_H
