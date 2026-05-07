#ifndef ISLAND_GA_PARAMS_H
#define ISLAND_GA_PARAMS_H

#pragma once

#include <cstddef>
#include "algorithms/AlgorithmConfig.h"

namespace galib {

    /**
     * @brief Parameters for the Island Model Genetic Algorithm.
     * 
     * Extends the base AlgorithmConfig with parameters specific to migration
     * and asynchronous communication between islands.
     */
    struct IslandGAParams : public AlgorithmConfig {
        /** @brief Number of generations between migration events. */
        std::size_t migration_interval = 50;

        /** @brief Number of individuals to send per migration event. */
        std::size_t migration_size = 5;

        /** @brief Maximum fraction of the population that can be replaced by migrants in one step. */
        double immigration_quota = 0.1;

        /** @brief Capacity of the circular buffer for incoming migrants (in terms of batches). */
        std::size_t buffer_capacity = 5;
    };
}

#endif // ISLAND_GA_PARAMS_H
