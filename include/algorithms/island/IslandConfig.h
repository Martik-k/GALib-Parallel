#ifndef ISLAND_CONFIG_H
#define ISLAND_CONFIG_H

#pragma once

#include <cstddef>

namespace galib {
    /**
     * @brief Configuration parameters for the Island Model Genetic Algorithm.
     * 
     * This struct aggregates standard GA parameters with island-specific 
     * settings for migration and asynchronous buffering.
     */
    struct IslandConfig {
        // Standard GA Parameters
        std::size_t population_size = 100;
        std::size_t max_generations = 1000;
        double mutation_rate = 0.05;
        double crossover_rate = 0.8;

        // Island Model Specific Parameters

        /**
         * @brief Number of generations between migration events.
         */
        std::size_t migration_interval = 50;

        /**
         * @brief Number of individuals to send during each migration.
         */
        std::size_t migration_size = 5;
        double immigration_quota = 0.1;

        /**
         * @brief Capacity of the Migration Buffer (number of batches/demes).
         * 
         * Determines how many incoming migrations from neighbors can be stored 
         * before the oldest ones are overwritten.
         */
        std::size_t buffer_capacity = 5;
    };
}

#endif // ISLAND_CONFIG_H
