#ifndef ISLAND_CONFIG_H
#define ISLAND_CONFIG_H

#pragma once

#include <cstddef>
#include <string>

namespace galib {
    struct IslandConfig {
        // Standard GA Parameters
        std::size_t population_size = 100;
        std::size_t max_generations = 1000;
        double mutation_rate = 0.05;
        double crossover_rate = 0.8;

        // Island Model Specific Parameters
        std::size_t migration_interval = 50;
        std::size_t migration_size = 5;
        double immigration_quota = 0.1;
        std::size_t buffer_capacity = 5;

        // Logging Parameters
        std::string log_directory = "logs";
        std::size_t log_interval = 0; // 0 means disabled
    };
}

#endif // ISLAND_CONFIG_H
