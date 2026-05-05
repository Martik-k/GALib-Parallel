#ifndef CONSOLE_LOGGER_H
#define CONSOLE_LOGGER_H

#pragma once

#include <iostream>
#include <cstddef>
#include "core/Population.h"

namespace galib::utils {
    /**
     * @brief Logger for printing optimization progress to the console.
     */
    template <typename GeneType>
    class ConsoleLogger {
    private:
        std::size_t interval_m;

    public:
        explicit ConsoleLogger(const std::size_t interval) : interval_m(interval) {}

        void log(const std::size_t gen, const Population<GeneType>& pop) {
            if (interval_m > 0 && (gen % interval_m == 0 || gen == 0)) {
                std::cout << "Generation " << gen
                    << " | Best Fitness: " << pop.getBestIndividual().getFitness()
                    << std::endl;
            }
        }
    };
} // namespace galib::utils

#endif // CONSOLE_LOGGER_H
