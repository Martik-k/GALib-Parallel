#ifndef ALGORITHM_H
#define ALGORITHM_H

#pragma once

#include "core/Population.h"
#include "utils/ConsoleLogger.h"
#include "utils/FileLogger.h"
#include <memory>

namespace galib {

    /**
     * @brief Abstract base class for all optimization algorithms.
     * 
     * Defines the core interface for running an evolutionary process and 
     * provides common functionality for logging to console and files.
     * 
     * @tparam GeneType The type of the genes (e.g., double, int).
     */
    template <typename GeneType>
    class Algorithm {
    protected:
        std::unique_ptr<utils::ConsoleLogger<GeneType>> console_logger_m;
        std::unique_ptr<utils::FileLogger<GeneType>> file_logger_m;

        /**
         * @brief Notifies all active loggers to record the current state.
         * @param gen The current generation index.
         * @param pop The current population.
         */
        void notifyLoggers(std::size_t gen, const Population<GeneType>& pop) {
            if (console_logger_m) {
                console_logger_m->log(gen, pop);
            }
            if (file_logger_m) {
                file_logger_m->log(gen, pop);
            }
        }

    public:
        virtual ~Algorithm() = default;

        /**
         * @brief Executes the optimization process on the given population.
         * 
         * This is a pure-virtual method that must be implemented by concrete 
         * algorithms (e.g., StandardGA, IslandGA).
         * 
         * @param population The population to optimize. Modified in-place.
         * @note  The population must be initialized before calling this method.
         */
        virtual void run(Population<GeneType>& population) = 0;

        /**
         * @brief Enables logging of progress to the standard output.
         * @param interval The frequency of logging (every N generations). 0 disables.
         */
        virtual void enableConsoleLogging(const std::size_t interval) {
            console_logger_m = std::make_unique<utils::ConsoleLogger<GeneType>>(interval);
        }

        /**
         * @brief Enables logging of population snapshots to a CSV file.
         * @param path     The path where the CSV file will be created.
         * @param interval The frequency of logging (every N generations). 0 disables.
         */
        virtual void enableFileLogging(const std::string& path, std::size_t interval) {
            file_logger_m = std::make_unique<utils::FileLogger<GeneType>>(path, interval);
        }

        /**
         * @brief Placeholder for generic logging configuration.
         */
        virtual void enableLogging(const std::string&) {}
    };

} // namespace galib

#endif // ALGORITHM_H
