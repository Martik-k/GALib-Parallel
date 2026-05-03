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
         * @param population The population to optimize.
         */
        virtual void run(Population<GeneType>& population) = 0;

        /**
         * @brief Enables logging to the console.
         * @param interval The frequency of logging (every N generations).
         */
        virtual void enableConsoleLogging(std::size_t interval) {
            console_logger_m = std::make_unique<utils::ConsoleLogger<GeneType>>(interval);
        }

        /**
         * @brief Enables logging to a file.
         * @param path The path to the log file.
         * @param interval The frequency of logging (every N generations).
         */
        virtual void enableFileLogging(const std::string& path, std::size_t interval) {
            file_logger_m = std::make_unique<utils::FileLogger<GeneType>>(path, interval);
        }
    };

} // namespace galib

#endif // ALGORITHM_H
