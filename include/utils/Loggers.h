#ifndef LOGGERS_H
#define LOGGERS_H

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
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
        explicit ConsoleLogger(std::size_t interval) : interval_m(interval) {}

        void log(std::size_t gen, const Population<GeneType>& pop) {
            if (interval_m > 0 && (gen % interval_m == 0 || gen == 0)) {
                std::cout << "Generation " << gen 
                          << " | Best Fitness: " << pop.getBestIndividual().getFitness() 
                          << std::endl;
            }
        }
    };

    /**
     * @brief Logger for writing optimization progress to a CSV file.
     */
    template <typename GeneType>
    class FileLogger {
    private:
        std::string path_m;
        std::size_t interval_m;
        bool header_written_m = false;

    public:
        FileLogger(const std::string& path, std::size_t interval) 
            : path_m(path), interval_m(interval) {}

        void log(std::size_t gen, const Population<GeneType>& pop) {
            if (interval_m == 0 || (gen % interval_m != 0 && gen != 0)) {
                return;
            }

            std::filesystem::path p(path_m);
            if (!header_written_m) {
                if (p.has_parent_path()) {
                    std::filesystem::create_directories(p.parent_path());
                }
            }

            std::ofstream file;
            if (!header_written_m) {
                file.open(path_m, std::ios::out);
                if (file.is_open()) {
                    file << "generation,individual_idx,fitness,genotype\n";
                    header_written_m = true;
                }
            } else {
                file.open(path_m, std::ios::app);
            }

            if (!file.is_open()) {
                return;
            }

            for (std::size_t i = 0; i < pop.size(); ++i) {
                file << gen << "," << i << "," << pop[i].getFitness() << ",";
                const auto& genotype = pop[i].getGenotype();
                for (std::size_t j = 0; j < genotype.size(); ++j) {
                    file << genotype[j] << (j == genotype.size() - 1 ? "" : ";");
                }
                file << "\n";
            }
            file.close();
        }
    };

} // namespace galib::utils

#endif // LOGGERS_H
