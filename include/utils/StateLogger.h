#ifndef STATE_LOGGER_H
#define STATE_LOGGER_H

#pragma once

#include "core/Population.h"
#include <fstream>
#include <string>
#include <filesystem>
#include <iostream>

namespace galib {
    namespace utils {

        template <typename GeneType = double>
        class StateLogger {
        private:
            std::ofstream log_file_m;
            std::string log_dir_m;
            std::size_t rank_m;
            bool is_opened_m = false;

            void createDirectory() const {
                if (!std::filesystem::exists(log_dir_m)) {
                    std::filesystem::create_directories(log_dir_m);
                }
            }

            [[nodiscard]] std::string getLogFilePath() const {
                return (std::filesystem::path(log_dir_m) / ("island_log_rank_" + std::to_string(rank_m) + ".csv")).string();
            }

        public:
            explicit StateLogger(const std::string& log_dir, const std::size_t rank)
                : log_dir_m(log_dir), rank_m(rank) {
                try {
                    createDirectory();
                    const std::string file_path = getLogFilePath();
                    log_file_m.open(file_path, std::ios::out | std::ios::trunc);
                    if (log_file_m.is_open()) {
                        is_opened_m = true;
                    } else {
                        std::cerr << "Failed to open log file: " << file_path << std::endl;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "StateLogger initialization error: " << e.what() << std::endl;
                }
            }

            ~StateLogger() {
                if (log_file_m.is_open()) {
                    log_file_m.close();
                }
            }

            void writeHeader(const std::size_t num_genes) {
                if (!is_opened_m) return;

                log_file_m << "generation,individual_id,fitness";
                for (std::size_t i = 0; i < num_genes; ++i) {
                    log_file_m << ",gene_" << i;
                }
                log_file_m << "\n";
            }

            void log(const Population<GeneType>& population, const std::size_t generation_idx) {
                if (!is_opened_m) return;

                for (std::size_t i = 0; i < population.size(); ++i) {
                    const auto& individual = population[i];
                    log_file_m << generation_idx << "," << i << "," << individual.getFitness();

                    const auto& genotype = individual.getGenotype();
                    for (const auto& gene : genotype) {
                        log_file_m << "," << gene;
                    }
                    log_file_m << "\n";
                }
                log_file_m.flush();
            }

            [[nodiscard]] bool is_opened() const { return is_opened_m; }
        };
    }
}

#endif // STATE_LOGGER_H
