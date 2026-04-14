#ifndef DIFFERENTIAL_EVOLUTION_GA_H
#define DIFFERENTIAL_EVOLUTION_GA_H

#pragma once

#include "core/Population.h"
#include "core/FitnessFunction.h"
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <omp.h>
#include <filesystem>
#include <algorithm> // для std::clamp

namespace galib {

    template <typename GeneType = double>
    class DifferentialEvolutionGA {
    private:
        FitnessFunction<GeneType>& fitness_function_m;

        double F_m; // differential weight [0, 2]
        double CR_m; // crossover rate (probability) [0, 1]
        std::size_t max_generations_m;
        std::string log_file_m = "";

        void evaluatePopulation(Population<GeneType>& population) {
            int pop_size = static_cast<int>(population.size());
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < pop_size; ++i) {
                double score = fitness_function_m.evaluate(population[i].getGenotype());
                population[i].setFitness(score);
            }
        }

    public:
        DifferentialEvolutionGA(
            FitnessFunction<GeneType>& ff,
            double f_weight,
            double cr_rate,
            std::size_t max_gen
        ) : fitness_function_m(ff), F_m(f_weight), CR_m(cr_rate), max_generations_m(max_gen) {}

        void enableLogging(const std::string& filename) { log_file_m = filename; }

        void run(Population<GeneType>& population) {
            std::ofstream log;
            if (!log_file_m.empty()) {
                std::filesystem::path p(log_file_m);
                if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
                log.open(log_file_m);
                if (log.is_open()) log << "generation,individual_idx,x,y\n";
            }

            if (population.empty()) return;

            std::size_t pop_size = population.size();
            std::size_t num_genes = population.getNumGenes();

            Population<GeneType> new_population(pop_size, num_genes);
            evaluatePopulation(population);

            std::random_device master_rd;
            unsigned int master_seed = master_rd();

            for (std::size_t generation_idx = 0; generation_idx < max_generations_m; ++generation_idx) {

                if (log.is_open()) {
                    for (std::size_t i = 0; i < pop_size; ++i) {
                        log << generation_idx << "," << i << ","
                            << population[i].getGenotype()[0] << ","
                            << population[i].getGenotype()[1] << "\n";
                    }
                }

                #pragma omp parallel
                {
                    std::mt19937_64 tl_gen(master_seed + omp_get_thread_num() + generation_idx);
                    std::uniform_real_distribution<double> tl_dist(0.0, 1.0);
                    std::uniform_int_distribution<std::size_t> tl_int_dist(0, pop_size - 1);
                    std::uniform_int_distribution<std::size_t> tl_gene_dist(0, num_genes - 1);

                    #pragma omp for schedule(static)
                    for (int i = 0; i < static_cast<int>(pop_size); ++i) {

                        std::size_t r1, r2, r3;
                        do { r1 = tl_int_dist(tl_gen); } while (r1 == static_cast<std::size_t>(i));
                        do { r2 = tl_int_dist(tl_gen); } while (r2 == static_cast<std::size_t>(i) || r2 == r1);
                        do { r3 = tl_int_dist(tl_gen); } while (r3 == static_cast<std::size_t>(i) || r3 == r1 || r3 == r2);

                        const auto& target = population[i].getGenotype();
                        const auto& x_r1 = population[r1].getGenotype();
                        const auto& x_r2 = population[r2].getGenotype();
                        const auto& x_r3 = population[r3].getGenotype();

                        Individual<GeneType> trial_ind(num_genes);
                        auto& trial = trial_ind.getGenotype();

                        std::size_t j_rand = tl_gene_dist(tl_gen);

                        // mutation
                        if (num_genes >= 4) {
                            #pragma omp simd
                            for (std::size_t j = 0; j < num_genes; ++j) {
                                trial[j] = x_r1[j] + F_m * (x_r2[j] - x_r3[j]);
                            }
                        } else {
                            for (std::size_t j = 0; j < num_genes; ++j) {
                                trial[j] = x_r1[j] + F_m * (x_r2[j] - x_r3[j]);
                            }
                        }

                        // crossover
                        for (std::size_t j = 0; j < num_genes; ++j) {
                            if (tl_dist(tl_gen) <= CR_m || j == j_rand) {
                                trial[j] = std::clamp(trial[j], fitness_function_m.getLowerBound(j), fitness_function_m.getUpperBound(j));
                            } else {
                                trial[j] = target[j];
                            }
                        }

                        // selection
                        double trial_fitness = fitness_function_m.evaluate(trial);
                        trial_ind.setFitness(trial_fitness);

                        if (trial_fitness <= population[i].getFitness()) {
                            new_population[i] = std::move(trial_ind);
                        } else {
                            new_population[i] = population[i];
                        }
                    }
                }

                std::swap(population, new_population);

                if ((generation_idx + 1) % 50 == 0 || generation_idx == 0) {
                    std::cout << "Generation " << (generation_idx + 1)
                              << " | Best Fitness: " << population.getBestIndividual().getFitness() << std::endl;
                }
            }
            if (log.is_open()) log.close();
        }
    };
}

#endif // DIFFERENTIAL_EVOLUTION_GA_H