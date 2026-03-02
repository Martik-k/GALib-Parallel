#ifndef STANDARD_GA_H
#define STANDARD_GA_H

#pragma once

#include "../core/Population.h"
#include "../core/FitnessFunction.h"
#include "../operators/selection/Selection.h"
#include "../operators/mutation/Mutation.h"
#include "../operators/crossover/Crossover.h"

#include <iostream>
#include <random>
#include <utility>
#include <fstream>

namespace galib {

    template <typename GeneType = double>
    class StandardGA {
    private:
        FitnessFunction<GeneType>& fitness_function_m;
        Selection<GeneType>& selection_m;
        Mutation<GeneType>& mutation_m;
        Crossover<GeneType>& crossover_m;

        double mutation_rate_m;
        double crossover_rate_m;
        std::size_t max_generations_m;
        bool use_elitism_m;

        std::random_device rd_m;
        std::mt19937_64 gen_m;
        std::uniform_real_distribution<double> distribution_m;

        std::string log_file_m = "";

        void evaluatePopulation(Population<GeneType>& population) {
            int population_size = static_cast<int>(population.size());

        #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < population_size; ++i) {
                double score = fitness_function_m.evaluate(population[i].getGenotype());
                population[i].setFitness(score);
            }
        }

    public:
        StandardGA(
            FitnessFunction<GeneType>& ff,
            Selection<GeneType>& sel,
            Mutation<GeneType>& mu,
            Crossover<GeneType>& cs,
            double m_rate = 0.1,
            double c_rate = 0.9,
            std::size_t max_gen = 100,
            bool elitism = true
        ) : fitness_function_m(ff), selection_m(sel), mutation_m(mu), crossover_m(cs),
            mutation_rate_m(m_rate), crossover_rate_m(c_rate), max_generations_m(max_gen),
            use_elitism_m(elitism), rd_m(), gen_m(rd_m()), distribution_m(0.0, 1.0) {}

        void enableLogging(const std::string& filename) { log_file_m = filename; }

        void run(Population<GeneType>& population) {
            std::ofstream log;
            if (!log_file_m.empty()) {
                log.open(log_file_m);
                log << "generation,individual_idx,x,y\n";
            }

            if (population.empty()) { return; }

            std::size_t population_size = population.size();
            std::size_t num_genes = population.getNumGenes();

            evaluatePopulation(population);

            for (std::size_t generation_idx = 0; generation_idx < max_generations_m; ++generation_idx) {

                if (log.is_open()) {
                    for (std::size_t i = 0; i < population.size(); ++i) {
                        log << generation_idx << "," << i << ","
                            << population[i].getGenotype()[0] << ","
                            << population[i].getGenotype()[1] << "\n";
                    }
                }

                Population<GeneType> new_population(population_size, num_genes);
                std::size_t population_idx = 0;

                if (use_elitism_m) {
                    new_population[0] = population.getBestIndividual();
                    population_idx = 1;
                }

                while (population_idx < population_size) {
                    const Individual<GeneType>& parent1 = selection_m.select(population);
                    const Individual<GeneType>& parent2 = selection_m.select(population);

                    Individual<GeneType> child1 = parent1;
                    Individual<GeneType> child2 = parent2;

                    if (distribution_m(gen_m) < crossover_rate_m) {
                        auto children = crossover_m.crossover(parent1, parent2);
                        child1 = children.first;
                        child2 = children.second;
                    }

                    mutation_m.mutate(child1, mutation_rate_m);
                    mutation_m.mutate(child2, mutation_rate_m);

                    new_population[population_idx++] = child1;
                    if (population_idx < population_size) {
                        new_population[population_idx++] = child2;
                    }
                }

                population = std::move(new_population);
                evaluatePopulation(population);

                if ((generation_idx + 1) % 10 == 0 || generation_idx == 0) {
                    std::cout << "Generation " << (generation_idx + 1)
                              << " | Best Fitness: " << population.getBestIndividual().getFitness()
                              << std::endl;
                }
            }
            if (log.is_open()) log.close();
        }
    };

}

#endif // STANDARD_GA_H
