#ifndef CELLULAR_GA_H
#define CELLULAR_GA_H

#pragma once

#include "core/GridPopulation.h"
#include "core/FitnessFunction.h"
#include "algorithms/cellular/selection/LocalSelection.h"
#include "operators/mutation/Mutation.h"
#include "operators/crossover/Crossover.h"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <iostream>
#include <random>
#include <utility>
#include <fstream>
#include <filesystem>
#include <string>

namespace galib {

template <typename GeneType = double>
class CellularGA {
private:
    FitnessFunction<GeneType>& fitness_function_m;
    LocalSelection<GeneType>& local_selection_m;
    Mutation<GeneType>& mutation_m;
    Crossover<GeneType>& crossover_m;

    double mutation_rate_m;
    double crossover_rate_m;
    std::size_t max_generations_m;
    bool use_local_elitism_m;

    std::string log_file_m = "";

private:
    void evaluatePopulation(GridPopulation<GeneType>& population) {
        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, population.size()),
            [&](const tbb::blocked_range<std::size_t>& range) {
                for (std::size_t i = range.begin(); i < range.end(); ++i) {
                    auto& individual = population.linearAt(i);
                    double score = fitness_function_m.evaluate(individual.getGenotype());
                    individual.setFitness(score);
                }
            }
        );
    }

    Individual<GeneType> evolveCell(
        const GridPopulation<GeneType>& population,
        std::size_t row,
        std::size_t col
    ) const {
        thread_local static std::random_device rd;
        thread_local static std::mt19937_64 gen(rd());
        thread_local static std::uniform_real_distribution<double> dist(0.0, 1.0);

        const Individual<GeneType>& current = population.at(row, col);
        const Individual<GeneType>& neighbor = local_selection_m.select(population, row, col);

        Individual<GeneType> child1;
        Individual<GeneType> child2;

        if (dist(gen) < crossover_rate_m) {
            auto children = crossover_m.crossover(current, neighbor);
            child1 = std::move(children.first);
            child2 = std::move(children.second);
        } else {
            child1 = current;
            child2 = neighbor;
        }

        mutation_m.mutate(child1, mutation_rate_m);
        mutation_m.mutate(child2, mutation_rate_m);

        child1.setFitness(fitness_function_m.evaluate(child1.getGenotype()));
        child2.setFitness(fitness_function_m.evaluate(child2.getGenotype()));

        if (use_local_elitism_m) {
            const Individual<GeneType>* best = &current;

            if (child1.getFitness() < best->getFitness()) {
                best = &child1;
            }
            if (child2.getFitness() < best->getFitness()) {
                best = &child2;
            }

            return *best;
        } else {
            return (child1.getFitness() < child2.getFitness()) ? child1 : child2;
        }
    }

public:
    CellularGA(
        FitnessFunction<GeneType>& ff,
        LocalSelection<GeneType>& local_sel,
        Mutation<GeneType>& mu,
        Crossover<GeneType>& cs,
        double m_rate,
        double c_rate,
        std::size_t max_gen,
        bool elitism = false
    )
        : fitness_function_m(ff),
          local_selection_m(local_sel),
          mutation_m(mu),
          crossover_m(cs),
          mutation_rate_m(m_rate),
          crossover_rate_m(c_rate),
          max_generations_m(max_gen),
          use_local_elitism_m(elitism) {}

    void enableLogging(const std::string& filename) {
        log_file_m = filename;
    }

    void run(GridPopulation<GeneType>& population) {
        std::ofstream log;

        if (!log_file_m.empty()) {
            std::filesystem::path p(log_file_m);

            if (p.has_parent_path()) {
                std::filesystem::create_directories(p.parent_path());
            }

            log.open(log_file_m);

            if (!log.is_open()) {
                std::cerr << "Error: Could not open log file " << log_file_m << std::endl;
            } else {
                log << "generation,cell_idx,x,y\n";
            }
        }

        if (population.empty()) {
            return;
        }

        GridPopulation<GeneType> new_population(
            population.rows(),
            population.cols(),
            population.getNumGenes()
        );

        evaluatePopulation(population);

        for (std::size_t generation_idx = 0; generation_idx < max_generations_m; ++generation_idx) {

            if (log.is_open()) {
                for (std::size_t i = 0; i < population.size(); ++i) {
                    const auto& ind = population.linearAt(i);
                    log << generation_idx << "," << i << ","
                        << ind.getGenotype()[0] << ","
                        << ind.getGenotype()[1] << "\n";
                }
            }

            tbb::parallel_for(
                tbb::blocked_range<std::size_t>(0, population.size()),
                [&](const tbb::blocked_range<std::size_t>& range) {
                    for (std::size_t i = range.begin(); i < range.end(); ++i) {
                        std::size_t row = i / population.cols();
                        std::size_t col = i % population.cols();

                        new_population.at(row, col) = evolveCell(population, row, col);
                    }
                }
            );

            std::swap(population, new_population);

            if ((generation_idx + 1) % 50 == 0 || generation_idx == 0) {
                std::cout << "Generation " << (generation_idx + 1)
                        << " | Best Fitness: " << population.getBestIndividual().getFitness()
                        << std::endl;
            }
        }

        if (log.is_open()) {
            log.close();
        }
    }
};

} // namespace galib

#endif // CELLULAR_GA_H
