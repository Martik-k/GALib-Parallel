#ifndef CELLULAR_GA_H
#define CELLULAR_GA_H

#pragma once

#include "core/Population.h"
#include "core/FitnessFunction.h"
#include "algorithms/cellular/selection/LocalSelection.h"
#include "operators/mutation/Mutation.h"
#include "operators/crossover/Crossover.h"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>

#include <memory>
#include <random>
#include <utility>
#include <fstream>
#include <filesystem>
#include <string>
#include <cmath>
#include <stdexcept>

#include "algorithms/Algorithm.h"

namespace galib {

/**
 * @brief Cellular genetic algorithm that evolves individuals on a local 2D neighbourhood.
 *
 * The algorithm accepts a regular @ref Population and interprets it as a 2D toroidal
 * grid. Grid dimensions are inferred automatically from the population size by choosing
 * a factorisation close to a square. Each cell evolves using only its local neighbourhood,
 * which preserves spatial diversity better than fully mixing the whole population.
 *
 * The evolutionary loop is:
 *  1. Infer grid dimensions from @p population.size().
 *  2. Evaluate the initial population.
 *  3. For each generation, evolve every cell independently from its local neighbourhood.
 *  4. Replace the old population with the newly produced one.
 *
 * @tparam GeneType Numeric type of each gene (default: double).
 *
 * @note Fitness is minimised: lower values are considered better.
 * @note The grid uses wrap-around boundaries, so edge cells still have neighbours.
 * @note When @p threads is 0, TBB chooses the level of parallelism automatically.
 */
template <typename GeneType = double>
class CellularGA : public Algorithm<GeneType> {
private:
    FitnessFunction<GeneType>& fitness_function_m;
    std::unique_ptr<LocalSelection<GeneType>> local_selection_m;
    std::unique_ptr<Mutation<GeneType>> mutation_m;
    std::unique_ptr<Crossover<GeneType>> crossover_m;

    double mutation_rate_m;
    double crossover_rate_m;
    std::size_t max_generations_m;
    std::size_t threads_m;
    bool use_local_elitism_m;

    std::size_t rows_m;
    std::size_t cols_m;

private:
    static std::pair<std::size_t, std::size_t> inferGridShape(std::size_t population_size) {
        if (population_size == 0) {
            throw std::invalid_argument("Population size cannot be zero.");
        }

        std::size_t rows = static_cast<std::size_t>(std::sqrt(static_cast<long double>(population_size)));
        while (rows > 1 && population_size % rows != 0) {
            --rows;
        }

        return {rows, population_size / rows};
    }

    void evaluatePopulation(Population<GeneType>& population) {
        const std::size_t population_size = population.size();

        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, population_size),
            [&](const tbb::blocked_range<std::size_t>& range) {
                for (std::size_t i = range.begin(); i < range.end(); ++i) {
                    const double score = fitness_function_m.evaluate(population[i].getGenotype());
                    population[i].setFitness(score);
                }
            }
        );
    }

    Individual<GeneType> evolveCell(
        const Population<GeneType>& population,
        std::size_t row,
        std::size_t col
    ) const {
        thread_local static std::random_device rd;
        thread_local static std::mt19937_64 gen(rd());
        thread_local static std::uniform_real_distribution<double> dist(0.0, 1.0);

        const Individual<GeneType>& current = population[row * cols_m + col];
        const Individual<GeneType>& neighbor = local_selection_m->select(population, row, col, rows_m, cols_m);

        Individual<GeneType> child1;
        Individual<GeneType> child2;

        if (dist(gen) < crossover_rate_m) {
            auto children = crossover_m->crossover(current, neighbor);
            child1 = std::move(children.first);
            child2 = std::move(children.second);
        } else {
            child1 = current;
            child2 = neighbor;
        }

        mutation_m->mutate(child1, mutation_rate_m);
        mutation_m->mutate(child2, mutation_rate_m);

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
    /**
     * @brief Constructs the cellular genetic algorithm with all required operators.
     *
     * @param ff         Fitness function used to evaluate individuals. Must outlive this object.
     * @param local_sel  Local neighbourhood selection strategy. Ownership is transferred.
     * @param mu         Mutation operator applied to produced offspring. Ownership is transferred.
     * @param cs         Crossover operator used to combine the current cell with a neighbour.
     *                   Ownership is transferred.
     * @param m_rate     Mutation probability in the range [0, 1].
     * @param c_rate     Crossover probability in the range [0, 1].
     * @param max_gen    Number of generations to run.
     * @param threads    Maximum number of TBB worker threads. If 0, TBB decides automatically.
     * @param elitism    If true, the current individual competes with both children and the best
     *                   one survives in the next generation.
     */
    CellularGA(
        FitnessFunction<GeneType>& ff,
        std::unique_ptr<LocalSelection<GeneType>> local_sel,
        std::unique_ptr<Mutation<GeneType>> mu,
        std::unique_ptr<Crossover<GeneType>> cs,
        double m_rate,
        double c_rate,
        std::size_t max_gen,
        std::size_t threads,
        bool elitism = false
    )
        : fitness_function_m(ff),
          local_selection_m(std::move(local_sel)),
          mutation_m(std::move(mu)),
          crossover_m(std::move(cs)),
          mutation_rate_m(m_rate),
          crossover_rate_m(c_rate),
          max_generations_m(max_gen),
          threads_m(threads),
          rows_m(0),
          cols_m(0),
          use_local_elitism_m(elitism) {}

    /**
     * @brief Runs the cellular evolutionary optimisation loop.
     *
     * The input population is interpreted as a 2D grid whose shape is inferred from
     * @p population.size(). The population is evolved in place and overwritten with
     * the final generation.
     *
     * @param population In/out population to evolve.
     * @note The population must be initialised before calling this method.
     * @note If the population is empty, the method returns immediately.
     */
    void run(Population<GeneType>& population) override {
        if (population.empty()) {
            return;
        }

        std::tie(rows_m, cols_m) = inferGridShape(population.size());

        Population<GeneType> new_population(population.size(), population.getNumGenes());
        std::unique_ptr<tbb::global_control> tbb_control;

        if (threads_m > 0) {
            tbb_control = std::make_unique<tbb::global_control>(
                tbb::global_control::max_allowed_parallelism,
                threads_m
            );
        }

        evaluatePopulation(population);

        for (std::size_t generation_idx = 0; generation_idx < max_generations_m; ++generation_idx) {
            this->notifyLoggers(generation_idx, population);

            tbb::parallel_for(
                tbb::blocked_range<std::size_t>(0, population.size()),
                [&](const tbb::blocked_range<std::size_t>& range) {
                    for (std::size_t i = range.begin(); i < range.end(); ++i) {
                        const std::size_t row = i / cols_m;
                        const std::size_t col = i % cols_m;

                        new_population[i] = evolveCell(population, row, col);
                    }
                }
            );

            std::swap(population, new_population);
        }

        this->notifyLoggers(max_generations_m, population);
    }
    };

} // namespace galib

#endif // CELLULAR_GA_H
