#ifndef TOURNAMENT_SELECTION_H
#define TOURNAMENT_SELECTION_H

#pragma once

#include "Selection.h"
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cstddef>

namespace galib {

    /**
     * @brief Performs k-way tournament selection.
     * 
     * Randomly samples @p k individuals from the population and returns the one 
     * with the best fitness. Higher tournament sizes result in higher selection pressure.
     * 
     * @tparam GeneType The numeric type of the genes.
     */
    template <typename GeneType>
    class TournamentSelection : public Selection<GeneType> {
    private:
        std::size_t tournament_size_m;
    public:
        /**
         * @brief Constructs the selector.
         * @param tournament_size Number of candidates per tournament. Must be >= 1.
         * @throws std::invalid_argument if tournament_size is 0.
         */
        explicit TournamentSelection(size_t tournament_size) : tournament_size_m(tournament_size) {
            if (tournament_size_m == 0) {
                throw std::invalid_argument("Tournament size must be at least 1.");
            }
        }

        std::size_t getTournamentSize() const { return tournament_size_m; }

        /**
         * @brief Conducts a tournament and returns the winner.
         * @param population Source population.
         * @return Reference to the fittest individual in the sampled group.
         */
        const Individual<GeneType>& select(const Population<GeneType>& population) override {

            thread_local static std::random_device rd;
            thread_local static std::mt19937 gen(rd());

            std::size_t pop_size = population.size();
            std::uniform_int_distribution<std::size_t> distribution(0, pop_size - 1);

            const Individual<GeneType>* winner = &population[distribution(gen)];

            for (std::size_t i = 1; i < tournament_size_m; ++i) {
                const Individual<GeneType>& candidate = population[distribution(gen)];

                if (candidate.getFitness() < winner->getFitness()) {
                    winner = &candidate;
                }
            }
            return *winner;
        }
    };

}

#endif // TOURNAMENT_SELECTION_H
