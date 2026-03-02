#ifndef TOURNAMENT_SELECTION_H
#define TOURNAMENT_SELECTION_H

#pragma once

#include "Selection.h"
#include <random>
#include <algorithm>

namespace galib {

    template <typename GeneType>
    class TournamentSelection : public Selection<GeneType> {
    private:
        size_t tournament_size_m;
        std::random_device rd_m;
        std::mt19937 gen_m;
    public:
        explicit TournamentSelection(size_t tournament_size = 3) :
            tournament_size_m(tournament_size), rd_m(), gen_m(rd_m()) {}
        const Individual<GeneType>& select(const Population<GeneType>& population) override {

            if (tournament_size_m == 0) {
                throw std::invalid_argument("Tournament size must be at least 1.");
            }

            std::size_t pop_size = population.size();

            std::uniform_int_distribution<std::size_t> distribution(0, pop_size - 1);

            const Individual<GeneType>* winner = nullptr;

            for (std::size_t i = 0; i < tournament_size_m; ++i) {
                const Individual<GeneType>& candidate = population[distribution(gen_m)];

                if (winner == nullptr || candidate.getFitness() < winner->getFitness()) {
                    winner = &candidate;
                }
            }
            return *winner;
        }
    };

}

#endif // TOURNAMENT_SELECTION_H
