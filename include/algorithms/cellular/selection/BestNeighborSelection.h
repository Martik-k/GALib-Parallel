#ifndef BEST_NEIGHBOR_SELECTION_H
#define BEST_NEIGHBOR_SELECTION_H

#pragma once

#include "algorithms/cellular/selection/LocalSelection.h"
#include <stdexcept>

namespace galib {

template <typename GeneType>
class BestNeighborSelection : public LocalSelection<GeneType> {
public:
    const Individual<GeneType>& select(
        const Population<GeneType>& population,
        std::size_t row,
        std::size_t col,
        std::size_t rows,
        std::size_t cols
    ) const override {
        if (rows == 0 || cols == 0) {
            throw std::invalid_argument("BestNeighborSelection requires non-zero grid dimensions.");
        }

        const std::size_t up = (row == 0) ? rows - 1 : row - 1;
        const std::size_t down = (row + 1) % rows;
        const std::size_t left = (col == 0) ? cols - 1 : col - 1;
        const std::size_t right = (col + 1) % cols;

        const Individual<GeneType>* best = &population[up * cols + col];

        const std::size_t candidates[] = {
            down * cols + col,
            row * cols + left,
            row * cols + right
        };

        for (const std::size_t idx : candidates) {
            const Individual<GeneType>& candidate = population[idx];
            if (candidate.getFitness() < best->getFitness()) {
                best = &candidate;
            }
        }

        return *best;
    }
};

} // namespace galib

#endif // BEST_NEIGHBOR_SELECTION_H
