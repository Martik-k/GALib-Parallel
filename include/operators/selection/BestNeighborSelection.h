#ifndef BEST_NEIGHBOR_SELECTION_H
#define BEST_NEIGHBOR_SELECTION_H

#pragma once

#include "LocalSelection.h"
#include <stdexcept>

namespace galib {

template <typename GeneType>
class BestNeighborSelection : public LocalSelection<GeneType> {
public:
    const Individual<GeneType>& select(
        const GridPopulation<GeneType>& population,
        std::size_t row,
        std::size_t col
    ) const override {
        auto neighbors = population.getNeighbors(row, col);

        if (neighbors.empty()) {
            throw std::runtime_error("No neighbors available for local selection");
        }

        const Individual<GeneType>* best = nullptr;

        for (const auto& [nr, nc] : neighbors) {
            const Individual<GeneType>& candidate = population.at(nr, nc);

            if (best == nullptr || candidate.getFitness() < best->getFitness()) {
                best = &candidate;
            }
        }

        return *best;
    }
};

} // namespace galib

#endif // BEST_NEIGHBOR_SELECTION_H