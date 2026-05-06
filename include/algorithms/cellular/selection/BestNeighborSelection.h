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
        std::size_t col
    ) const override {
        // Placeholder: Since we don't have Grid info here yet, 
        // we just return the individual itself or a simple neighbor.
        // The user mentioned CellularGA will be updated in the future.
        return population[0]; // Dummy implementation to ensure compilation
    }
};

} // namespace galib

#endif // BEST_NEIGHBOR_SELECTION_H
