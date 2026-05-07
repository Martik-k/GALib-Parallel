#ifndef LOCAL_SELECTION_H
#define LOCAL_SELECTION_H

#pragma once

#include "core/Population.h"

namespace galib {

/**
 * @brief Abstract interface for neighbourhood-based parent selection in a cellular GA.
 *
 * Implementations choose one individual for the current cell using only the
 * local neighbourhood of that cell in the inferred 2D grid.
 *
 * @tparam GeneType Numeric type of each gene.
 */
template <typename GeneType>
class LocalSelection {
public:
    /**
     * @brief Virtual destructor for polymorphic use.
     */
    virtual ~LocalSelection() = default;

    /**
     * @brief Selects one individual from the neighbourhood of a grid cell.
     *
     * @param population Population interpreted as a 2D grid in row-major order.
     * @param row        Row of the current cell.
     * @param col        Column of the current cell.
     * @param rows       Total number of grid rows.
     * @param cols       Total number of grid columns.
     * @return Const reference to the selected neighbouring individual.
     * @note Lower fitness values are considered better.
     */
    virtual const Individual<GeneType>& select(
        const Population<GeneType>& population,
        std::size_t row,
        std::size_t col,
        std::size_t rows,
        std::size_t cols
    ) const = 0;
};

} // namespace galib

#endif // LOCAL_SELECTION_H
