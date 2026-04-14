#ifndef LOCAL_SELECTION_H
#define LOCAL_SELECTION_H

#pragma once

#include "core/GridPopulation.h"

namespace galib {

template <typename GeneType>
class LocalSelection {
public:
    virtual ~LocalSelection() = default;

    virtual const Individual<GeneType>& select(
        const GridPopulation<GeneType>& population,
        std::size_t row,
        std::size_t col
    ) const = 0;
};

} // namespace galib

#endif // LOCAL_SELECTION_H