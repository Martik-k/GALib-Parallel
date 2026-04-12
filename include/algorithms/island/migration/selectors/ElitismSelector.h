#ifndef ELITISM_SELECTOR_H
#define ELITISM_SELECTOR_H

#pragma once

#include "operators/migration/selectors/DemeSelector.h"
#include "core/Individual.h"
#include "core/Population.h"
#include <vector>
#include <numeric>
#include <algorithm>

namespace galib {

    template <typename GeneType>
    class ElitismSelector : public DemeSelector<GeneType> {
    public:
        std::vector<Individual<GeneType>> selectDeme(
            const Population<GeneType>& population,
            std::size_t count
        ) const override {
            std::vector<std::size_t> indices(population.size());

            std::iota(indices.begin(), indices.end(), 0);

            count = std::min(count, population.size());

            std::ranges::nth_element(
                indices,
                indices.begin() + count,
                std::less{},
                [&](const std::size_t i) -> const Individual<GeneType>& { return population[i]; }
            );

            std::vector<Individual<GeneType>> selectedDeme;
            selectedDeme.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                selectedDeme.push_back(population[indices[i]]);
            }

            return selectedDeme;
        }
    };

}

#endif // ELITISM_SELECTOR_H
