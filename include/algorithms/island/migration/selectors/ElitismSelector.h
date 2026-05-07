#ifndef ELITISM_SELECTOR_H
#define ELITISM_SELECTOR_H

#pragma once

#include "algorithms/island/migration/selectors/DemeSelector.h"
#include "core/Individual.h"
#include "core/Population.h"
#include <vector>
#include <numeric>
#include <algorithm>

namespace galib {

    /**
     * @brief Selection strategy that chooses the best individuals for migration.
     * 
     * Picks the top @p count fittest individuals from the local population
     * to be shared with neighbors.
     * 
     * @tparam GeneType The gene type of the individuals.
     */
    template <typename GeneType>
    class ElitismSelector : public DemeSelector<GeneType> {
    public:
        /**
         * @brief Selects the elite members of the population.
         * 
         * @param population Source population.
         * @param count      Number of elites to select.
         * @return Vector of selected elite individuals.
         */
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
