#ifndef WORST_REPLACER_H
#define WORST_REPLACER_H

#pragma once

#include "algorithms/island/migration/replacers/DemeReplacer.h"
#include "core/Individual.h"
#include "core/Population.h"
#include <vector>
#include <numeric>
#include <algorithm>

namespace galib {

    template<typename GeneType>
    class WorstReplacer : public DemeReplacer<GeneType> {
    public:
        void replaceDeme(
            Population<GeneType>& population,
            const std::vector<Individual<GeneType>>& deme
        ) const override {
            const std::size_t count = std::min(deme.size(), population.size());

            std::vector<std::size_t> indices(population.size());

            std::iota(indices.begin(), indices.end(), 0);

            std::ranges::nth_element(
                indices,
                indices.begin() + count,
                std::greater{},
                [&](const std::size_t i) { return population[i]; }
            );

            for (std::size_t i = 0; i < count; ++i) {
                population[indices[i]] = deme[i];
            }
        }
    };

}

#endif //WORST_REPLACER_H
