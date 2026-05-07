#ifndef DEME_REPLACER_H
#define DEME_REPLACER_H

#pragma once

#include "core/Individual.h"
#include "core/Population.h"
#include <vector>

namespace galib {

    /**
     * @brief Interface for integrating incoming migrants into the local population.
     * 
     * Implementations define the strategy for which local individuals are replaced
     * when a "deme" (sub-population) arrives from another island.
     * 
     * @tparam GeneType The gene type of the individuals.
     */
    template<typename GeneType>
    class DemeReplacer {
    public:
        virtual ~DemeReplacer() = default;

        /**
         * @brief Replaces a portion of the local population with the given migrants.
         * 
         * @param population The local population to modify.
         * @param deme       The incoming individuals (migrants).
         */
        virtual void replaceDeme(
            Population<GeneType>& population,
            std::vector<Individual<GeneType>>&& deme
        ) const = 0;
    };

}

#endif // DEME_REPLACER_H
