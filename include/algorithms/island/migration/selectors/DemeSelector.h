#ifndef DEME_SELECTOR_H
#define DEME_SELECTOR_H

#pragma once

#include "core/Individual.h"
#include "core/Population.h"
#include <vector>

namespace galib {

    /**
     * @brief Interface for choosing individuals to migrate out of the island.
     * 
     * Implementations define which members of the local population are selected
     * to be sent to neighboring islands.
     * 
     * @tparam GeneType The gene type of the individuals.
     */
    template<typename GeneType>
    class DemeSelector {
    public:
        virtual ~DemeSelector() = default;

        /**
         * @brief Selects individuals for migration.
         * 
         * @param population The local population to select from.
         * @param count      The number of individuals to select.
         * @return A vector containing copies of the selected individuals.
         */
        virtual std::vector<Individual<GeneType>> selectDeme(
            const Population<GeneType>& population,
            std::size_t count
        ) const = 0;
    };

}

#endif // DEME_SELECTOR_H
