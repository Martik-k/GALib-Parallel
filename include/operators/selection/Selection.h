#ifndef SELECTION_H
#define SELECTION_H

#pragma once

#include "core/Individual.h"
#include "core/Population.h"

namespace galib {

    /**
     * @brief Interface for selection operators.
     * 
     * Selection operators choose individuals from a population to act as parents
     * for the next generation, typically biasing towards more fit individuals.
     * 
     * @tparam GeneType The numeric type of the genes.
     */
    template <typename GeneType>
    class Selection {
    public:
        virtual ~Selection() = default;

        /**
         * @brief Selects one individual from the population.
         * @param population The source population.
         * @return Constant reference to the selected individual.
         */
        virtual const Individual<GeneType>& select(const Population<GeneType>& population) = 0;
    };

}

#endif // SELECTION_H