#ifndef CROSSOVER_H
#define CROSSOVER_H

#pragma once

#include "core/Individual.h"
#include <utility>

namespace galib {

    /**
     * @brief Interface for crossover (recombination) operators.
     * 
     * Crossover takes two parent individuals and produces two children by 
     * combining their genetic material.
     * 
     * @tparam GeneType The numeric type of the genes.
     */
    template <typename GeneType>
    class Crossover {
    public:
        virtual ~Crossover() = default;

        /**
         * @brief Performs crossover between two parents.
         * 
         * @param parent1 The first parent.
         * @param parent2 The second parent.
         * @return A pair containing the two resulting children.
         */
        virtual std::pair<Individual<GeneType>, Individual<GeneType>> crossover (
            const Individual<GeneType>& parent1, const Individual<GeneType>& parent2) = 0;
    };

}

#endif // CROSSOVER_H