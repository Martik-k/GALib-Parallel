#ifndef MUTATION_H
#define MUTATION_H

#pragma once

#include "core/Individual.h"

namespace galib {

    /**
     * @brief Interface for mutation operators.
     * 
     * Mutation introduces small, random variations into an individual's genotype
     * to maintain genetic diversity and prevent premature convergence.
     * 
     * @tparam GeneType The numeric type of the genes.
     */
    template <typename GeneType>
    class Mutation {
    public:
        virtual ~Mutation() = default;

        /**
         * @brief Applies mutation to an individual.
         * 
         * @param individual    The individual to mutate (modified in-place).
         * @param mutation_rate Probability of mutation for each gene [0.0, 1.0].
         */
        virtual void mutate(Individual<GeneType>& individual, double mutation_rate) = 0;
    };

}

#endif // MUTATION_H