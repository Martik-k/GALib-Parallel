#ifndef BOUNDARY_MUTATION_H
#define BOUNDARY_MUTATION_H

#pragma once

#include "Mutation.h"
#include <random>
#include <vector>
#include <cstddef>

namespace galib {

    /**
     * @brief Performs boundary mutation.
     * 
     * Replaces a gene with either the lower or upper bound of the search space.
     * This is useful for exploring the edges of the feasible region.
     * 
     * @tparam GeneType The numeric type of the genes.
     */
    template <typename GeneType>
    class BoundaryMutation : public Mutation<GeneType> {
    private:
        GeneType lower_m;
        GeneType upper_m;

    public:
        /**
         * @brief Constructs a boundary mutation operator.
         * @param lower The lower bound of the search space.
         * @param upper The upper bound of the search space.
         */
        BoundaryMutation(GeneType lower, GeneType upper)
            : lower_m(lower), upper_m(upper) {}

        /**
         * @brief Randomly resets genes to @p lower or @p upper bounds.
         * @param individual    Target individual.
         * @param mutation_rate Probability per gene.
         */
        void mutate(Individual<GeneType>& individual, double mutation_rate) override {
            if (mutation_rate <= 0.0) return;

            thread_local static std::random_device rd;
            thread_local static std::mt19937 gen(rd());
            std::uniform_real_distribution<double> prob(0.0, 1.0);
            std::uniform_int_distribution<int> coin(0, 1);

            std::vector<GeneType>& genotype = individual.getGenotype();

            for (std::size_t i = 0; i < genotype.size(); ++i) {
                if (prob(gen) < mutation_rate) {
                    genotype[i] = (coin(gen) == 0) ? lower_m : upper_m;
                }
            }
        }
    };

}

#endif // BOUNDARY_MUTATION_H