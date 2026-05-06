#ifndef UNIFORM_MUTATION_H
#define UNIFORM_MUTATION_H

#pragma once

#include "Mutation.h"
#include <random>
#include <vector>
#include <cstddef>

namespace galib {

    /**
     * @brief Performs uniform mutation.
     * 
     * Replaces a gene with a new value sampled uniformly from the range [lower, upper].
     * 
     * @tparam GeneType The numeric type of the genes.
     */
    template <typename GeneType>
    class UniformMutation : public Mutation<GeneType> {
    private:
        GeneType lower_m;
        GeneType upper_m;

    public:
        /**
         * @brief Constructs the mutation operator.
         * @param lower Lower bound for new random values.
         * @param upper Upper bound for new random values.
         */
        UniformMutation(GeneType lower, GeneType upper)
            : lower_m(lower), upper_m(upper) {}

        /**
         * @brief Replaces genes with random values from [lower, upper].
         * @param individual    Target individual.
         * @param mutation_rate Probability per gene.
         */
        void mutate(Individual<GeneType>& individual, double mutation_rate) override {
            if (mutation_rate <= 0.0) return;

            thread_local static std::random_device rd;
            thread_local static std::mt19937 gen(rd());
            std::uniform_real_distribution<GeneType> value_dist(lower_m, upper_m);
            std::uniform_real_distribution<double> prob(0.0, 1.0);

            std::vector<GeneType>& genotype = individual.getGenotype();

            for (std::size_t i = 0; i < genotype.size(); ++i) {
                if (prob(gen) < mutation_rate) {
                    genotype[i] = value_dist(gen);
                }
            }
        }
    };

}

#endif // UNIFORM_MUTATION_H