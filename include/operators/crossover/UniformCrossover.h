#ifndef UNIFORM_CROSSOVER_H
#define UNIFORM_CROSSOVER_H

#pragma once

#include "Crossover.h"
#include <random>
#include <stdexcept>
#include <utility>

namespace galib {

    /**
     * @brief Performs uniform crossover.
     * 
     * For every gene position, there is a 50% chance to take the gene from 
     * Parent 1 and a 50% chance to take it from Parent 2.
     * 
     * @tparam GeneType The numeric type of the genes.
     */
    template <typename GeneType>
    class UniformCrossover : public Crossover<GeneType> {
    public:
        /**
         * @brief Randomly picks genes from either parent at each position.
         * @param parent1 First parent.
         * @param parent2 Second parent.
         * @return Pair of children.
         * @throws std::invalid_argument if parents have different genotype sizes.
         */
        std::pair<Individual<GeneType>, Individual<GeneType>>
        crossover(const Individual<GeneType>& parent1,
                  const Individual<GeneType>& parent2) override {

            if (parent1.size() != parent2.size()) {
                throw std::invalid_argument("Parents must have the same genotype size");
            }

            thread_local static std::random_device rd;
            thread_local static std::mt19937 gen(rd());
            thread_local static std::bernoulli_distribution pick_first(0.5);

            Individual<GeneType> child1(parent1.size());
            Individual<GeneType> child2(parent2.size());

            const auto& g1 = parent1.getGenotype();
            const auto& g2 = parent2.getGenotype();

            auto& c1 = child1.getGenotype();
            auto& c2 = child2.getGenotype();

            for (std::size_t i = 0; i < g1.size(); ++i) {
                if (pick_first(gen)) {
                    c1[i] = g1[i];
                    c2[i] = g2[i];
                } else {
                    c1[i] = g2[i];
                    c2[i] = g1[i];
                }
            }

            return {child1, child2};
        }
    };

}

#endif // UNIFORM_CROSSOVER_Hдавай так рідмі потім перепише спочатку 