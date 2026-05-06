#ifndef SINGLE_POINT_CROSSOVER_H
#define SINGLE_POINT_CROSSOVER_H

#pragma once

#include "Crossover.h"
#include "core/Individual.h"
#include <stdexcept>
#include <random>
#include <utility>
#include <algorithm>

namespace galib {

    /**
     * @brief Performs traditional single-point crossover.
     * 
     * A random crossover point is chosen, and the genetic material after 
     * that point is swapped between the two parents to create two children.
     * 
     * @tparam GeneType The numeric type of the genes.
     */
    template <typename GeneType>
    class SinglePointCrossover : public Crossover<GeneType> {
    public:
        SinglePointCrossover() = default;

        /**
         * @brief Swaps genotype segments at a random crossover point.
         * @param parent1 First parent.
         * @param parent2 Second parent.
         * @return Pair of children.
         * @throws std::invalid_argument if parents have different genotype sizes.
         */
        std::pair<Individual<GeneType>, Individual<GeneType>> crossover (
            const Individual<GeneType>& parent1, const Individual<GeneType>& parent2) override {

            const std::vector<GeneType>& parent1_genotype = parent1.getGenotype();
            const std::vector<GeneType>& parent2_genotype = parent2.getGenotype();

            if (parent1_genotype.size() != parent2_genotype.size()) {
                throw std::invalid_argument("Parents must have the same genotype length for crossover.");
            }
            std::size_t genotype_size = parent1_genotype.size();

            Individual<GeneType> child1 = parent1;
            Individual<GeneType> child2 = parent2;

            if (genotype_size > 1) {
				thread_local static std::random_device rd;
                thread_local static std::mt19937 gen(rd());

                std::uniform_int_distribution<std::size_t> distribution(1, genotype_size - 1);
                std::size_t crossover_point = distribution(gen);

                std::vector<GeneType>& child1_genotype = child1.getGenotype();
                std::vector<GeneType>& child2_genotype = child2.getGenotype();

                std::swap_ranges(
                    child1_genotype.begin() + crossover_point,
                    child1_genotype.end(),
                    child2_genotype.begin() + crossover_point
                );
            }
            return std::make_pair(child1, child2);
        }
    };
}

#endif // SINGLE_POINT_CROSSOVER_H
