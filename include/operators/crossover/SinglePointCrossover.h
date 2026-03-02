#ifndef SINGLE_POINT_CROSSOVER_H
#define SINGLE_POINT_CROSSOVER_H

#pragma once

#include "Crossover.h"
#include "../../core/Individual.h"
#include <stdexcept>
#include <random>
#include <utility>


namespace galib {

    template <typename GeneType>
    class SinglePointCrossover : public Crossover<GeneType> {
    private:
        std::random_device rd_m;
        std::mt19937 gen_m;
    public:
        explicit SinglePointCrossover() : rd_m(), gen_m(rd_m()) {}
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
                std::uniform_int_distribution<std::size_t> distribution(1, genotype_size - 1);
                std::size_t crossover_point = distribution(gen_m);

                std::vector<GeneType>& child1_genotype = child1.getGenotype();
                std::vector<GeneType>& child2_genotype = child2.getGenotype();

                for (size_t i = crossover_point; i < genotype_size; ++i) {
                    child1_genotype[i] = parent2_genotype[i];
                    child2_genotype[i] = parent1_genotype[i];
                }
            }
            return std::make_pair(child1, child2);
        }
    };
}

#endif // SINGLE_POINT_CROSSOVER_H
