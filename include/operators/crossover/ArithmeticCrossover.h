#ifndef ARITHMETIC_CROSSOVER_H
#define ARITHMETIC_CROSSOVER_H

#pragma once

#include "Crossover.h"
#include <random>
#include <stdexcept>
#include <utility>

namespace galib {

    template <typename GeneType>
    class ArithmeticCrossover : public Crossover<GeneType> {
    public:
        std::pair<Individual<GeneType>, Individual<GeneType>>
        crossover(const Individual<GeneType>& parent1,
                  const Individual<GeneType>& parent2) override {

            if (parent1.size() != parent2.size()) {
                throw std::invalid_argument("Parents must have the same genotype size");
            }

            thread_local static std::random_device rd;
            thread_local static std::mt19937 gen(rd());
            thread_local static std::uniform_real_distribution<double> alpha_dist(0.0, 1.0);

            double alpha = alpha_dist(gen);

            Individual<GeneType> child1(parent1.size());
            Individual<GeneType> child2(parent2.size());

            const auto& g1 = parent1.getGenotype();
            const auto& g2 = parent2.getGenotype();

            auto& c1 = child1.getGenotype();
            auto& c2 = child2.getGenotype();

            for (std::size_t i = 0; i < g1.size(); ++i) {
                c1[i] = static_cast<GeneType>(alpha * g1[i] + (1.0 - alpha) * g2[i]);
                c2[i] = static_cast<GeneType>(alpha * g2[i] + (1.0 - alpha) * g1[i]);
            }

            return {child1, child2};
        }
    };

}

#endif // ARITHMETIC_CROSSOVER_H