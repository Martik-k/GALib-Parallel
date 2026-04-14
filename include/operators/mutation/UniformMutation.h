#ifndef UNIFORM_MUTATION_H
#define UNIFORM_MUTATION_H

#pragma once

#include "Mutation.h"
#include <random>
#include <vector>
#include <cstddef>

namespace galib {

    template <typename GeneType>
    class UniformMutation : public Mutation<GeneType> {
    private:
        GeneType lower_m;
        GeneType upper_m;

    public:
        UniformMutation(GeneType lower, GeneType upper)
            : lower_m(lower), upper_m(upper) {}

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