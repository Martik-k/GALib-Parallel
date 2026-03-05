#ifndef GAUSSIAN_MUTATION_H
#define GAUSSIAN_MUTATION_H

#pragma once

#include "Mutation.h"
#include <random>
#include <vector>
#include <cstddef>

namespace galib {

    template <typename GeneType>
    class GaussianMutation : public Mutation<GeneType> {
    private:
        double sigma_m;
    public:
        explicit GaussianMutation(double sigma)
            : sigma_m(sigma) {}
        void mutate(Individual<GeneType>& individual, double mutation_rate) override {
            if (mutation_rate <= 0.0) { return; }

			thread_local static std::random_device rd;
            thread_local static std::mt19937 gen(rd());
            thread_local static std::uniform_real_distribution<double> distribution(0.0, 1.0);
			thread_local static std::normal_distribution<double> gaussian_distribution;

            std::vector<GeneType>& genotype = individual.getGenotype();

            for (std::size_t i = 0; i < genotype.size(); ++i) {
                if (distribution(gen) < mutation_rate) {
                    genotype[i] += gaussian_distribution(gen, std::normal_distribution<double>::param_type(0.0, sigma_m));
                }
            }
        }
    };

}

#endif // GAUSSIAN_MUTATION_H
