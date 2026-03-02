#ifndef GAUSSIAN_MUTATION_H
#define GAUSSIAN_MUTATION_H

#include "Mutation.h"
#include <random>

namespace galib {

    template <typename GeneType>
    class GaussianMutation : public Mutation<GeneType> {
    private:
        double sigma_m;
        std::random_device rd_m;
        std::mt19937 gen_m;
    public:
        explicit GaussianMutation(double sigma = 0.1)
            : sigma_m(sigma), rd_m(), gen_m(rd_m()) {}
        void mutate(Individual<GeneType>& individual, double mutation_rate) override {
            if (mutation_rate <= 0.0) { return; }

            std::vector<GeneType>& genotype = individual.getGenotype();

            std::uniform_real_distribution<double> distribution(0.0, 1.0);
            std::normal_distribution<double> gaussian_distribution(0.0, sigma_m);

            for (std::size_t i = 0; i < genotype.size(); ++i) {
                if (distribution(gen_m) < mutation_rate) {
                    genotype[i] += gaussian_distribution(gen_m);
                }
            }
        }
    };

}

#endif // GAUSSIAN_MUTATION_H
