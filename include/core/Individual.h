#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#pragma once

#include <vector>
#include <limits>

namespace galib {

    template <typename GeneType = double>
    class Individual {
    private:
        std::vector<GeneType> genotype;
        double fitness;
    public:
        Individual() : fitness(std::numeric_limits<double>::max()) {}

        explicit Individual(std::size_t num_genes)
            : genotype(num_genes), fitness(std::numeric_limits<double>::max()) {}

        std::size_t size() const { return genotype.size(); }

        std::vector<GeneType>& getGenotype() { return genotype; }
        const std::vector<GeneType>& getGenotype() const { return genotype; }
        void setGenotype(const std::vector<GeneType>& new_genotype) { genotype = new_genotype; }

        double getFitness() const { return fitness; }
        void setFitness(double new_fitness) { fitness = new_fitness; }

        bool operator <(const Individual& other) const {
            return fitness < other.fitness;
        }
    };

}

#endif // INDIVIDUAL_H
