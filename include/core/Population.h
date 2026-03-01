#ifndef POPULATION_H
#define POPULATION_H

#pragma once

#include "Individual.h"
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace galib {

    template <typename GeneType>
    class Population {
    private:
        std::vector<Individual<GeneType>> individuals;
    public:
        explicit Population(std::size_t size, std::size_t num_genes) {
            if (size == 0) throw std::invalid_argument("Population size cannot be zero");
            individuals.assign(size, Individual<GeneType>(num_genes));
        }

        std::size_t size() const { return individuals.size(); }

        Individual<GeneType>& operator [](std::size_t index) { return individuals[index]; }
        const Individual<GeneType>& operator [](std::size_t index) const { return individuals[index]; }

        typename std::vector<Individual<GeneType>>::iterator begin() { return individuals.begin(); }
        typename std::vector<Individual<GeneType>>::iterator end() { return individuals.end(); }
        typename std::vector<Individual<GeneType>>::const_iterator begin() const { return individuals.begin(); }
        typename std::vector<Individual<GeneType>>::const_iterator end() const { return individuals.end(); }

        const Individual<GeneType>& getBestIndividual() const {
            return *std::min_element(individuals.begin(), individuals.end());
        }
    };
}

#endif // POPULATION_H
